#include "layers.h"

namespace rnet {

// ─── Slot Memory: Top-K Sparse Cross-Attention ──────────────────────
// 1. query = x @ proj_q^T          [batch*seq, dim]
// 2. scores = query @ slot_keys     [batch*seq, n_slots]
// 3. top-k selection (k=2)
// 4. softmax over top-k scores
// 5. retrieved = weighted sum of slot_values at top-k indices
// 6. output = retrieved @ proj_out^T

// ─── Top-K + Softmax + Retrieval (fused) ────────────────────────────
__global__ void topk_retrieve_kernel(
        const float* __restrict__ scores,      // [batch_seq, n_slots]
        const float* __restrict__ slot_values, // [dim, n_slots]
        float* __restrict__ retrieved,         // [batch_seq, dim]
        int* __restrict__ topk_idx,            // [batch_seq, top_k]
        float* __restrict__ topk_weights,      // [batch_seq, top_k]
        int n_slots, int top_k, int dim) {
    int row = blockIdx.x;  // which token
    const float* sr = scores + (size_t)row * n_slots;
    float* rr = retrieved + (size_t)row * dim;
    int* ti = topk_idx + row * top_k;
    float* tw = topk_weights + row * top_k;

    // Thread 0 does top-k selection (k is small, typically 2)
    __shared__ int s_idx[8];     // max top_k = 8
    __shared__ float s_val[8];
    __shared__ float s_weight[8];

    if (threadIdx.x == 0) {
        // Initialize with -inf
        for (int k = 0; k < top_k; k++) {
            s_idx[k] = 0;
            s_val[k] = -1e30f;
        }

        // Find top-k
        for (int i = 0; i < n_slots; i++) {
            float v = sr[i];
            // Check if v is larger than smallest in top-k
            int min_k = 0;
            for (int k = 1; k < top_k; k++) {
                if (s_val[k] < s_val[min_k]) min_k = k;
            }
            if (v > s_val[min_k]) {
                s_val[min_k] = v;
                s_idx[min_k] = i;
            }
        }

        // Softmax over top-k scores
        float max_s = s_val[0];
        for (int k = 1; k < top_k; k++) max_s = fmaxf(max_s, s_val[k]);
        float sum_exp = 0.0f;
        for (int k = 0; k < top_k; k++) {
            s_val[k] = expf(s_val[k] - max_s);
            sum_exp += s_val[k];
        }
        for (int k = 0; k < top_k; k++) {
            s_weight[k] = s_val[k] / sum_exp;
            ti[k] = s_idx[k];
            tw[k] = s_weight[k];
        }
    }
    __syncthreads();

    // Weighted retrieval: retrieved[d] = sum_k weight[k] * slot_values[d, idx[k]]
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < top_k; k++) {
            acc += s_weight[k] * slot_values[(size_t)d * n_slots + s_idx[k]];
        }
        rr[d] = acc;
    }
}

void slot_memory_forward(const float* x, float* y,
                         const SlotMemoryParams& p,
                         float* q_cache, float* scores_cache,
                         int* topk_idx_cache, float* topk_w_cache,
                         int batch, int seq_len, int dim,
                         int n_slots, int top_k,
                         cublasHandle_t cublas, cudaStream_t stream) {
    int total_tokens = batch * seq_len;
    float alpha = 1.0f, beta_val = 0.0f;
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    // 1. query = x @ proj_q^T  [total_tokens, dim]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             dim, total_tokens, dim,
                             &alpha, p.proj_q, dim, x, dim,
                             &beta_val, q_cache, dim));

    // 2. scores = query @ slot_keys  [total_tokens, n_slots]
    //    slot_keys is [dim, n_slots]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             n_slots, total_tokens, dim,
                             &alpha, p.slot_keys, n_slots, q_cache, dim,
                             &beta_val, scores_cache, n_slots));

    // 3-5. Top-k + softmax + retrieval (fused kernel)
    float* retrieved;
    CUDA_CHECK(cudaMalloc(&retrieved, (size_t)total_tokens * dim * sizeof(float)));

    topk_retrieve_kernel<<<total_tokens, 256, 0, stream>>>(
        scores_cache, p.slot_values, retrieved,
        topk_idx_cache, topk_w_cache,
        n_slots, top_k, dim);

    // 6. output = retrieved @ proj_out^T  [total_tokens, dim]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             dim, total_tokens, dim,
                             &alpha, p.proj_out, dim, retrieved, dim,
                             &beta_val, y, dim));

    CUDA_CHECK(cudaFree(retrieved));
}

// ─── Backward ───────────────────────────────────────────────────────
__global__ void topk_retrieve_bwd_kernel(
        const float* __restrict__ d_retrieved,  // [batch_seq, dim]
        const float* __restrict__ slot_values,  // [dim, n_slots]
        const int* __restrict__ topk_idx,       // [batch_seq, top_k]
        const float* __restrict__ topk_weights, // [batch_seq, top_k]
        float* __restrict__ d_slot_values,      // [dim, n_slots]
        float* __restrict__ d_scores,           // [batch_seq, n_slots] (sparse, only top-k)
        int n_slots, int top_k, int dim) {
    int row = blockIdx.x;
    const float* dr = d_retrieved + (size_t)row * dim;
    const int* ti = topk_idx + row * top_k;
    const float* tw = topk_weights + row * top_k;
    float* ds_row = d_scores + (size_t)row * n_slots;

    // Zero d_scores for this row
    for (int i = threadIdx.x; i < n_slots; i += blockDim.x) {
        ds_row[i] = 0.0f;
    }
    __syncthreads();

    // d_slot_values[d, idx[k]] += weight[k] * d_retrieved[d]
    // d_weight[k] = sum_d d_retrieved[d] * slot_values[d, idx[k]]
    __shared__ float s_dw[8];  // d_weight for top_k
    if (threadIdx.x < top_k) s_dw[threadIdx.x] = 0.0f;
    __syncthreads();

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        for (int k = 0; k < top_k; k++) {
            int slot_idx = ti[k];
            atomicAdd(&d_slot_values[(size_t)d * n_slots + slot_idx],
                      tw[k] * dr[d]);
            // Accumulate d_weight[k] (this thread's contribution)
            float contrib = dr[d] * slot_values[(size_t)d * n_slots + slot_idx];
            atomicAdd(&s_dw[k], contrib);
        }
    }
    __syncthreads();

    // Convert d_weight to d_scores through softmax backward
    // softmax backward: ds[k] = w[k] * (dw[k] - sum_j w[j]*dw[j])
    if (threadIdx.x == 0) {
        float dot = 0.0f;
        for (int k = 0; k < top_k; k++) dot += tw[k] * s_dw[k];
        for (int k = 0; k < top_k; k++) {
            ds_row[ti[k]] = tw[k] * (s_dw[k] - dot);
        }
    }
}

void slot_memory_backward(const float* dy, const float* x,
                          const SlotMemoryParams& p,
                          const float* q_cache, const float* scores_cache,
                          const int* topk_idx_cache, const float* topk_w_cache,
                          float* dx, SlotMemoryParams& dp,
                          int batch, int seq_len, int dim,
                          int n_slots, int top_k,
                          cublasHandle_t cublas, cudaStream_t stream) {
    int total_tokens = batch * seq_len;
    float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    // d_retrieved = dy @ proj_out  [total_tokens, dim]
    float *d_retrieved, *d_scores;
    CUDA_CHECK(cudaMalloc(&d_retrieved, (size_t)total_tokens * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, (size_t)total_tokens * n_slots * sizeof(float)));

    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             dim, total_tokens, dim,
                             &alpha, p.proj_out, dim, dy, dim,
                             &beta_zero, d_retrieved, dim));

    // d_proj_out += retrieved^T @ dy (need to recompute retrieved)
    // For now, recompute retrieved
    float* retrieved;
    CUDA_CHECK(cudaMalloc(&retrieved, (size_t)total_tokens * dim * sizeof(float)));
    topk_retrieve_kernel<<<total_tokens, 256, 0, stream>>>(
        scores_cache, p.slot_values, retrieved,
        (int*)topk_idx_cache, (float*)topk_w_cache,
        n_slots, top_k, dim);

    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                             dim, dim, total_tokens,
                             &alpha, retrieved, dim, dy, dim,
                             &beta_one, dp.proj_out, dim));

    // Backward through top-k retrieval
    topk_retrieve_bwd_kernel<<<total_tokens, 256, 0, stream>>>(
        d_retrieved, p.slot_values, topk_idx_cache, topk_w_cache,
        dp.slot_values, d_scores, n_slots, top_k, dim);

    // d_scores → d_query: d_query = d_scores @ slot_keys^T  [total_tokens, dim]
    float* d_query;
    CUDA_CHECK(cudaMalloc(&d_query, (size_t)total_tokens * dim * sizeof(float)));
    // d_query = d_scores @ slot_keys^T
    // slot_keys is [dim, n_slots], so slot_keys^T is [n_slots, dim]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             dim, total_tokens, n_slots,
                             &alpha, p.slot_keys, n_slots, d_scores, n_slots,
                             &beta_zero, d_query, dim));

    // d_slot_keys += query^T @ d_scores → needs [dim, n_slots]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                             n_slots, dim, total_tokens,
                             &alpha, d_scores, n_slots, q_cache, dim,
                             &beta_one, dp.slot_keys, n_slots));

    // d_query → dx through proj_q: dx += d_query @ proj_q
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             dim, total_tokens, dim,
                             &alpha, p.proj_q, dim, d_query, dim,
                             &beta_one, dx, dim));

    // d_proj_q += x^T @ d_query
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                             dim, dim, total_tokens,
                             &alpha, x, dim, d_query, dim,
                             &beta_one, dp.proj_q, dim));

    CUDA_CHECK(cudaFree(d_retrieved));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(retrieved));
    CUDA_CHECK(cudaFree(d_query));
}

} // namespace rnet
