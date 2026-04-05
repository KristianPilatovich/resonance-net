#include "layers.h"

namespace rnet {

// ─── Embedding Forward ──────────────────────────────────────────────
__global__ void embed_fwd_kernel(const int* __restrict__ tokens,
                                  const float* __restrict__ table,
                                  float* __restrict__ output,
                                  int batch_seq, int dim) {
    int idx = blockIdx.x;  // which token position
    int d = threadIdx.x;   // which dim
    if (idx >= batch_seq) return;

    int tok = tokens[idx];
    for (int i = d; i < dim; i += blockDim.x) {
        output[(size_t)idx * dim + i] = table[(size_t)tok * dim + i];
    }
}

void embed_forward(const int* tokens, const float* table,
                   float* output, int batch_seq, int vocab, int dim,
                   cudaStream_t stream) {
    embed_fwd_kernel<<<batch_seq, 256, 0, stream>>>(
        tokens, table, output, batch_seq, dim);
}

// ─── Embedding Backward ─────────────────────────────────────────────
__global__ void embed_bwd_kernel(const float* __restrict__ dout,
                                  const int* __restrict__ tokens,
                                  float* __restrict__ dtable,
                                  int batch_seq, int dim) {
    int idx = blockIdx.x;
    if (idx >= batch_seq) return;

    int tok = tokens[idx];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        atomicAdd(&dtable[(size_t)tok * dim + i],
                  dout[(size_t)idx * dim + i]);
    }
}

void embed_backward(const float* dout, const int* tokens,
                    float* dtable, int batch_seq, int vocab, int dim,
                    cudaStream_t stream) {
    embed_bwd_kernel<<<batch_seq, 256, 0, stream>>>(
        dout, tokens, dtable, batch_seq, dim);
}

// ─── Linear Forward (for output logits): y = x @ W^T ────────────────
void linear_forward(const float* input, const float* weight,
                    float* output, int M, int N, int K,
                    cublasHandle_t cublas, cudaStream_t stream) {
    // input:  [M, K]
    // weight: [N, K] (row-major, so W^T is [K, N] in col-major)
    // output: [M, N]
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(cublas, stream));
    // cublas is col-major: C = alpha * B^T * A + beta * C
    // We want: output[M,N] = input[M,K] @ weight[N,K]^T
    // In col-major: output^T[N,M] = weight[N,K] @ input^T[K,M]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             weight, K,    // weight [N,K] → col-major [K,N], transposed = [N,K]
                             input, K,     // input [M,K] → col-major [K,M]
                             &beta,
                             output, N));  // output [M,N] → col-major [N,M]
}

// ─── Utility: residual add ──────────────────────────────────────────
__global__ void residual_add_kernel(float* x, const float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

void residual_add(float* x, const float* y, int n, cudaStream_t stream) {
    int thr = 256;
    int blk = (n + thr - 1) / thr;
    residual_add_kernel<<<blk, thr, 0, stream>>>(x, y, n);
}

// ─── Scale gradients ────────────────────────────────────────────────
__global__ void scale_kernel(float* data, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= scale;
}

void scale_grads(float* grads, float scale, int n, cudaStream_t stream) {
    int thr = 256;
    int blk = (n + thr - 1) / thr;
    scale_kernel<<<blk, thr, 0, stream>>>(grads, scale, n);
}

// ─── Cross-entropy loss ─────────────────────────────────────────────
// Fused log-softmax + NLL loss + backward (dlogits)
// Uses shared memory block reduction (correct for any thread count)
__global__ void cross_entropy_kernel(const float* __restrict__ logits,
                                      const int* __restrict__ targets,
                                      float* __restrict__ loss,
                                      float* __restrict__ dlogits,
                                      int vocab) {
    extern __shared__ float smem[];  // [blockDim.x]

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const float* lr = logits + (size_t)row * vocab;
    float* dr = dlogits + (size_t)row * vocab;
    int target = targets[row];

    // ── Find max (block reduction via shared memory) ──
    float local_max = -1e30f;
    for (int i = tid; i < vocab; i += nthreads) {
        local_max = fmaxf(local_max, lr[i]);
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    float max_val = smem[0];
    __syncthreads();

    // ── Sum exp (block reduction) ──
    float local_sum = 0.0f;
    for (int i = tid; i < vocab; i += nthreads) {
        local_sum += expf(lr[i] - max_val);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float sum_exp = smem[0];
    float log_sum = logf(sum_exp);

    // ── Loss ──
    if (tid == 0) {
        float token_loss = -(lr[target] - max_val - log_sum);
        atomicAdd(loss, token_loss);
    }

    // ── Gradient: softmax - one_hot ──
    for (int i = tid; i < vocab; i += nthreads) {
        float softmax_i = expf(lr[i] - max_val - log_sum);
        dr[i] = softmax_i - (i == target ? 1.0f : 0.0f);
    }
}

void cross_entropy_forward(const float* logits, const int* targets,
                           float* loss, float* dlogits,
                           int batch_seq, int vocab,
                           cudaStream_t stream) {
    // Zero loss
    CUDA_CHECK(cudaMemsetAsync(loss, 0, sizeof(float), stream));
    // Launch one block per token, 256 threads for reduction over vocab=256
    int threads = min(256, vocab);
    cross_entropy_kernel<<<batch_seq, threads, threads * sizeof(float), stream>>>(
        logits, targets, loss, dlogits, vocab);
}

} // namespace rnet
