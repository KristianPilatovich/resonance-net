#include "layers.h"

namespace rnet {

// ─── Depthwise Causal Convolution Kernel ────────────────────────────
// Each thread handles one (position, channel) pair.
// Causal = only look at past tokens: output[t] uses input[t-k+1 .. t]
// Left-padded with zeros for t < k-1.
__global__ void depthwise_causal_conv_kernel(
        const float* __restrict__ x,       // [batch, seq, dim]
        const float* __restrict__ weight,  // [kernel_size, dim]
        float* __restrict__ y,             // [batch, seq, dim]
        int batch, int seq_len, int dim, int kernel_size) {
    // Grid: (batch * seq_len, ceil(dim/blockDim.x))
    int pos = blockIdx.x;
    int b = pos / seq_len;
    int t = pos % seq_len;
    if (b >= batch) return;

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            int src_t = t - (kernel_size - 1) + k;
            if (src_t >= 0) {
                sum += x[((size_t)b * seq_len + src_t) * dim + d] * weight[k * dim + d];
            }
        }
        y[((size_t)b * seq_len + t) * dim + d] = sum;
    }
}

// ─── Sum three conv outputs ─────────────────────────────────────────
__global__ void sum3_kernel(const float* __restrict__ a,
                             const float* __restrict__ b,
                             const float* __restrict__ c,
                             float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i] + c[i];
}

// ─── Forward ────────────────────────────────────────────────────────
// 1) conv3, conv7, conv15 in parallel
// 2) sum outputs
// 3) linear mix: y = sum @ mix_w^T
void causal_conv_forward(const float* x, float* y,
                         const CausalConvParams& p,
                         float* conv_buf,
                         int batch, int seq_len, int dim,
                         cublasHandle_t cublas, cudaStream_t stream) {
    int total_tokens = batch * seq_len;
    int total_elems = total_tokens * dim;
    int thr = 256;

    // conv_buf layout: [3 * batch * seq * dim]
    // [0]: conv3_out, [1]: conv7_out, [2]: conv15_out
    float* c3 = conv_buf;
    float* c7 = conv_buf + total_elems;
    float* c15 = conv_buf + 2 * total_elems;

    // Launch 3 convolutions (could use 3 streams for true parallelism)
    depthwise_causal_conv_kernel<<<total_tokens, thr, 0, stream>>>(
        x, p.conv3_w, c3, batch, seq_len, dim, 3);
    depthwise_causal_conv_kernel<<<total_tokens, thr, 0, stream>>>(
        x, p.conv7_w, c7, batch, seq_len, dim, 7);
    depthwise_causal_conv_kernel<<<total_tokens, thr, 0, stream>>>(
        x, p.conv15_w, c15, batch, seq_len, dim, 15);

    // Sum: sum_out = c3 + c7 + c15 (reuse c3 as sum buffer)
    float* sum_out = c3;  // overwrite c3
    int blk = (total_elems + thr - 1) / thr;
    sum3_kernel<<<blk, thr, 0, stream>>>(c3, c7, c15, sum_out, total_elems);

    // Mix: y = sum_out @ mix_w^T  [total_tokens, dim]
    float alpha = 1.0f, beta_val = 0.0f;
    CUBLAS_CHECK(cublasSetStream(cublas, stream));
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             dim, total_tokens, dim,
                             &alpha, p.mix_w, dim, sum_out, dim,
                             &beta_val, y, dim));
}

// ─── Depthwise Causal Conv Backward ─────────────────────────────────
// dx[t,d] += sum_k weight[k,d] * dy[t + (kernel_size-1) - k, d]
// dw[k,d] += sum_t x[t - (kernel_size-1) + k, d] * dy[t, d]
__global__ void depthwise_causal_conv_bwd_data_kernel(
        const float* __restrict__ dy,
        const float* __restrict__ weight,
        float* __restrict__ dx,
        int batch, int seq_len, int dim, int kernel_size) {
    int pos = blockIdx.x;
    int b = pos / seq_len;
    int t = pos % seq_len;
    if (b >= batch) return;

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            int out_t = t + (kernel_size - 1) - k;
            if (out_t < seq_len) {
                sum += weight[k * dim + d] *
                       dy[((size_t)b * seq_len + out_t) * dim + d];
            }
        }
        atomicAdd(&dx[((size_t)b * seq_len + t) * dim + d], sum);
    }
}

__global__ void depthwise_causal_conv_bwd_weight_kernel(
        const float* __restrict__ x,
        const float* __restrict__ dy,
        float* __restrict__ dw,
        int batch, int seq_len, int dim, int kernel_size) {
    // One block per (k, d_block)
    int k = blockIdx.x;
    if (k >= kernel_size) return;

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < seq_len; t++) {
                int src_t = t - (kernel_size - 1) + k;
                if (src_t >= 0) {
                    sum += x[((size_t)b * seq_len + src_t) * dim + d] *
                           dy[((size_t)b * seq_len + t) * dim + d];
                }
            }
        }
        atomicAdd(&dw[k * dim + d], sum);
    }
}

void causal_conv_backward(const float* dy_mixed, const float* x,
                          const CausalConvParams& p,
                          const float* conv_buf,
                          float* dx, CausalConvParams& dp,
                          int batch, int seq_len, int dim,
                          cublasHandle_t cublas, cudaStream_t stream) {
    int total_tokens = batch * seq_len;
    int total_elems = total_tokens * dim;
    float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;
    int thr = 256;

    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    // dy_mixed is gradient w.r.t. output of mix projection
    // d_sum = dy_mixed @ mix_w  (backprop through linear mix)
    float* d_sum;
    CUDA_CHECK(cudaMalloc(&d_sum, (size_t)total_elems * sizeof(float)));
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             dim, total_tokens, dim,
                             &alpha, p.mix_w, dim, dy_mixed, dim,
                             &beta_zero, d_sum, dim));

    // d_mix_w += sum_out^T @ dy_mixed
    // sum_out was stored in conv_buf[0] (was overwritten by sum3, need recompute or cache)
    // For simplicity, recompute sum from cached conv outputs
    const float* c3 = conv_buf;
    const float* c7 = conv_buf + total_elems;
    const float* c15 = conv_buf + 2 * total_elems;
    float* recomp_sum;
    CUDA_CHECK(cudaMalloc(&recomp_sum, (size_t)total_elems * sizeof(float)));
    int blk = (total_elems + thr - 1) / thr;
    sum3_kernel<<<blk, thr, 0, stream>>>(c3, c7, c15, recomp_sum, total_elems);

    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                             dim, dim, total_tokens,
                             &alpha, recomp_sum, dim, dy_mixed, dim,
                             &beta_one, dp.mix_w, dim));

    // d_sum flows equally to all 3 conv branches (since sum = c3+c7+c15)
    // Backward through each depthwise conv
    // dx accumulates from all 3
    depthwise_causal_conv_bwd_data_kernel<<<total_tokens, thr, 0, stream>>>(
        d_sum, p.conv3_w, dx, batch, seq_len, dim, 3);
    depthwise_causal_conv_bwd_data_kernel<<<total_tokens, thr, 0, stream>>>(
        d_sum, p.conv7_w, dx, batch, seq_len, dim, 7);
    depthwise_causal_conv_bwd_data_kernel<<<total_tokens, thr, 0, stream>>>(
        d_sum, p.conv15_w, dx, batch, seq_len, dim, 15);

    // Weight gradients
    depthwise_causal_conv_bwd_weight_kernel<<<3, thr, 0, stream>>>(
        x, d_sum, dp.conv3_w, batch, seq_len, dim, 3);
    depthwise_causal_conv_bwd_weight_kernel<<<7, thr, 0, stream>>>(
        x, d_sum, dp.conv7_w, batch, seq_len, dim, 7);
    depthwise_causal_conv_bwd_weight_kernel<<<15, thr, 0, stream>>>(
        x, d_sum, dp.conv15_w, batch, seq_len, dim, 15);

    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(recomp_sum));
}

} // namespace rnet
