#include "layers.h"

namespace rnet {

// ─── MinGRU (Sequential, matching original ResonanceNet V5) ─────────
// h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
// z_t = sigmoid(x_t @ Wz + bz)
// h_tilde_t = x_t @ Wh + bh

// ─── Bias add kernel ────────────────────────────────────────────────
__global__ void bias_add_kernel(float* __restrict__ data,
                                 const float* __restrict__ bias,
                                 int total, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) data[i] += bias[i % dim];
}

// ─── Sequential Forward Kernel ──────────────────────────────────────
// One thread per (batch, dim), iterates over sequence
__global__ void min_gru_fwd_kernel(const float* __restrict__ proj_z,
                                    const float* __restrict__ proj_h,
                                    const float* __restrict__ bz,
                                    const float* __restrict__ bh,
                                    float* __restrict__ z_out,
                                    float* __restrict__ h_out,
                                    float* __restrict__ y,
                                    int batch, int seq_len, int dim) {
    int bd = blockIdx.x * blockDim.x + threadIdx.x;
    int b = bd / dim;
    int d = bd % dim;
    if (b >= batch) return;

    float h = 0.0f;
    float bz_d = bz[d];
    float bh_d = bh[d];

    for (int t = 0; t < seq_len; t++) {
        size_t idx = ((size_t)b * seq_len + t) * dim + d;
        float z = 1.0f / (1.0f + expf(-(proj_z[idx] + bz_d)));
        float h_tilde = proj_h[idx] + bh_d;

        h = (1.0f - z) * h + z * h_tilde;

        z_out[idx] = z;
        h_out[idx] = h;
        y[idx] = h;
    }
}

void min_gru_forward(const float* x, float* y,
                     const MinGRUParams& p,
                     float* z_cache, float* h_cache,
                     int batch, int seq_len, int dim,
                     cublasHandle_t cublas, cudaStream_t stream) {
    int total_tokens = batch * seq_len;
    size_t total_elems = (size_t)total_tokens * dim;
    float alpha = 1.0f, beta_val = 0.0f;
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    float *proj_z, *proj_h;
    CUDA_CHECK(cudaMalloc(&proj_z, total_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&proj_h, total_elems * sizeof(float)));

    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             dim, total_tokens, dim,
                             &alpha, p.Wz, dim, x, dim,
                             &beta_val, proj_z, dim));

    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             dim, total_tokens, dim,
                             &alpha, p.Wh, dim, x, dim,
                             &beta_val, proj_h, dim));

    int bd_total = batch * dim;
    int thr = 256;
    int blk = (bd_total + thr - 1) / thr;
    min_gru_fwd_kernel<<<blk, thr, 0, stream>>>(
        proj_z, proj_h, p.bz, p.bh,
        z_cache, h_cache, y,
        batch, seq_len, dim);

    CUDA_CHECK(cudaFree(proj_z));
    CUDA_CHECK(cudaFree(proj_h));
}

// ─── Sequential Backward Kernel ─────────────────────────────────────
__global__ void min_gru_bwd_kernel(const float* __restrict__ dy,
                                    const float* __restrict__ z_cache,
                                    const float* __restrict__ h_cache,
                                    const float* __restrict__ htilde,
                                    float* __restrict__ dz_out,
                                    float* __restrict__ dhtilde_out,
                                    int batch, int seq_len, int dim) {
    int bd = blockIdx.x * blockDim.x + threadIdx.x;
    int b = bd / dim;
    int d = bd % dim;
    if (b >= batch) return;

    float dh = 0.0f;
    for (int t = seq_len - 1; t >= 0; t--) {
        size_t idx = ((size_t)b * seq_len + t) * dim + d;
        dh += dy[idx];

        float z = z_cache[idx];
        float h_prev = (t > 0) ? h_cache[((size_t)b * seq_len + t - 1) * dim + d] : 0.0f;
        float ht = htilde[idx];

        dz_out[idx] = dh * (ht - h_prev) * z * (1.0f - z);
        dhtilde_out[idx] = dh * z;
        dh = dh * (1.0f - z);
    }
}

// ─── Bias gradient kernel ───────────────────────────────────────────
__global__ void bias_grad_kernel(const float* __restrict__ dvals,
                                  float* __restrict__ dbias,
                                  int total_tokens, int dim) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    float sum = 0.0f;
    for (int t = 0; t < total_tokens; t++) {
        sum += dvals[(size_t)t * dim + d];
    }
    atomicAdd(&dbias[d], sum);
}

void min_gru_backward(const float* dy, const float* x,
                      const MinGRUParams& p,
                      const float* z_cache, const float* h_cache,
                      float* dx, MinGRUParams& dp,
                      int batch, int seq_len, int dim,
                      cublasHandle_t cublas, cudaStream_t stream) {
    int total_tokens = batch * seq_len;
    size_t total_elems = (size_t)total_tokens * dim;
    float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;
    int thr = 256;
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    // Recompute h_tilde = x @ Wh^T + bh
    float *htilde, *dz, *dhtilde;
    CUDA_CHECK(cudaMalloc(&htilde, total_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dz, total_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dhtilde, total_elems * sizeof(float)));

    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             dim, total_tokens, dim,
                             &alpha, p.Wh, dim, x, dim,
                             &beta_zero, htilde, dim));

    // Add bias: htilde[i] += bh[i % dim]
    int blk_add = ((int)total_elems + thr - 1) / thr;
    bias_add_kernel<<<blk_add, thr, 0, stream>>>(htilde, p.bh, (int)total_elems, dim);

    // Backward through time
    int bd_total = batch * dim;
    int bwd_blk = (bd_total + thr - 1) / thr;
    min_gru_bwd_kernel<<<bwd_blk, thr, 0, stream>>>(
        dy, z_cache, h_cache, htilde,
        dz, dhtilde,
        batch, seq_len, dim);

    // Backprop through Wz: dx += dz @ Wz, dWz += x^T @ dz
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             dim, total_tokens, dim,
                             &alpha, p.Wz, dim, dz, dim,
                             &beta_one, dx, dim));
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                             dim, dim, total_tokens,
                             &alpha, x, dim, dz, dim,
                             &beta_one, dp.Wz, dim));

    // Backprop through Wh: dx += dhtilde @ Wh, dWh += x^T @ dhtilde
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             dim, total_tokens, dim,
                             &alpha, p.Wh, dim, dhtilde, dim,
                             &beta_one, dx, dim));
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                             dim, dim, total_tokens,
                             &alpha, x, dim, dhtilde, dim,
                             &beta_one, dp.Wh, dim));

    // Bias gradients: dbz += sum(dz), dbh += sum(dhtilde)
    int bias_blk = (dim + thr - 1) / thr;
    bias_grad_kernel<<<bias_blk, thr, 0, stream>>>(dz, dp.bz, total_tokens, dim);
    bias_grad_kernel<<<bias_blk, thr, 0, stream>>>(dhtilde, dp.bh, total_tokens, dim);

    CUDA_CHECK(cudaFree(htilde));
    CUDA_CHECK(cudaFree(dz));
    CUDA_CHECK(cudaFree(dhtilde));
}

} // namespace rnet
