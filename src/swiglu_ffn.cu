#include "layers.h"

namespace rnet {

// ─── SwiGLU Forward ─────────────────────────────────────────────────
// gate = x @ gate_w^T    [batch_seq, d_ff]
// up   = x @ up_w^T      [batch_seq, d_ff]
// hidden = SiLU(gate) * up
// y    = hidden @ down_w^T [batch_seq, d_model]

__global__ void silu_mul_kernel(const float* __restrict__ gate,
                                 const float* __restrict__ up,
                                 float* __restrict__ hidden,
                                 int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));  // SiLU(g) = g * sigmoid(g)
        hidden[i] = silu_g * up[i];
    }
}

void swiglu_forward(const float* x, float* y,
                    const SwiGLUParams& p,
                    float* gate_cache, float* up_cache,
                    int batch_seq, int dim, int d_ff,
                    cublasHandle_t cublas, cudaStream_t stream) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    // gate = x @ gate_w^T  →  [batch_seq, d_ff]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             d_ff, batch_seq, dim,
                             &alpha, p.gate_w, dim, x, dim,
                             &beta, gate_cache, d_ff));

    // up = x @ up_w^T  →  [batch_seq, d_ff]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             d_ff, batch_seq, dim,
                             &alpha, p.up_w, dim, x, dim,
                             &beta, up_cache, d_ff));

    // hidden = SiLU(gate) * up
    int total = batch_seq * d_ff;
    int thr = 256;
    int blk = (total + thr - 1) / thr;
    // We need gate_cache preserved for backward, so write to a temp
    // Actually we can compute in-place into y temporarily, then matmul
    float* hidden_buf;
    CUDA_CHECK(cudaMalloc(&hidden_buf, (size_t)total * sizeof(float)));
    silu_mul_kernel<<<blk, thr, 0, stream>>>(gate_cache, up_cache, hidden_buf, total);

    // y = hidden @ down_w^T  →  [batch_seq, dim]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             dim, batch_seq, d_ff,
                             &alpha, p.down_w, d_ff, hidden_buf, d_ff,
                             &beta, y, dim));

    CUDA_CHECK(cudaFree(hidden_buf));
}

// ─── SwiGLU Backward ────────────────────────────────────────────────
// Let h = SiLU(gate) * up, y = h @ down_w^T
// dy → dh = dy @ down_w        [batch_seq, d_ff]
//    → d_down_w += h^T @ dy    (gradient for down)
// dh → d_gate = dh * up * d_silu(gate)
//    → d_up   = dh * SiLU(gate)
// d_gate → dx += d_gate @ gate_w, d_gate_w += x^T @ d_gate
// d_up   → dx += d_up @ up_w,   d_up_w   += x^T @ d_up

__global__ void silu_mul_bwd_kernel(const float* __restrict__ dh,
                                     const float* __restrict__ gate,
                                     const float* __restrict__ up,
                                     float* __restrict__ d_gate,
                                     float* __restrict__ d_up,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        float silu_g = g * sig;
        float dsilu = sig + g * sig * (1.0f - sig);  // d/dg[g*sigmoid(g)]
        d_gate[i] = dh[i] * up[i] * dsilu;
        d_up[i]   = dh[i] * silu_g;
    }
}

void swiglu_backward(const float* dy, const float* x,
                     const SwiGLUParams& p,
                     const float* gate_cache, const float* up_cache,
                     float* dx, SwiGLUParams& dp,
                     int batch_seq, int dim, int d_ff,
                     cublasHandle_t cublas, cudaStream_t stream) {
    float alpha = 1.0f, beta = 0.0f, beta_one = 1.0f;
    CUBLAS_CHECK(cublasSetStream(cublas, stream));

    int total = batch_seq * d_ff;
    int thr = 256, blk = (total + thr - 1) / thr;

    // Recompute hidden = SiLU(gate) * up
    float *hidden_buf, *dh, *dg, *du;
    CUDA_CHECK(cudaMalloc(&hidden_buf, (size_t)total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dh, (size_t)total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dg, (size_t)total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&du, (size_t)total * sizeof(float)));

    silu_mul_kernel<<<blk, thr, 0, stream>>>(gate_cache, up_cache, hidden_buf, total);

    // dh = dy @ down_w  [batch_seq, d_ff]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             d_ff, batch_seq, dim,
                             &alpha, p.down_w, d_ff, dy, dim,
                             &beta, dh, d_ff));

    // d_down_w += hidden^T @ dy  [d_ff, dim]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                             d_ff, dim, batch_seq,
                             &alpha, hidden_buf, d_ff, dy, dim,
                             &beta_one, dp.down_w, d_ff));

    // d_gate, d_up from dh
    silu_mul_bwd_kernel<<<blk, thr, 0, stream>>>(dh, gate_cache, up_cache, dg, du, total);

    // dx += d_gate @ gate_w  [batch_seq, dim]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             dim, batch_seq, d_ff,
                             &alpha, p.gate_w, dim, dg, d_ff,
                             &beta_one, dx, dim));

    // d_gate_w += x^T @ d_gate  [dim, d_ff]  stored as [d_model, d_ff]
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                             dim, d_ff, batch_seq,
                             &alpha, x, dim, dg, d_ff,
                             &beta_one, dp.gate_w, dim));

    // dx += d_up @ up_w
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             dim, batch_seq, d_ff,
                             &alpha, p.up_w, dim, du, d_ff,
                             &beta_one, dx, dim));

    // d_up_w += x^T @ d_up
    CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                             dim, d_ff, batch_seq,
                             &alpha, x, dim, du, d_ff,
                             &beta_one, dp.up_w, dim));

    CUDA_CHECK(cudaFree(hidden_buf));
    CUDA_CHECK(cudaFree(dh));
    CUDA_CHECK(cudaFree(dg));
    CUDA_CHECK(cudaFree(du));
}

} // namespace rnet
