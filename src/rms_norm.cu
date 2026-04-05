#include "layers.h"
#include <cub/cub.cuh>

namespace rnet {

// ─── Forward: y[i] = (x[i] / rms) * w[i] ──────────────────────────
// One block per row (batch*seq), threads collaborate on reduction
__global__ void rms_norm_fwd_kernel(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     float* __restrict__ y,
                                     float* __restrict__ rms_out,
                                     int dim, float eps) {
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * dim;
    float* yr = y + (size_t)row * dim;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = xr[i];
        sum_sq += v * v;
    }

    // Block reduce
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    sum_sq = BlockReduce(temp).Sum(sum_sq);

    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf(sum_sq / dim + eps);
        if (rms_out) rms_out[row] = s_rms;
    }
    __syncthreads();

    float rms_inv = s_rms;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        yr[i] = xr[i] * rms_inv * w[i];
    }
}

void rms_norm_forward(const float* x, const float* w, float* y,
                      float* rms_cache, int batch_seq, int dim,
                      float eps, cudaStream_t stream) {
    rms_norm_fwd_kernel<<<batch_seq, 256, 0, stream>>>(
        x, w, y, rms_cache, dim, eps);
}

// ─── Backward ───────────────────────────────────────────────────────
// dx[i] = w[i] * rms_inv * (dy[i] - x[i] * rms_inv^2 * dot(dy*w, x) / dim)
// dw[i] += x[i] * rms_inv * dy[i]   (accumulated across rows)
__global__ void rms_norm_bwd_kernel(const float* __restrict__ dy,
                                     const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     const float* __restrict__ rms_cache,
                                     float* __restrict__ dx,
                                     float* __restrict__ dw,
                                     int dim) {
    int row = blockIdx.x;
    const float* dyr = dy + (size_t)row * dim;
    const float* xr  = x  + (size_t)row * dim;
    float* dxr = dx + (size_t)row * dim;
    float rms_inv = rms_cache[row];

    // dot(dy * w, x * rms_inv)
    float dot_val = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dot_val += dyr[i] * w[i] * xr[i] * rms_inv;
    }

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    dot_val = BlockReduce(temp).Sum(dot_val);
    __shared__ float s_dot;
    if (threadIdx.x == 0) s_dot = dot_val / dim;
    __syncthreads();

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float norm_x = xr[i] * rms_inv;
        dxr[i] = rms_inv * (dyr[i] * w[i] - norm_x * s_dot);
        atomicAdd(&dw[i], dyr[i] * norm_x);
    }
}

void rms_norm_backward(const float* dy, const float* x, const float* w,
                       const float* rms_cache, float* dx, float* dw,
                       int batch_seq, int dim, cudaStream_t stream) {
    rms_norm_bwd_kernel<<<batch_seq, 256, 0, stream>>>(
        dy, x, w, rms_cache, dx, dw, dim);
}

} // namespace rnet
