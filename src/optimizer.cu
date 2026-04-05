#include "layers.h"

namespace rnet {

// ─── AdamW Kernel ───────────────────────────────────────────────────
__global__ void adamw_kernel(float* __restrict__ param,
                              const float* __restrict__ grad,
                              float* __restrict__ m,
                              float* __restrict__ v,
                              float lr, float beta1, float beta2,
                              float eps, float weight_decay,
                              float bc1, float bc2,  // bias correction: 1-beta^t
                              int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];
    float m_new = beta1 * m[i] + (1.0f - beta1) * g;
    float v_new = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = m_new;
    v[i] = v_new;

    float m_hat = m_new / bc1;
    float v_hat = v_new / bc2;

    param[i] = param[i] * (1.0f - lr * weight_decay)
               - lr * m_hat / (sqrtf(v_hat) + eps);
}

// ─── Gradient Clipping (global norm) ────────────────────────────────
__global__ void grad_norm_kernel(const float* __restrict__ grad,
                                  float* __restrict__ partial_norm,
                                  int n) {
    extern __shared__ float sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? grad[i] * grad[i] : 0.0f;
    sdata[threadIdx.x] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(partial_norm, sdata[0]);
}

__global__ void clip_kernel(float* __restrict__ grad, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad[i] *= scale;
}

// ─── Interface used by ResonanceNet::step() ─────────────────────────
// Collects all params and grads into flat arrays, runs AdamW

struct AdamWRunner {
    static void run(float** params, float** grads,
                    size_t* sizes, int num_tensors,
                    float* m, float* v,
                    float lr, float beta1, float beta2,
                    float eps, float weight_decay, float grad_clip,
                    int step, cudaStream_t stream) {
        // Compute bias correction
        float bc1 = 1.0f - powf(beta1, (float)step);
        float bc2 = 1.0f - powf(beta2, (float)step);

        // Gradient clipping: compute global norm
        float* d_norm;
        cudaMalloc(&d_norm, sizeof(float));
        cudaMemsetAsync(d_norm, 0, sizeof(float), stream);

        size_t total = 0;
        for (int t = 0; t < num_tensors; t++) total += sizes[t];

        int thr = 256;
        size_t offset = 0;
        for (int t = 0; t < num_tensors; t++) {
            int n = (int)sizes[t];
            int blk = (n + thr - 1) / thr;
            grad_norm_kernel<<<blk, thr, thr * sizeof(float), stream>>>(
                grads[t], d_norm, n);
            offset += sizes[t];
        }

        float h_norm;
        cudaMemcpyAsync(&h_norm, d_norm, sizeof(float),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        h_norm = sqrtf(h_norm);

        // Clip if needed
        if (grad_clip > 0 && h_norm > grad_clip) {
            float scale = grad_clip / (h_norm + 1e-6f);
            offset = 0;
            for (int t = 0; t < num_tensors; t++) {
                int n = (int)sizes[t];
                int blk = (n + thr - 1) / thr;
                clip_kernel<<<blk, thr, 0, stream>>>(grads[t], scale, n);
                offset += sizes[t];
            }
        }

        // AdamW update
        offset = 0;
        for (int t = 0; t < num_tensors; t++) {
            int n = (int)sizes[t];
            int blk = (n + thr - 1) / thr;
            adamw_kernel<<<blk, thr, 0, stream>>>(
                params[t], grads[t], m + offset, v + offset,
                lr, beta1, beta2, eps, weight_decay, bc1, bc2, n);
            offset += sizes[t];
        }

        cudaFree(d_norm);
    }
};

} // namespace rnet
