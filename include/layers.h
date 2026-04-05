#pragma once
#include "config.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = (call);                                           \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t st = (call);                                         \
    if (st != CUBLAS_STATUS_SUCCESS) {                                  \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n",                 \
                __FILE__, __LINE__, (int)st);                           \
        exit(1);                                                        \
    }                                                                   \
} while(0)

namespace rnet {

// ─── RMSNorm ────────────────────────────────────────────────────────
// Forward: y = (x / rms(x)) * w
// Backward: computes dx, dw
void rms_norm_forward(const float* x, const float* w, float* y,
                      float* rms_cache, int batch_seq, int dim,
                      float eps, cudaStream_t stream);

void rms_norm_backward(const float* dy, const float* x, const float* w,
                       const float* rms_cache, float* dx, float* dw,
                       int batch_seq, int dim, cudaStream_t stream);

// ─── Embedding ──────────────────────────────────────────────────────
void embed_forward(const int* tokens, const float* table,
                   float* output, int batch_seq, int vocab, int dim,
                   cudaStream_t stream);

void embed_backward(const float* dout, const int* tokens,
                    float* dtable, int batch_seq, int vocab, int dim,
                    cudaStream_t stream);

// Output logits: hidden @ weight^T → logits
void linear_forward(const float* input, const float* weight,
                    float* output, int M, int N, int K,
                    cublasHandle_t cublas, cudaStream_t stream);

// ─── Multi-Scale Causal Convolution ─────────────────────────────────
// 3 parallel depthwise causal convs (k=3,7,15) → sum → linear mix
struct CausalConvParams {
    float* conv3_w;    // [3, d_model]
    float* conv7_w;    // [7, d_model]
    float* conv15_w;   // [15, d_model]
    float* mix_w;      // [d_model, d_model]
};

void causal_conv_forward(const float* x, float* y,
                         const CausalConvParams& p,
                         float* conv_buf,  // temp buffer for conv outputs
                         int batch, int seq_len, int dim,
                         cublasHandle_t cublas, cudaStream_t stream);

void causal_conv_backward(const float* dy, const float* x,
                          const CausalConvParams& p,
                          const float* conv_buf,
                          float* dx, CausalConvParams& dp,
                          int batch, int seq_len, int dim,
                          cublasHandle_t cublas, cudaStream_t stream);

// ─── MinGRU (Parallel Scan) ─────────────────────────────────────────
struct MinGRUParams {
    float* Wz;   // [d_model, d_model]
    float* Wh;   // [d_model, d_model]
    float* bz;   // [d_model]
    float* bh;   // [d_model]
};

void min_gru_forward(const float* x, float* y,
                     const MinGRUParams& p,
                     float* z_cache,   // [batch, seq, dim] gate values
                     float* h_cache,   // [batch, seq, dim] hidden states
                     int batch, int seq_len, int dim,
                     cublasHandle_t cublas, cudaStream_t stream);

void min_gru_backward(const float* dy, const float* x,
                      const MinGRUParams& p,
                      const float* z_cache, const float* h_cache,
                      float* dx, MinGRUParams& dp,
                      int batch, int seq_len, int dim,
                      cublasHandle_t cublas, cudaStream_t stream);

// ─── Slot Memory (Top-K Cross-Attention) ────────────────────────────
struct SlotMemoryParams {
    float* slot_keys;   // [d_model, n_slots]
    float* slot_values; // [d_model, n_slots]
    float* proj_q;      // [d_model, d_model]
    float* proj_out;    // [d_model, d_model]
};

void slot_memory_forward(const float* x, float* y,
                         const SlotMemoryParams& p,
                         float* q_cache,       // [batch, seq, dim]
                         float* scores_cache,  // [batch, seq, n_slots]
                         int* topk_idx_cache,  // [batch, seq, top_k]
                         float* topk_w_cache,  // [batch, seq, top_k]
                         int batch, int seq_len, int dim,
                         int n_slots, int top_k,
                         cublasHandle_t cublas, cudaStream_t stream);

void slot_memory_backward(const float* dy, const float* x,
                          const SlotMemoryParams& p,
                          const float* q_cache, const float* scores_cache,
                          const int* topk_idx_cache, const float* topk_w_cache,
                          float* dx, SlotMemoryParams& dp,
                          int batch, int seq_len, int dim,
                          int n_slots, int top_k,
                          cublasHandle_t cublas, cudaStream_t stream);

// ─── SwiGLU FFN ─────────────────────────────────────────────────────
struct SwiGLUParams {
    float* gate_w;  // [d_model, d_ff]
    float* up_w;    // [d_model, d_ff]
    float* down_w;  // [d_ff, d_model]
};

void swiglu_forward(const float* x, float* y,
                    const SwiGLUParams& p,
                    float* gate_cache,  // [batch, seq, d_ff]
                    float* up_cache,    // [batch, seq, d_ff]
                    int batch_seq, int dim, int d_ff,
                    cublasHandle_t cublas, cudaStream_t stream);

void swiglu_backward(const float* dy, const float* x,
                     const SwiGLUParams& p,
                     const float* gate_cache, const float* up_cache,
                     float* dx, SwiGLUParams& dp,
                     int batch_seq, int dim, int d_ff,
                     cublasHandle_t cublas, cudaStream_t stream);

// ─── Loss ───────────────────────────────────────────────────────────
// Cross-entropy with numerically stable log-softmax
void cross_entropy_forward(const float* logits, const int* targets,
                           float* loss, float* dlogits,
                           int batch_seq, int vocab,
                           cudaStream_t stream);

// ─── Utility kernels ────────────────────────────────────────────────
void residual_add(float* x, const float* y, int n, cudaStream_t stream);
void scale_grads(float* grads, float scale, int n, cudaStream_t stream);

// ─── AdamW Optimizer ────────────────────────────────────────────────
void adamw_step(float* param, const float* grad, float* m, float* v,
                float lr, float beta1, float beta2, float eps,
                float weight_decay, int step, int n, cudaStream_t stream);
void clip_grads(float** grads, int* sizes, int num_tensors,
                float max_norm, cudaStream_t stream);

} // namespace rnet
