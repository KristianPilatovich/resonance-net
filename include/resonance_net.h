#pragma once
#include "config.h"
#include "layers.h"
#include <vector>
#include <string>

namespace rnet {

struct LayerWeights {
    // 4x RMSNorm
    float* norm1_w;   // pre-conv
    float* norm2_w;   // pre-gru
    float* norm3_w;   // pre-slot
    float* norm4_w;   // pre-ffn

    CausalConvParams conv;
    MinGRUParams gru;
    SlotMemoryParams slot;
    SwiGLUParams ffn;
};

struct LayerGrads {
    float* norm1_w;
    float* norm2_w;
    float* norm3_w;
    float* norm4_w;

    CausalConvParams conv;
    MinGRUParams gru;
    SlotMemoryParams slot;
    SwiGLUParams ffn;
};

struct LayerCache {
    // RMSNorm caches
    float* rms1; float* rms2; float* rms3; float* rms4;
    // Normed inputs
    float* normed1; float* normed2; float* normed3; float* normed4;
    // Conv
    float* conv_buf;
    // MinGRU
    float* z_cache; float* h_cache;
    // Slot memory
    float* q_cache; float* scores_cache;
    int* topk_idx; float* topk_w;
    // SwiGLU
    float* gate_cache; float* up_cache;
    // Residual stream snapshots for backward
    float* res_after_conv;
    float* res_after_gru;
    float* res_after_slot;
};

class ResonanceNet {
public:
    bool init(const ModelConfig& cfg);
    void destroy();

    // Forward pass: tokens [batch, seq] → logits [batch, seq, vocab]
    void forward(const int* d_tokens, float* d_logits, int batch, int seq_len);

    // Backward pass: from dlogits, compute all gradients
    void backward(const float* d_dlogits, const int* d_tokens, int batch, int seq_len);

    // Update weights with AdamW
    void step(const TrainConfig& tcfg, int step_num);

    // Zero all gradients
    void zero_grad();

    // Save/load checkpoint
    bool save(const std::string& path);
    bool load(const std::string& path);

    // Inference: generate tokens
    void generate(const int* prompt, int prompt_len, int* output,
                  int max_tokens, float temperature = 0.8f);

    const ModelConfig& config() const { return cfg_; }
    size_t param_count() const { return total_params_; }

private:
    ModelConfig cfg_;
    cublasHandle_t cublas_ = nullptr;
    cudaStream_t stream_ = nullptr;

    // Global weights
    float* d_embed_w_ = nullptr;   // [vocab, d_model]
    float* d_output_w_ = nullptr;  // [vocab, d_model]  (tied or separate)
    float* d_final_norm_ = nullptr;// [d_model]

    // Global grads
    float* d_embed_g_ = nullptr;
    float* d_output_g_ = nullptr;
    float* d_final_norm_g_ = nullptr;

    // Per-layer
    std::vector<LayerWeights> layers_;
    std::vector<LayerGrads> grads_;
    std::vector<LayerCache> cache_;

    // AdamW states
    float* d_m_ = nullptr;  // first moment
    float* d_v_ = nullptr;  // second moment

    // Working buffers
    float* d_hidden_ = nullptr;    // [batch, seq, d_model]
    float* d_dhidden_ = nullptr;   // grad for hidden
    float* d_logits_buf_ = nullptr;

    // Backward caches
    float* d_final_normed_ = nullptr;  // cached final norm output
    float* d_final_rms_ = nullptr;     // cached final norm rms values
    std::vector<float*> d_layer_inputs_; // hidden state before each layer
    int last_batch_ = 0;
    int last_seq_ = 0;
    int alloc_bs_ = 0;  // allocated batch*seq size

    size_t total_params_ = 0;

    void alloc_layer(int layer_idx);
    void init_weights();
    void layer_forward(int l, float* d_hidden, int batch, int seq_len);
    void layer_forward_infer(int l, float* d_hidden, int batch, int seq_len);
    void layer_backward(int l, float* d_dhidden, int batch, int seq_len);
};

} // namespace rnet
