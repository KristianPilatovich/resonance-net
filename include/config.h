#pragma once
#include <cstdint>
#include <string>

namespace rnet {

struct ModelConfig {
    int d_model      = 512;
    int n_layers     = 8;
    int d_ff         = 1024;
    int n_heads      = 8;
    int gru_dim      = 512;
    int n_slots      = 1024;
    int slot_top_k   = 2;
    int vocab_size   = 256;
    int seq_len      = 256;
    float rms_eps    = 1e-6f;

    // Convolution kernel sizes
    static constexpr int CONV_SCALES = 3;
    static constexpr int conv_kernels[CONV_SCALES] = {3, 7, 15};

    // Derived
    int head_dim() const { return d_model / n_heads; }

    // Parameter count estimate
    size_t param_count() const {
        size_t p = 0;
        // Embedding + output
        p += (size_t)vocab_size * d_model * 2;
        // Final norm
        p += d_model;
        // Per layer
        for (int l = 0; l < n_layers; l++) {
            // 4x RMSNorm
            p += 4 * d_model;
            // Causal conv: 3 depthwise + mix
            p += (3 + 7 + 15) * d_model;  // depthwise kernels
            p += (size_t)d_model * d_model; // mix matrix
            // MinGRU: Wz, Wh, bz, bh
            p += 2 * (size_t)d_model * d_model + 2 * d_model;
            // Slot memory: keys, values, proj_q, proj_out
            p += 2 * (size_t)d_model * n_slots;
            p += 2 * (size_t)d_model * d_model;
            // SwiGLU FFN: gate, up, down
            p += 2 * (size_t)d_model * d_ff + (size_t)d_ff * d_model;
        }
        return p;
    }
};

struct TrainConfig {
    float lr           = 3e-4f;
    float beta1        = 0.9f;
    float beta2        = 0.999f;
    float weight_decay = 0.01f;
    float eps          = 1e-8f;
    float grad_clip    = 1.0f;
    int batch_size     = 64;
    int max_steps      = 100000;
    int warmup_steps   = 1000;
    int log_interval   = 10;
    int save_interval  = 1000;
    int eval_interval  = 100;
    int eval_tokens    = 4096;
    std::string data_path;
    std::string save_path = "checkpoints/";
};

} // namespace rnet
