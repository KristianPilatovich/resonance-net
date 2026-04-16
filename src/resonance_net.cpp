#include "resonance_net.h"
#include "dist.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

namespace rnet {

static size_t total_allocated_ = 0;
static size_t alloc_gpu(float** ptr, size_t count) {
    size_t bytes = count * sizeof(float);
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        fprintf(stderr, "OOM: tried %.1f MB, already allocated %.1f MB, free=%.1f MB / %.1f MB\n",
                bytes / 1e6, total_allocated_ / 1e6, free_mem / 1e6, total_mem / 1e6);
        exit(1);
    }
    total_allocated_ += bytes;
    CUDA_CHECK(cudaMemset(*ptr, 0, bytes));
    return count;
}

static void xavier_init(float* d_ptr, size_t count, int fan_in, unsigned seed) {
    std::mt19937 rng(seed);
    float scale = 1.0f / sqrtf((float)fan_in);
    std::uniform_real_distribution<float> dist(-scale, scale);

    std::vector<float> h(count);
    for (size_t i = 0; i < count; i++) h[i] = dist(rng);
    CUDA_CHECK(cudaMemcpy(d_ptr, h.data(), count * sizeof(float),
                           cudaMemcpyHostToDevice));
}

static void ones_init(float* d_ptr, size_t count) {
    std::vector<float> h(count, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_ptr, h.data(), count * sizeof(float),
                           cudaMemcpyHostToDevice));
}

bool ResonanceNet::init(const ModelConfig& cfg) {
    cfg_ = cfg;
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));

    int D = cfg.d_model;
    int V = cfg.vocab_size;
    int FF = cfg.d_ff;
    int NS = cfg.n_slots;
    int NL = cfg.n_layers;

    printf("ResonanceNet V5 init: d=%d, layers=%d, ff=%d, slots=%d, vocab=%d\n",
           D, NL, FF, NS, V);

    // Global weights
    total_params_ = 0;
    total_params_ += alloc_gpu(&d_embed_w_, (size_t)V * D);
    total_params_ += alloc_gpu(&d_output_w_, (size_t)V * D);
    total_params_ += alloc_gpu(&d_final_norm_, D);

    // Global grads
    alloc_gpu(&d_embed_g_, (size_t)V * D);
    alloc_gpu(&d_output_g_, (size_t)V * D);
    alloc_gpu(&d_final_norm_g_, D);

    // Per-layer
    layers_.resize(NL);
    grads_.resize(NL);
    cache_.resize(NL);

    unsigned seed = 42;
    for (int l = 0; l < NL; l++) {
        alloc_layer(l);
        seed += 1000;
    }

    printf("Total parameters: %zu (%.1f M)\n", total_params_, total_params_ / 1e6f);

    // Init weights
    init_weights();

    // No optimizer states — plain SGD (delta accumulation)

    return true;
}

void ResonanceNet::alloc_layer(int l) {
    int D = cfg_.d_model;
    int FF = cfg_.d_ff;
    int NS = cfg_.n_slots;
    int S = cfg_.seq_len;
    // Compute max batch to fit cache in VRAM (~13 float buffers per layer of size B*S*D)
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t budget_per_layer = free_mem / (cfg_.n_layers + 1);
    size_t elems_per_batch = (size_t)S * D * 13 + (size_t)S * NS + (size_t)S * 2 + (size_t)S * FF * 2;
    int B = std::max(1, std::min(64, (int)(budget_per_layer / (elems_per_batch * 4))));

    auto& w = layers_[l];
    auto& g = grads_[l];
    auto& c = cache_[l];

    // RMSNorm weights (4 per layer)
    total_params_ += alloc_gpu(&w.norm1_w, D);
    total_params_ += alloc_gpu(&w.norm2_w, D);
    total_params_ += alloc_gpu(&w.norm3_w, D);
    total_params_ += alloc_gpu(&w.norm4_w, D);

    alloc_gpu(&g.norm1_w, D);
    alloc_gpu(&g.norm2_w, D);
    alloc_gpu(&g.norm3_w, D);
    alloc_gpu(&g.norm4_w, D);

    // Causal conv
    total_params_ += alloc_gpu(&w.conv.conv3_w, 3 * D);
    total_params_ += alloc_gpu(&w.conv.conv7_w, 7 * D);
    total_params_ += alloc_gpu(&w.conv.conv15_w, 15 * D);
    total_params_ += alloc_gpu(&w.conv.mix_w, (size_t)D * D);

    alloc_gpu(&g.conv.conv3_w, 3 * D);
    alloc_gpu(&g.conv.conv7_w, 7 * D);
    alloc_gpu(&g.conv.conv15_w, 15 * D);
    alloc_gpu(&g.conv.mix_w, (size_t)D * D);

    // MinGRU
    total_params_ += alloc_gpu(&w.gru.Wz, (size_t)D * D);
    total_params_ += alloc_gpu(&w.gru.Wh, (size_t)D * D);
    total_params_ += alloc_gpu(&w.gru.bz, D);
    total_params_ += alloc_gpu(&w.gru.bh, D);

    alloc_gpu(&g.gru.Wz, (size_t)D * D);
    alloc_gpu(&g.gru.Wh, (size_t)D * D);
    alloc_gpu(&g.gru.bz, D);
    alloc_gpu(&g.gru.bh, D);

    // Slot memory
    total_params_ += alloc_gpu(&w.slot.slot_keys, (size_t)D * NS);
    total_params_ += alloc_gpu(&w.slot.slot_values, (size_t)D * NS);
    total_params_ += alloc_gpu(&w.slot.proj_q, (size_t)D * D);
    total_params_ += alloc_gpu(&w.slot.proj_out, (size_t)D * D);

    alloc_gpu(&g.slot.slot_keys, (size_t)D * NS);
    alloc_gpu(&g.slot.slot_values, (size_t)D * NS);
    alloc_gpu(&g.slot.proj_q, (size_t)D * D);
    alloc_gpu(&g.slot.proj_out, (size_t)D * D);

    // SwiGLU
    total_params_ += alloc_gpu(&w.ffn.gate_w, (size_t)D * FF);
    total_params_ += alloc_gpu(&w.ffn.up_w, (size_t)D * FF);
    total_params_ += alloc_gpu(&w.ffn.down_w, (size_t)FF * D);

    alloc_gpu(&g.ffn.gate_w, (size_t)D * FF);
    alloc_gpu(&g.ffn.up_w, (size_t)D * FF);
    alloc_gpu(&g.ffn.down_w, (size_t)FF * D);

    // Cache
    size_t BS = (size_t)B * S;
    alloc_gpu(&c.rms1, BS); alloc_gpu(&c.rms2, BS);
    alloc_gpu(&c.rms3, BS); alloc_gpu(&c.rms4, BS);
    alloc_gpu(&c.normed1, BS * D); alloc_gpu(&c.normed2, BS * D);
    alloc_gpu(&c.normed3, BS * D); alloc_gpu(&c.normed4, BS * D);
    alloc_gpu(&c.conv_buf, 3 * BS * D);
    alloc_gpu(&c.z_cache, BS * D); alloc_gpu(&c.h_cache, BS * D);
    alloc_gpu(&c.q_cache, BS * D);
    alloc_gpu(&c.scores_cache, BS * NS);
    CUDA_CHECK(cudaMalloc(&c.topk_idx, BS * cfg_.slot_top_k * sizeof(int)));
    CUDA_CHECK(cudaMemset(c.topk_idx, 0, BS * cfg_.slot_top_k * sizeof(int)));
    alloc_gpu(&c.topk_w, BS * cfg_.slot_top_k);
    alloc_gpu(&c.gate_cache, BS * FF);
    alloc_gpu(&c.up_cache, BS * FF);
    alloc_gpu(&c.res_after_conv, BS * D);
    alloc_gpu(&c.res_after_gru, BS * D);
    alloc_gpu(&c.res_after_slot, BS * D);
}

void ResonanceNet::init_weights() {
    int D = cfg_.d_model;
    int V = cfg_.vocab_size;
    int FF = cfg_.d_ff;
    int NS = cfg_.n_slots;
    unsigned seed = 42;

    xavier_init(d_embed_w_, (size_t)V * D, D, seed++);
    xavier_init(d_output_w_, (size_t)V * D, D, seed++);
    ones_init(d_final_norm_, D);

    for (int l = 0; l < cfg_.n_layers; l++) {
        auto& w = layers_[l];
        ones_init(w.norm1_w, D);
        ones_init(w.norm2_w, D);
        ones_init(w.norm3_w, D);
        ones_init(w.norm4_w, D);

        xavier_init(w.conv.conv3_w, 3 * D, D, seed++);
        xavier_init(w.conv.conv7_w, 7 * D, D, seed++);
        xavier_init(w.conv.conv15_w, 15 * D, D, seed++);
        xavier_init(w.conv.mix_w, (size_t)D * D, D, seed++);

        xavier_init(w.gru.Wz, (size_t)D * D, D, seed++);
        xavier_init(w.gru.Wh, (size_t)D * D, D, seed++);
        // biases stay zero

        xavier_init(w.slot.slot_keys, (size_t)D * NS, D, seed++);
        xavier_init(w.slot.slot_values, (size_t)D * NS, D, seed++);
        xavier_init(w.slot.proj_q, (size_t)D * D, D, seed++);
        xavier_init(w.slot.proj_out, (size_t)D * D, D, seed++);

        xavier_init(w.ffn.gate_w, (size_t)D * FF, D, seed++);
        xavier_init(w.ffn.up_w, (size_t)D * FF, D, seed++);
        xavier_init(w.ffn.down_w, (size_t)FF * D, FF, seed++);
    }
}

void ResonanceNet::forward(const int* d_tokens, float* d_logits, int batch, int seq_len) {
    int D = cfg_.d_model;
    int BS = batch * seq_len;
    last_batch_ = batch;
    last_seq_ = seq_len;

    // Allocate/reallocate hidden buffer if size changed
    size_t needed = (size_t)BS * D;
    if (alloc_bs_ != BS) {
        auto sf = [](auto& p) { if (p) { cudaFree(p); p = nullptr; } };
        sf(d_hidden_); sf(d_final_normed_); sf(d_final_rms_);
        for (auto& p : d_layer_inputs_) sf(p);
        d_layer_inputs_.clear();

        alloc_gpu(&d_hidden_, needed);
        alloc_gpu(&d_final_normed_, needed);
        alloc_gpu(&d_final_rms_, BS);
        d_layer_inputs_.resize(cfg_.n_layers + 1);
        for (int l = 0; l <= cfg_.n_layers; l++)
            alloc_gpu(&d_layer_inputs_[l], needed);
        alloc_bs_ = BS;
    }

    // Embedding
    embed_forward(d_tokens, d_embed_w_, d_hidden_, BS, cfg_.vocab_size, D, stream_);

    // Layers — use cached forward for training, inference path otherwise
    bool use_cache = (BS <= 64 * cfg_.seq_len) && !cache_.empty();  // fits in pre-allocated cache
    for (int l = 0; l < cfg_.n_layers; l++) {
        if (use_cache) {
            CUDA_CHECK(cudaMemcpyAsync(d_layer_inputs_[l], d_hidden_,
                                        needed * sizeof(float),
                                        cudaMemcpyDeviceToDevice, stream_));
            layer_forward(l, d_hidden_, batch, seq_len);
        } else {
            layer_forward_infer(l, d_hidden_, batch, seq_len);
        }
    }
    if (use_cache) {
        CUDA_CHECK(cudaMemcpyAsync(d_layer_inputs_[cfg_.n_layers], d_hidden_,
                                    needed * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream_));
    }

    // Final norm — use temp buffers
    float* normed;
    CUDA_CHECK(cudaMalloc(&normed, needed * sizeof(float)));
    rms_norm_forward(d_hidden_, d_final_norm_, normed, d_final_rms_,
                     BS, D, cfg_.rms_eps, stream_);

    // Logits
    linear_forward(normed, d_output_w_, d_logits, BS, cfg_.vocab_size, D,
                   cublas_, stream_);

    // Copy normed for backward if needed
    CUDA_CHECK(cudaMemcpyAsync(d_final_normed_, normed, needed * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaFree(normed));
}

void ResonanceNet::layer_forward(int l, float* d_hidden, int batch, int seq_len) {
    int D = cfg_.d_model;
    int BS = batch * seq_len;
    auto& w = layers_[l];
    auto& c = cache_[l];

    // Save input for backward
    // (residual stream is d_hidden itself)

    // ── 1. Conv sub-layer ──
    rms_norm_forward(d_hidden, w.norm1_w, c.normed1, c.rms1,
                     BS, D, cfg_.rms_eps, stream_);

    float* conv_out;
    CUDA_CHECK(cudaMalloc(&conv_out, (size_t)BS * D * sizeof(float)));
    causal_conv_forward(c.normed1, conv_out, w.conv, c.conv_buf,
                        batch, seq_len, D, cublas_, stream_);
    residual_add(d_hidden, conv_out, BS * D, stream_);
    CUDA_CHECK(cudaMemcpyAsync(c.res_after_conv, d_hidden,
                                (size_t)BS * D * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaFree(conv_out));

    // ── 2. MinGRU sub-layer ──
    rms_norm_forward(d_hidden, w.norm2_w, c.normed2, c.rms2,
                     BS, D, cfg_.rms_eps, stream_);

    float* gru_out;
    CUDA_CHECK(cudaMalloc(&gru_out, (size_t)BS * D * sizeof(float)));
    min_gru_forward(c.normed2, gru_out, w.gru, c.z_cache, c.h_cache,
                    batch, seq_len, D, cublas_, stream_);
    residual_add(d_hidden, gru_out, BS * D, stream_);
    CUDA_CHECK(cudaMemcpyAsync(c.res_after_gru, d_hidden,
                                (size_t)BS * D * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaFree(gru_out));

    // ── 3. Slot Memory sub-layer ──
    rms_norm_forward(d_hidden, w.norm3_w, c.normed3, c.rms3,
                     BS, D, cfg_.rms_eps, stream_);

    float* slot_out;
    CUDA_CHECK(cudaMalloc(&slot_out, (size_t)BS * D * sizeof(float)));
    slot_memory_forward(c.normed3, slot_out, w.slot,
                        c.q_cache, c.scores_cache,
                        c.topk_idx, c.topk_w,
                        batch, seq_len, D, cfg_.n_slots, cfg_.slot_top_k,
                        cublas_, stream_);
    residual_add(d_hidden, slot_out, BS * D, stream_);
    CUDA_CHECK(cudaMemcpyAsync(c.res_after_slot, d_hidden,
                                (size_t)BS * D * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaFree(slot_out));

    // ── 4. SwiGLU FFN sub-layer ──
    rms_norm_forward(d_hidden, w.norm4_w, c.normed4, c.rms4,
                     BS, D, cfg_.rms_eps, stream_);

    float* ffn_out;
    CUDA_CHECK(cudaMalloc(&ffn_out, (size_t)BS * D * sizeof(float)));
    swiglu_forward(c.normed4, ffn_out, w.ffn,
                   c.gate_cache, c.up_cache,
                   BS, D, cfg_.d_ff, cublas_, stream_);
    residual_add(d_hidden, ffn_out, BS * D, stream_);
    CUDA_CHECK(cudaFree(ffn_out));
}

// ─── Inference-only forward (allocates temp buffers, no caching) ─────
void ResonanceNet::layer_forward_infer(int l, float* d_hidden, int batch, int seq_len) {
    int D = cfg_.d_model;
    int BS = batch * seq_len;
    int NS = cfg_.n_slots;
    int FF = cfg_.d_ff;
    auto& w = layers_[l];

    size_t sz = (size_t)BS * D;
    float *normed, *sub_out, *conv_buf, *z_tmp, *h_tmp;
    float *q_tmp, *sc_tmp, *tw_tmp, *gate_tmp, *up_tmp;
    int *ti_tmp;

    CUDA_CHECK(cudaMalloc(&normed, sz * sizeof(float)));

    // ── 1. Conv ──
    CUDA_CHECK(cudaMalloc(&conv_buf, 3 * sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub_out, sz * sizeof(float)));
    rms_norm_forward(d_hidden, w.norm1_w, normed, nullptr, BS, D, cfg_.rms_eps, stream_);
    causal_conv_forward(normed, sub_out, w.conv, conv_buf, batch, seq_len, D, cublas_, stream_);
    residual_add(d_hidden, sub_out, BS * D, stream_);
    CUDA_CHECK(cudaFree(conv_buf));
    CUDA_CHECK(cudaFree(sub_out));

    // ── 2. MinGRU ──
    CUDA_CHECK(cudaMalloc(&z_tmp, sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&h_tmp, sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub_out, sz * sizeof(float)));
    rms_norm_forward(d_hidden, w.norm2_w, normed, nullptr, BS, D, cfg_.rms_eps, stream_);
    min_gru_forward(normed, sub_out, w.gru, z_tmp, h_tmp, batch, seq_len, D, cublas_, stream_);
    residual_add(d_hidden, sub_out, BS * D, stream_);
    CUDA_CHECK(cudaFree(z_tmp));
    CUDA_CHECK(cudaFree(h_tmp));
    CUDA_CHECK(cudaFree(sub_out));

    // ── 3. Slot Memory ──
    CUDA_CHECK(cudaMalloc(&q_tmp, sz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sc_tmp, (size_t)BS * NS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ti_tmp, (size_t)BS * cfg_.slot_top_k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&tw_tmp, (size_t)BS * cfg_.slot_top_k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub_out, sz * sizeof(float)));
    rms_norm_forward(d_hidden, w.norm3_w, normed, nullptr, BS, D, cfg_.rms_eps, stream_);
    slot_memory_forward(normed, sub_out, w.slot, q_tmp, sc_tmp, ti_tmp, tw_tmp,
                        batch, seq_len, D, NS, cfg_.slot_top_k, cublas_, stream_);
    residual_add(d_hidden, sub_out, BS * D, stream_);
    CUDA_CHECK(cudaFree(q_tmp));
    CUDA_CHECK(cudaFree(sc_tmp));
    CUDA_CHECK(cudaFree(ti_tmp));
    CUDA_CHECK(cudaFree(tw_tmp));
    CUDA_CHECK(cudaFree(sub_out));

    // ── 4. SwiGLU FFN ──
    CUDA_CHECK(cudaMalloc(&gate_tmp, (size_t)BS * FF * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&up_tmp, (size_t)BS * FF * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sub_out, sz * sizeof(float)));
    rms_norm_forward(d_hidden, w.norm4_w, normed, nullptr, BS, D, cfg_.rms_eps, stream_);
    swiglu_forward(normed, sub_out, w.ffn, gate_tmp, up_tmp, BS, D, FF, cublas_, stream_);
    residual_add(d_hidden, sub_out, BS * D, stream_);
    CUDA_CHECK(cudaFree(gate_tmp));
    CUDA_CHECK(cudaFree(up_tmp));
    CUDA_CHECK(cudaFree(sub_out));

    CUDA_CHECK(cudaFree(normed));
}

// ─── Backward Pass ──────────────────────────────────────────────────
void ResonanceNet::backward(const float* d_dlogits, const int* d_tokens, int batch, int seq_len) {
    int D = cfg_.d_model;
    int V = cfg_.vocab_size;
    int BS = batch * seq_len;
    float alpha = 1.0f, beta_zero = 0.0f, beta_one = 1.0f;
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));

    if (!d_dhidden_) {
        alloc_gpu(&d_dhidden_, (size_t)BS * D);
    }

    // ── Backprop through output projection ──
    // d_logits = normed @ output_w^T
    // d_normed = d_logits @ output_w   [BS, D]
    float* d_dnormed;
    CUDA_CHECK(cudaMalloc(&d_dnormed, (size_t)BS * D * sizeof(float)));

    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                             D, BS, V,
                             &alpha, d_output_w_, D, d_dlogits, V,
                             &beta_zero, d_dnormed, D));

    // d_output_w += normed^T @ d_logits  [V, D]
    CUBLAS_CHECK(cublasSgemm(cublas_, CUBLAS_OP_N, CUBLAS_OP_T,
                             D, V, BS,
                             &alpha, d_final_normed_, D, d_dlogits, V,
                             &beta_one, d_output_g_, D));

    // ── Backprop through final RMSNorm ──
    rms_norm_backward(d_dnormed, d_layer_inputs_[cfg_.n_layers], d_final_norm_,
                      d_final_rms_, d_dhidden_, d_final_norm_g_,
                      BS, D, stream_);

    CUDA_CHECK(cudaFree(d_dnormed));

    // ── Backprop through layers in reverse ──
    for (int l = cfg_.n_layers - 1; l >= 0; l--) {
        layer_backward(l, d_dhidden_, batch, seq_len);
    }

    // ── Backprop through embedding ──
    embed_backward(d_dhidden_, d_tokens, d_embed_g_, BS, V, D, stream_);
}

void ResonanceNet::layer_backward(int l, float* d_dhidden, int batch, int seq_len) {
    int D = cfg_.d_model;
    int BS = batch * seq_len;
    auto& w = layers_[l];
    auto& g = grads_[l];
    auto& c = cache_[l];

    // d_dhidden comes in as gradient of residual stream after this layer
    // We need to backprop through: FFN → slot → GRU → conv (reverse order)

    float* d_dsub;  // gradient for sub-layer output
    CUDA_CHECK(cudaMalloc(&d_dsub, (size_t)BS * D * sizeof(float)));

    // ── 4. SwiGLU FFN backward ──
    // Input to FFN was c.normed4, cached in forward
    // d_dhidden flows through residual (unchanged) + through FFN
    float* dx_ffn;
    CUDA_CHECK(cudaMalloc(&dx_ffn, (size_t)BS * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(dx_ffn, 0, (size_t)BS * D * sizeof(float)));

    swiglu_backward(d_dhidden, c.normed4, w.ffn,
                    c.gate_cache, c.up_cache,
                    dx_ffn, g.ffn,
                    BS, D, cfg_.d_ff, cublas_, stream_);

    // dx_ffn is gradient w.r.t. normed4 → backprop through RMSNorm4
    // Input to norm4 was res_after_slot
    float* dx_norm4;
    CUDA_CHECK(cudaMalloc(&dx_norm4, (size_t)BS * D * sizeof(float)));
    rms_norm_backward(dx_ffn, c.res_after_slot, w.norm4_w, c.rms4,
                      dx_norm4, g.norm4_w, BS, D, stream_);

    // Accumulate: d_dhidden += dx_norm4 (residual path)
    residual_add(d_dhidden, dx_norm4, BS * D, stream_);
    CUDA_CHECK(cudaFree(dx_ffn));
    CUDA_CHECK(cudaFree(dx_norm4));

    // ── 3. Slot Memory backward ──
    float* dx_slot;
    CUDA_CHECK(cudaMalloc(&dx_slot, (size_t)BS * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(dx_slot, 0, (size_t)BS * D * sizeof(float)));

    slot_memory_backward(d_dhidden, c.normed3, w.slot,
                         c.q_cache, c.scores_cache,
                         c.topk_idx, c.topk_w,
                         dx_slot, g.slot,
                         batch, seq_len, D, cfg_.n_slots, cfg_.slot_top_k,
                         cublas_, stream_);

    float* dx_norm3;
    CUDA_CHECK(cudaMalloc(&dx_norm3, (size_t)BS * D * sizeof(float)));
    rms_norm_backward(dx_slot, c.res_after_gru, w.norm3_w, c.rms3,
                      dx_norm3, g.norm3_w, BS, D, stream_);
    residual_add(d_dhidden, dx_norm3, BS * D, stream_);
    CUDA_CHECK(cudaFree(dx_slot));
    CUDA_CHECK(cudaFree(dx_norm3));

    // ── 2. MinGRU backward ──
    float* dx_gru;
    CUDA_CHECK(cudaMalloc(&dx_gru, (size_t)BS * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(dx_gru, 0, (size_t)BS * D * sizeof(float)));

    min_gru_backward(d_dhidden, c.normed2, w.gru,
                     c.z_cache, c.h_cache,
                     dx_gru, g.gru,
                     batch, seq_len, D, cublas_, stream_);

    float* dx_norm2;
    CUDA_CHECK(cudaMalloc(&dx_norm2, (size_t)BS * D * sizeof(float)));
    rms_norm_backward(dx_gru, c.res_after_conv, w.norm2_w, c.rms2,
                      dx_norm2, g.norm2_w, BS, D, stream_);
    residual_add(d_dhidden, dx_norm2, BS * D, stream_);
    CUDA_CHECK(cudaFree(dx_gru));
    CUDA_CHECK(cudaFree(dx_norm2));

    // ── 1. Causal Conv backward ──
    float* dx_conv;
    CUDA_CHECK(cudaMalloc(&dx_conv, (size_t)BS * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(dx_conv, 0, (size_t)BS * D * sizeof(float)));

    causal_conv_backward(d_dhidden, c.normed1, w.conv,
                         c.conv_buf, dx_conv, g.conv,
                         batch, seq_len, D, cublas_, stream_);

    float* dx_norm1;
    CUDA_CHECK(cudaMalloc(&dx_norm1, (size_t)BS * D * sizeof(float)));
    rms_norm_backward(dx_conv, d_layer_inputs_[l], w.norm1_w, c.rms1,
                      dx_norm1, g.norm1_w, BS, D, stream_);
    residual_add(d_dhidden, dx_norm1, BS * D, stream_);
    CUDA_CHECK(cudaFree(dx_conv));
    CUDA_CHECK(cudaFree(dx_norm1));

    CUDA_CHECK(cudaFree(d_dsub));
}

// ─── Optimizer Step (SGD) ───────────────────────────────────────────
void ResonanceNet::step(const TrainConfig& tcfg, int step_num) {
    // Plain SGD: param += lr * grad (delta accumulation, original ResonanceNet)
    float lr = tcfg.lr;
    int D = cfg_.d_model, V = cfg_.vocab_size, FF = cfg_.d_ff, NS = cfg_.n_slots;

    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));

    // DDP: sum gradients across ranks before computing the update.
    // Batched into a single NCCL group op to amortize launch overhead.
    if (dist_ && dist_->active()) {
        auto reduce = [&](float* g, int count) {
            dist_allreduce_sum(*dist_, g, (size_t)count, stream_);
        };
        dist_group_start(*dist_);
        reduce(d_embed_g_, V * D);
        reduce(d_output_g_, V * D);
        reduce(d_final_norm_g_, D);
        for (int l = 0; l < cfg_.n_layers; l++) {
            auto& g = grads_[l];
            reduce(g.norm1_w, D); reduce(g.norm2_w, D);
            reduce(g.norm3_w, D); reduce(g.norm4_w, D);
            reduce(g.conv.conv3_w, 3*D); reduce(g.conv.conv7_w, 7*D);
            reduce(g.conv.conv15_w, 15*D); reduce(g.conv.mix_w, D*D);
            reduce(g.gru.Wz, D*D); reduce(g.gru.Wh, D*D);
            reduce(g.gru.bz, D); reduce(g.gru.bh, D);
            reduce(g.slot.slot_keys, D*NS); reduce(g.slot.slot_values, D*NS);
            reduce(g.slot.proj_q, D*D); reduce(g.slot.proj_out, D*D);
            reduce(g.ffn.gate_w, D*FF); reduce(g.ffn.up_w, D*FF);
            reduce(g.ffn.down_w, FF*D);
        }
        dist_group_end(*dist_);
        // Grads are now SUM across world_size ranks. Averaging is folded
        // into the effective learning rate below so we don't need a second
        // kernel pass over every tensor.
        lr /= (float)dist_->world_size;
    }

    // Gradient clipping: compute global norm, clip if > 1.0
    float grad_norm_sq = 0.0f;
    auto accum_norm = [&](float* grad, int count) {
        float result;
        CUBLAS_CHECK(cublasSdot(cublas_, count, grad, 1, grad, 1, &result));
        grad_norm_sq += result;
    };
    // Compute norm over all grads
    accum_norm(d_embed_g_, V * D);
    accum_norm(d_output_g_, V * D);
    accum_norm(d_final_norm_g_, D);
    for (int l = 0; l < cfg_.n_layers; l++) {
        auto& g = grads_[l];
        accum_norm(g.norm1_w, D); accum_norm(g.norm2_w, D);
        accum_norm(g.norm3_w, D); accum_norm(g.norm4_w, D);
        accum_norm(g.conv.conv3_w, 3*D); accum_norm(g.conv.conv7_w, 7*D);
        accum_norm(g.conv.conv15_w, 15*D); accum_norm(g.conv.mix_w, D*D);
        accum_norm(g.gru.Wz, D*D); accum_norm(g.gru.Wh, D*D);
        accum_norm(g.gru.bz, D); accum_norm(g.gru.bh, D);
        accum_norm(g.slot.slot_keys, D*NS); accum_norm(g.slot.slot_values, D*NS);
        accum_norm(g.slot.proj_q, D*D); accum_norm(g.slot.proj_out, D*D);
        accum_norm(g.ffn.gate_w, D*FF); accum_norm(g.ffn.up_w, D*FF);
        accum_norm(g.ffn.down_w, FF*D);
    }
    float grad_norm = sqrtf(grad_norm_sq);
    // When DDP is active, grads hold the SUM across ranks. The norm of the
    // average gradient — which is what the lr-scaled update effectively uses —
    // is grad_norm / world_size. Clip against that so behaviour matches single-GPU.
    float world = (dist_ && dist_->active()) ? (float)dist_->world_size : 1.0f;
    float avg_grad_norm = grad_norm / world;
    float clip = tcfg.grad_clip;
    if (clip > 0 && avg_grad_norm > clip) {
        lr *= clip / avg_grad_norm;
    }

    float neg_lr = -lr;

    auto sgd = [&](float* param, float* grad, int count) {
        CUBLAS_CHECK(cublasSaxpy(cublas_, count, &neg_lr, grad, 1, param, 1));
    };

    // Global weights
    sgd(d_embed_w_, d_embed_g_, V * D);
    sgd(d_output_w_, d_output_g_, V * D);
    sgd(d_final_norm_, d_final_norm_g_, D);

    // Per-layer
    for (int l = 0; l < cfg_.n_layers; l++) {
        auto& w = layers_[l];
        auto& g = grads_[l];

        sgd(w.norm1_w, g.norm1_w, D);
        sgd(w.norm2_w, g.norm2_w, D);
        sgd(w.norm3_w, g.norm3_w, D);
        sgd(w.norm4_w, g.norm4_w, D);

        sgd(w.conv.conv3_w, g.conv.conv3_w, 3 * D);
        sgd(w.conv.conv7_w, g.conv.conv7_w, 7 * D);
        sgd(w.conv.conv15_w, g.conv.conv15_w, 15 * D);
        sgd(w.conv.mix_w, g.conv.mix_w, D * D);

        sgd(w.gru.Wz, g.gru.Wz, D * D);
        sgd(w.gru.Wh, g.gru.Wh, D * D);
        sgd(w.gru.bz, g.gru.bz, D);
        sgd(w.gru.bh, g.gru.bh, D);

        sgd(w.slot.slot_keys, g.slot.slot_keys, D * NS);
        sgd(w.slot.slot_values, g.slot.slot_values, D * NS);
        sgd(w.slot.proj_q, g.slot.proj_q, D * D);
        sgd(w.slot.proj_out, g.slot.proj_out, D * D);

        sgd(w.ffn.gate_w, g.ffn.gate_w, D * FF);
        sgd(w.ffn.up_w, g.ffn.up_w, D * FF);
        sgd(w.ffn.down_w, g.ffn.down_w, FF * D);
    }
}

void ResonanceNet::zero_grad() {
    int D = cfg_.d_model;
    int V = cfg_.vocab_size;
    int FF = cfg_.d_ff;
    int NS = cfg_.n_slots;

    CUDA_CHECK(cudaMemsetAsync(d_embed_g_, 0, (size_t)V * D * sizeof(float), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_output_g_, 0, (size_t)V * D * sizeof(float), stream_));
    CUDA_CHECK(cudaMemsetAsync(d_final_norm_g_, 0, D * sizeof(float), stream_));

    for (int l = 0; l < cfg_.n_layers; l++) {
        auto& g = grads_[l];
        CUDA_CHECK(cudaMemsetAsync(g.norm1_w, 0, D * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.norm2_w, 0, D * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.norm3_w, 0, D * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.norm4_w, 0, D * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.conv.conv3_w, 0, 3 * D * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.conv.conv7_w, 0, 7 * D * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.conv.conv15_w, 0, 15 * D * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.conv.mix_w, 0, (size_t)D * D * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.gru.Wz, 0, (size_t)D * D * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.gru.Wh, 0, (size_t)D * D * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.gru.bz, 0, D * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.gru.bh, 0, D * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.slot.slot_keys, 0, (size_t)D * NS * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.slot.slot_values, 0, (size_t)D * NS * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.slot.proj_q, 0, (size_t)D * D * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.slot.proj_out, 0, (size_t)D * D * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.ffn.gate_w, 0, (size_t)D * FF * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.ffn.up_w, 0, (size_t)D * FF * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(g.ffn.down_w, 0, (size_t)FF * D * sizeof(float), stream_));
    }
}

void ResonanceNet::destroy() {
    // Free all GPU memory
    auto sf = [](auto& p) { if (p) { cudaFree(p); p = nullptr; } };
    sf(d_embed_w_); sf(d_output_w_); sf(d_final_norm_);
    sf(d_embed_g_); sf(d_output_g_); sf(d_final_norm_g_);
    sf(d_hidden_); sf(d_dhidden_); sf(d_logits_buf_);
    sf(d_final_normed_); sf(d_final_rms_);
    for (auto& p : d_layer_inputs_) sf(p);
    d_layer_inputs_.clear();
    // No optimizer states to free (plain SGD)

    for (int l = 0; l < (int)layers_.size(); l++) {
        auto& w = layers_[l];
        auto& g = grads_[l];
        auto& c = cache_[l];

        sf(w.norm1_w); sf(w.norm2_w); sf(w.norm3_w); sf(w.norm4_w);
        sf(w.conv.conv3_w); sf(w.conv.conv7_w); sf(w.conv.conv15_w); sf(w.conv.mix_w);
        sf(w.gru.Wz); sf(w.gru.Wh); sf(w.gru.bz); sf(w.gru.bh);
        sf(w.slot.slot_keys); sf(w.slot.slot_values); sf(w.slot.proj_q); sf(w.slot.proj_out);
        sf(w.ffn.gate_w); sf(w.ffn.up_w); sf(w.ffn.down_w);

        sf(g.norm1_w); sf(g.norm2_w); sf(g.norm3_w); sf(g.norm4_w);
        sf(g.conv.conv3_w); sf(g.conv.conv7_w); sf(g.conv.conv15_w); sf(g.conv.mix_w);
        sf(g.gru.Wz); sf(g.gru.Wh); sf(g.gru.bz); sf(g.gru.bh);
        sf(g.slot.slot_keys); sf(g.slot.slot_values); sf(g.slot.proj_q); sf(g.slot.proj_out);
        sf(g.ffn.gate_w); sf(g.ffn.up_w); sf(g.ffn.down_w);

        sf(c.rms1); sf(c.rms2); sf(c.rms3); sf(c.rms4);
        sf(c.normed1); sf(c.normed2); sf(c.normed3); sf(c.normed4);
        sf(c.conv_buf);
        sf(c.z_cache); sf(c.h_cache);
        sf(c.q_cache); sf(c.scores_cache);
        if (c.topk_idx) { cudaFree(c.topk_idx); c.topk_idx = nullptr; }
        sf(c.topk_w);
        sf(c.gate_cache); sf(c.up_cache);
        sf(c.res_after_conv); sf(c.res_after_gru); sf(c.res_after_slot);
    }

    if (cublas_) { cublasDestroy(cublas_); cublas_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

// ─── Save/Load checkpoints ──────────────────────────────────────────
bool ResonanceNet::save(const std::string& path) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;

    // Write config
    fwrite(&cfg_, sizeof(ModelConfig), 1, f);

    // Helper to save GPU tensor
    auto save_tensor = [&](float* d_ptr, size_t count) {
        std::vector<float> h(count);
        cudaMemcpy(h.data(), d_ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
        fwrite(h.data(), sizeof(float), count, f);
    };

    int D = cfg_.d_model, V = cfg_.vocab_size, FF = cfg_.d_ff, NS = cfg_.n_slots;

    save_tensor(d_embed_w_, (size_t)V * D);
    save_tensor(d_output_w_, (size_t)V * D);
    save_tensor(d_final_norm_, D);

    for (int l = 0; l < cfg_.n_layers; l++) {
        auto& w = layers_[l];
        save_tensor(w.norm1_w, D); save_tensor(w.norm2_w, D);
        save_tensor(w.norm3_w, D); save_tensor(w.norm4_w, D);
        save_tensor(w.conv.conv3_w, 3*D); save_tensor(w.conv.conv7_w, 7*D);
        save_tensor(w.conv.conv15_w, 15*D); save_tensor(w.conv.mix_w, (size_t)D*D);
        save_tensor(w.gru.Wz, (size_t)D*D); save_tensor(w.gru.Wh, (size_t)D*D);
        save_tensor(w.gru.bz, D); save_tensor(w.gru.bh, D);
        save_tensor(w.slot.slot_keys, (size_t)D*NS);
        save_tensor(w.slot.slot_values, (size_t)D*NS);
        save_tensor(w.slot.proj_q, (size_t)D*D);
        save_tensor(w.slot.proj_out, (size_t)D*D);
        save_tensor(w.ffn.gate_w, (size_t)D*FF);
        save_tensor(w.ffn.up_w, (size_t)D*FF);
        save_tensor(w.ffn.down_w, (size_t)FF*D);
    }

    fclose(f);
    return true;
}

bool ResonanceNet::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    ModelConfig loaded_cfg;
    fread(&loaded_cfg, sizeof(ModelConfig), 1, f);

    if (!d_embed_w_) init(loaded_cfg);

    auto load_tensor = [&](float* d_ptr, size_t count) {
        std::vector<float> h(count);
        fread(h.data(), sizeof(float), count, f);
        cudaMemcpy(d_ptr, h.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    };

    int D = cfg_.d_model, V = cfg_.vocab_size, FF = cfg_.d_ff, NS = cfg_.n_slots;

    load_tensor(d_embed_w_, (size_t)V * D);
    load_tensor(d_output_w_, (size_t)V * D);
    load_tensor(d_final_norm_, D);

    for (int l = 0; l < cfg_.n_layers; l++) {
        auto& w = layers_[l];
        load_tensor(w.norm1_w, D); load_tensor(w.norm2_w, D);
        load_tensor(w.norm3_w, D); load_tensor(w.norm4_w, D);
        load_tensor(w.conv.conv3_w, 3*D); load_tensor(w.conv.conv7_w, 7*D);
        load_tensor(w.conv.conv15_w, 15*D); load_tensor(w.conv.mix_w, (size_t)D*D);
        load_tensor(w.gru.Wz, (size_t)D*D); load_tensor(w.gru.Wh, (size_t)D*D);
        load_tensor(w.gru.bz, D); load_tensor(w.gru.bh, D);
        load_tensor(w.slot.slot_keys, (size_t)D*NS);
        load_tensor(w.slot.slot_values, (size_t)D*NS);
        load_tensor(w.slot.proj_q, (size_t)D*D);
        load_tensor(w.slot.proj_out, (size_t)D*D);
        load_tensor(w.ffn.gate_w, (size_t)D*FF);
        load_tensor(w.ffn.up_w, (size_t)D*FF);
        load_tensor(w.ffn.down_w, (size_t)FF*D);
    }

    fclose(f);
    return true;
}

} // namespace rnet
