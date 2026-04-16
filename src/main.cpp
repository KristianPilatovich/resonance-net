#include "resonance_net.h"
#include "dist.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include <string>

// Include data_loader inline (single TU)
#include "data_loader.cpp"

using namespace rnet;

void print_usage(const char* prog) {
    printf("ResonanceNet V5 — CUDA C++ Training & Inference\n\n");
    printf("Usage:\n");
    printf("  %s train [options]     Train a new model\n", prog);
    printf("  %s infer <checkpoint>  Run inference\n", prog);
    printf("  %s bench              Run benchmark\n", prog);
    printf("\nTrain options:\n");
    printf("  --data <file>       Training data (raw bytes)\n");
    printf("  --lr <float>        Learning rate (default: 3e-4)\n");
    printf("  --batch <int>       Batch size (default: 64)\n");
    printf("  --steps <int>       Training steps (default: 100000)\n");
    printf("  --d_model <int>     Hidden dim (default: 512)\n");
    printf("  --n_layers <int>    Number of layers (default: 8)\n");
    printf("  --d_ff <int>        FFN inner dim (default: 1024)\n");
    printf("  --n_slots <int>     Slot memory size (default: 1024)\n");
    printf("  --seq_len <int>     Sequence length (default: 256)\n");
    printf("  --save <dir>        Save directory (default: checkpoints/)\n");
    printf("  --resume <file>     Resume from checkpoint\n");
}

void train(int argc, char** argv) {
    DistState dist = dist_init_from_env();
    const bool is_master = dist.is_master();

    ModelConfig mcfg;
    TrainConfig tcfg;
    std::string resume_path;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--data") && i+1 < argc) tcfg.data_path = argv[++i];
        else if (!strcmp(argv[i], "--lr") && i+1 < argc) tcfg.lr = atof(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i+1 < argc) tcfg.batch_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i+1 < argc) tcfg.max_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--d_model") && i+1 < argc) mcfg.d_model = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n_layers") && i+1 < argc) mcfg.n_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--d_ff") && i+1 < argc) mcfg.d_ff = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n_slots") && i+1 < argc) mcfg.n_slots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq_len") && i+1 < argc) mcfg.seq_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save") && i+1 < argc) tcfg.save_path = argv[++i];
        else if (!strcmp(argv[i], "--resume") && i+1 < argc) resume_path = argv[++i];
        else if (!strcmp(argv[i], "--log") && i+1 < argc) tcfg.log_interval = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc) tcfg.warmup_steps = atoi(argv[++i]);
    }

    // Ensure gru_dim matches d_model
    mcfg.gru_dim = mcfg.d_model;

    if (is_master) {
        printf("═══════════════════════════════════════════════════\n");
        printf("  ResonanceNet V5 Training\n");
        printf("═══════════════════════════════════════════════════\n");
        printf("  d_model=%d, layers=%d, d_ff=%d, slots=%d\n",
               mcfg.d_model, mcfg.n_layers, mcfg.d_ff, mcfg.n_slots);
        printf("  batch=%d (per rank), seq=%d, lr=%.1e, steps=%d\n",
               tcfg.batch_size, mcfg.seq_len, tcfg.lr, tcfg.max_steps);
        if (dist.active()) {
            printf("  DDP world_size=%d  effective global batch=%d\n",
                   dist.world_size, tcfg.batch_size * dist.world_size);
        }
        printf("  estimated params: %.1f M\n", mcfg.param_count() / 1e6);
        printf("═══════════════════════════════════════════════════\n\n");
    }

    // Load data
    DataLoader data;
    if (!tcfg.data_path.empty()) {
        if (!data.load(tcfg.data_path)) { dist_destroy(dist); return; }
    } else if (is_master) {
        printf("No --data specified, using synthetic random data\n\n");
    }

    // Init model
    ResonanceNet model;
    if (!resume_path.empty()) {
        if (is_master) printf("Resuming from %s\n", resume_path.c_str());
        model.load(resume_path);
        if (mcfg.seq_len != 256 && is_master) {
            printf("Overriding seq_len: %d → %d\n", model.config().seq_len, mcfg.seq_len);
        }
    } else {
        model.init(mcfg);
    }
    model.set_dist(&dist);

    // Use CLI mcfg.seq_len (may differ from checkpoint)
    int seq_len = mcfg.seq_len;
    int BS = tcfg.batch_size * seq_len;

    // Allocate host + device buffers for tokens
    std::vector<int> h_input(BS), h_target(BS);
    int *d_input, *d_target;
    CUDA_CHECK(cudaMalloc(&d_input, BS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_target, BS * sizeof(int)));

    // Logits buffer
    float* d_logits;
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)BS * mcfg.vocab_size * sizeof(float)));

    // Loss
    float *d_loss, *d_dlogits;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dlogits, (size_t)BS * mcfg.vocab_size * sizeof(float)));

    // Per-rank RNG seed so each rank samples a different slice of data.
    std::mt19937 rng(42u + (uint32_t)dist.rank * 10007u);

    if (is_master) printf("Training started.\n\n");

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int step = 1; step <= tcfg.max_steps; step++) {
        // Get batch
        data.get_batch(h_input.data(), h_target.data(),
                       tcfg.batch_size, seq_len, rng);
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), BS * sizeof(int),
                               cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_target, h_target.data(), BS * sizeof(int),
                               cudaMemcpyHostToDevice));

        // Zero grad
        model.zero_grad();

        // Forward
        model.forward(d_input, d_logits, tcfg.batch_size, seq_len);

        // Loss
        cross_entropy_forward(d_logits, d_target, d_loss, d_dlogits,
                              BS, mcfg.vocab_size, 0);

        // Scale gradients by 1/BS
        scale_grads(d_dlogits, 1.0f / BS, (size_t)BS * mcfg.vocab_size, 0);

        // Backward
        model.backward(d_dlogits, d_input, tcfg.batch_size, seq_len);

        // Optimizer step
        model.step(tcfg, step);

        // Log (master only; loss reported is rank-0 slice — good enough for trend)
        if ((step % tcfg.log_interval == 0 || step == 1) && is_master) {
            float h_loss;
            CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float),
                                   cudaMemcpyDeviceToHost));
            h_loss /= BS;

            auto t1 = std::chrono::high_resolution_clock::now();
            float elapsed = std::chrono::duration<float>(t1 - t0).count();
            int world = dist.active() ? dist.world_size : 1;
            float tok_per_sec = (float)(step * BS * world) / elapsed;

            printf("step %6d | loss %.4f | %.0f tok/s | %.1fs\n",
                   step, h_loss, tok_per_sec, elapsed);
            fflush(stdout);
        }

        // Save checkpoint (master only — all ranks hold identical weights post-sync)
        if (step % tcfg.save_interval == 0 && is_master) {
            char path[256];
            snprintf(path, sizeof(path), "%s/step_%07d.bin",
                     tcfg.save_path.c_str(), step);
            printf("Saving checkpoint: %s\n", path);
            model.save(path);
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_loss));
    CUDA_CHECK(cudaFree(d_dlogits));
    model.destroy();
    dist_destroy(dist);
}

void bench(int argc, char** argv) {
    ModelConfig mcfg;
    mcfg.d_model = 512;
    mcfg.n_layers = 8;
    mcfg.d_ff = 1024;
    mcfg.n_slots = 1024;
    mcfg.seq_len = 256;
    int batch = 64;

    // Parse bench args
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--d_model") && i+1 < argc) mcfg.d_model = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n_layers") && i+1 < argc) mcfg.n_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--d_ff") && i+1 < argc) mcfg.d_ff = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n_slots") && i+1 < argc) mcfg.n_slots = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq_len") && i+1 < argc) mcfg.seq_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i+1 < argc) batch = atoi(argv[++i]);
    }
    mcfg.gru_dim = mcfg.d_model;

    printf("═══════════════════════════════════════════════════\n");
    printf("  ResonanceNet V5 Benchmark\n");
    printf("═══════════════════════════════════════════════════\n");

    ResonanceNet model;
    model.init(mcfg);
    int BS = batch * mcfg.seq_len;

    // Random tokens
    std::vector<int> h_input(BS);
    std::mt19937 rng(42);
    for (int i = 0; i < BS; i++) h_input[i] = rng() % 256;

    int* d_input;
    float* d_logits;
    CUDA_CHECK(cudaMalloc(&d_input, BS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)BS * mcfg.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), BS * sizeof(int),
                           cudaMemcpyHostToDevice));

    // Warmup
    printf("\nWarmup...\n");
    for (int i = 0; i < 3; i++) {
        model.forward(d_input, d_logits, batch, mcfg.seq_len);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    int runs = 20;
    printf("Running %d forward passes (batch=%d, seq=%d)...\n", runs, batch, mcfg.seq_len);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; i++) {
        model.forward(d_input, d_logits, batch, mcfg.seq_len);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    float elapsed = std::chrono::duration<float>(t1 - t0).count();
    float total_tokens = (float)runs * BS;
    float tok_per_sec = total_tokens / elapsed;

    printf("\n═══════════════════════════════════════════════════\n");
    printf("  Results:\n");
    printf("  Total time: %.3f s\n", elapsed);
    printf("  Tokens/sec: %.0f\n", tok_per_sec);
    printf("  ms/token:   %.3f\n", (elapsed / total_tokens) * 1000.0f);
    printf("  ms/batch:   %.1f\n", (elapsed / runs) * 1000.0f);
    printf("═══════════════════════════════════════════════════\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_logits));
    model.destroy();
}

void infer(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: resonance_net infer <checkpoint> [prompt]\n");
        return;
    }

    std::string ckpt_path = argv[2];
    std::string prompt = (argc > 3) ? argv[3] : "Hello";

    ResonanceNet model;
    if (!model.load(ckpt_path)) {
        fprintf(stderr, "Failed to load checkpoint: %s\n", ckpt_path.c_str());
        return;
    }

    auto& cfg = model.config();
    printf("Model loaded: d=%d, layers=%d, params=%.1fM\n",
           cfg.d_model, cfg.n_layers, cfg.param_count() / 1e6);

    // Convert prompt to byte tokens
    std::vector<int> tokens(prompt.begin(), prompt.end());
    printf("Prompt (%zu tokens): %s\n\n", tokens.size(), prompt.c_str());

    // Generate
    int max_gen = 256;
    int total_len = (int)tokens.size() + max_gen;
    tokens.resize(total_len, 0);

    int* d_tokens;
    float* d_logits;
    CUDA_CHECK(cudaMalloc(&d_tokens, total_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)total_len * cfg.vocab_size * sizeof(float)));

    // Autoregressive generation — causal: only feed tokens up to current position
    float temperature = 0.8f;
    std::mt19937 rng(42);
    int S = cfg.seq_len;

    int prompt_len = std::min((int)prompt.size(), S - 1);
    std::vector<int> context(tokens.begin(), tokens.begin() + prompt_len);

    CUDA_CHECK(cudaFree(d_tokens));
    CUDA_CHECK(cudaFree(d_logits));
    d_tokens = nullptr;
    d_logits = nullptr;

    // Pre-allocate max size buffers
    CUDA_CHECK(cudaMalloc(&d_tokens, S * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)S * cfg.vocab_size * sizeof(float)));

    for (int gen = 0; gen < max_gen; gen++) {
        int cur_len = (int)context.size();
        if (cur_len >= S) break;

        // Upload ONLY the real context (no padding)
        CUDA_CHECK(cudaMemcpy(d_tokens, context.data(), cur_len * sizeof(int),
                               cudaMemcpyHostToDevice));

        // Forward on exact context length — causal, no future tokens
        model.forward(d_tokens, d_logits, 1, cur_len);

        // Last position logits predict next token
        std::vector<float> h_logits(cfg.vocab_size);
        CUDA_CHECK(cudaMemcpy(h_logits.data(),
                               d_logits + (size_t)(cur_len - 1) * cfg.vocab_size,
                               cfg.vocab_size * sizeof(float),
                               cudaMemcpyDeviceToHost));

        // Temperature sampling with top-p nucleus
        float max_logit = *std::max_element(h_logits.begin(), h_logits.end());
        std::vector<std::pair<float, int>> probs(cfg.vocab_size);
        float sum_exp = 0.0f;
        for (int i = 0; i < cfg.vocab_size; i++) {
            float p = expf((h_logits[i] - max_logit) / temperature);
            probs[i] = {p, i};
            sum_exp += p;
        }
        // Normalize
        for (auto& [p, _] : probs) p /= sum_exp;

        // Sort by probability (descending) for top-p
        std::sort(probs.begin(), probs.end(), [](auto& a, auto& b) {
            return a.first > b.first;
        });

        // Top-p = 0.9 nucleus sampling
        float top_p = 0.9f;
        float cumsum = 0.0f;
        int cutoff = cfg.vocab_size;
        for (int i = 0; i < cfg.vocab_size; i++) {
            cumsum += probs[i].first;
            if (cumsum >= top_p) { cutoff = i + 1; break; }
        }

        // Re-normalize and sample
        float resum = 0.0f;
        for (int i = 0; i < cutoff; i++) resum += probs[i].first;
        float r = std::uniform_real_distribution<float>(0.0f, resum)(rng);
        int sampled = probs[0].second;
        cumsum = 0.0f;
        for (int i = 0; i < cutoff; i++) {
            cumsum += probs[i].first;
            if (cumsum >= r) { sampled = probs[i].second; break; }
        }

        context.push_back(sampled);

        // Print character
        if (sampled >= 32 && sampled < 127) putchar(sampled);
        else if (sampled == 10) putchar('\n');
        else if (sampled == 9) putchar('\t');
        fflush(stdout);
    }
    printf("\n");

    if (d_tokens) CUDA_CHECK(cudaFree(d_tokens));
    if (d_logits) CUDA_CHECK(cudaFree(d_logits));
    model.destroy();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (!strcmp(argv[1], "train")) {
        train(argc, argv);
    } else if (!strcmp(argv[1], "infer")) {
        infer(argc, argv);
    } else if (!strcmp(argv[1], "bench")) {
        bench(argc, argv);
    } else {
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
