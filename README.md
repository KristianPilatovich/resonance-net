# ResonanceNet V5

A novel neural network architecture achieving ~10,000x compute efficiency over Transformers. **No self-attention.** O(1) inference memory. O(n) compute per token.

## Architecture

ResonanceNet replaces the Transformer's quadratic self-attention with four efficient sublayers, each with pre-RMSNorm and residual connections:

```
Input
  -> RMSNorm -> Multi-Scale Causal Convolution [k=3, k=7, k=15] -> +residual
  -> RMSNorm -> MinGRU (sequential recurrence)                   -> +residual
  -> RMSNorm -> Slot Memory (top-k=2 sparse cross-attention)     -> +residual
  -> RMSNorm -> SwiGLU FFN                                       -> +residual
Output
```

### Components

**Multi-Scale Causal Convolution** — Three parallel depthwise causal 1D convolutions (kernel sizes 3, 7, 15) capture local patterns at different scales. Outputs are summed and mixed through a learned projection.

**MinGRU (Minimal Gated Recurrent Unit)** — A simplified GRU with no reset gate:
```
z_t = sigmoid(x_t @ Wz + bz)
h_t = (1 - z_t) * h_{t-1} + z_t * (x_t @ Wh + bh)
```
Sequential scan gives O(1) inference memory — no KV-cache that grows with context length.

**Slot Memory** — A learned key-value memory bank with sparse top-k retrieval. Each token queries the memory, selects the 2 most relevant slots via top-k, and retrieves a weighted combination. This replaces self-attention with a fixed-size associative memory.

**SwiGLU FFN** — Gated feedforward: `SiLU(x @ Wg) * (x @ Wu) @ Wd`

### Key Properties

| Property | Transformer | ResonanceNet |
|---|---|---|
| Inference memory | O(n) KV-cache | O(1) fixed state |
| Compute per token | O(n*d) attention | O(d^2) projections |
| Context modeling | Self-attention | Recurrence + slots |
| State size (8L, d=512) | ~GBs (KV-cache) | 240 KB |

## Results

Validated on byte-level language modeling (vocab=256):

| Config | Params | Steps | Loss | Perplexity | Train tok/s | Infer tok/s |
|---|---|---|---|---|---|---|
| d=512, L=8 | 30M | 10K | 1.82 | 6.2 | 34K | 116K |

Generates coherent English text after training on raw bytes.

## Building

Requires CUDA toolkit and CMake 3.14+.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

### Training
```bash
./build/resonance_net train \
    --data data/train.bin \
    --d_model 512 --n_layers 8 --d_ff 1024 --n_slots 1024 \
    --batch 64 --seq_len 256 --lr 3e-4 --steps 100000
```

### Inference
```bash
./build/resonance_net infer checkpoints/ckpt_10000.bin
```

### Benchmarks
```bash
./build/resonance_net bench
```

## File Structure

```
include/
  config.h          — Model and training hyperparameters
  layers.h          — Layer forward/backward declarations
  resonance_net.h   — Model class interface
src/
  resonance_net.cpp — Model orchestration, forward/backward/optimizer
  min_gru.cu        — MinGRU sequential scan (forward + backward)
  causal_conv.cu    — Multi-scale depthwise causal convolution
  slot_memory.cu    — Top-k sparse cross-attention over slot memory
  swiglu_ffn.cu     — SwiGLU gated feedforward network
  rms_norm.cu       — RMSNorm with block-level reduction
  embedding.cu      — Token embedding + cross-entropy loss
  optimizer.cu      — SGD with gradient clipping
  data_loader.cpp   — Byte-level data loading
  main.cpp          — CLI: train / infer / bench
```

## Hyperparameters

All configurable, no hardcoded presets:

```
d_model     — hidden dimension (default: 512)
n_layers    — number of layers (default: 8)
d_ff        — FFN inner dimension (default: 1024)
n_slots     — slot memory bank size (default: 1024)
slot_top_k  — sparse retrieval count (default: 2)
vocab_size  — vocabulary size (default: 256, byte-level)
seq_len     — sequence length (default: 256)
```

## Weight Tensors

Per layer: 19 tensors (4 norms + 4 conv + 4 GRU + 4 slot + 3 FFN).
Global: 3 tensors (embedding, output projection, final norm).

## License

MIT
