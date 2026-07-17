# Codebase Analysis: Persistent Kernel Scheduling for Long-Running Agentic Inference Loops

## File Inventory (in-scope files only)

| File | Lines | Role |
|------|-------|------|
| `paper.tex` | 696 | Self-contained NeurIPS-style paper, no .bbl needed |
| `persistent_kernel.cu` | 935 | CUDA reference implementation with clock64() instrumentation |
| `triton_persistent_agent_kernel.py` | 149 | Higher-level Triton/CUDA-graph prototype (scaffold) |
| `benchmark.py` | 394 | Benchmark harness: simulate / gpu / persistent-cuda modes |
| `gpu_results_device_sigma.json` | 20 | **Primary result** — clock64 sigma=0.09490us |
| `device_sigma_kernel.json` | 18 | Earlier run (sigma=0.09470us, consistent) |
| `gpu_results_v2.json` | 90 | 48-op chain kappa/sigma — backs Table 1 |
| `gpu_results_v3.json` | 116 | 48-op + real-model cross-check |
| `gpu_results.json` | 88 | Superseded single-op measurement |
| `sim_results.json` | 88 | CPU-only cost-model validation |

---

## persistent_kernel.cu — Deep Dive

### Transformer Config (hardcoded, lines 32-39)

```c
#define DIM          256
#define N_HEADS      4
#define HEAD_DIM     64
#define MLP_RATIO    4
#define FFN_DIM      1024
#define MAX_SEQ_LEN  512
#define N_LAYERS     2
```

Small model: 2 layers, dim=256, 4 heads. Matches benchmark.py `--real-model` defaults.

### Queue Protocol (lines 44-57)

```c
enum SlotState { SLOT_EMPTY=0, SLOT_READY=1, SLOT_DONE=2, SLOT_SHUTDOWN=3 };

struct QueueEntry {
    int32_t state;      // host writes READY, device sets DONE, host writes SHUTDOWN
    int32_t step_id;
    int32_t seq_len;
    float   input[256];
    float   output[256];
};
```

Allocated via `cudaHostAlloc` (mapped memory) so both host and device see the same physical address. Host writes to it from CPU; device polls via `volatile int*`.

### Weight Layout (lines 87-104)

Per-layer offsets (in float units) for 12 weight tensors:
```
[0]  ln1_weight  DIM
[1]  ln1_bias    DIM
[2]  qkv_weight  DIM * 3*DIM
[3]  qkv_bias    3*DIM
[4]  attn_o_wt   DIM * DIM
[5]  attn_o_bias DIM
[6]  ln2_weight  DIM
[7]  ln2_bias    DIM
[8]  fc1_weight  DIM * FFN_DIM
[9]  fc1_bias    FFN_DIM
[10] fc2_weight  FFN_DIM * DIM
[11] fc2_bias    DIM
```

LAYR_FLOATS = 2*DIM + DIM*3*DIM + 3*DIM + DIM*DIM + DIM + 2*DIM + DIM*FFN_DIM + FFN_DIM + FFN_DIM*DIM + DIM

For the small config: 2*256 + 256*768 + 768 + 65536 + 256 + 512 + 256*1024 + 1024 + 1024*256 + 256 = 524,288 floats = 2 MB per layer.

### KV Cache Layout (lines 106-116)

```
[layer][head][pos][dim]
K_cache[layer * N_HEADS * MAX_SEQ_LEN * HEAD_DIM + head * MAX_SEQ_LEN * HEAD_DIM + pos * HEAD_DIM + d]
```

Per layer: 4 heads * 512 seq * 64 dim = 131,072 floats = 512 KB (x2 for K and V = 1 MB/layer).

### Shared Memory Workspace (lines 348-364)

```c
struct DecodeWorkspace {
    float h[256];          // hidden state
    float residual[256];
    float normed[256];
    float q[256], k[256], v[256];
    float attn_out[256];
    float qkv[768];        // 3 * DIM
    float attn_proj[256];
    float fc1_out[1024];   // FFN_DIM
    float fc2_out[256];
    float q_head[64], k_head[64], v_head[64], out_head[64];  // HEAD_DIM
};
// Total: ~17 KB — fits in sm_89 shared mem (48 KB/block configurable)
```

### Device Helpers

- **`gelu_fwd`** (line 138): tanh approximation of GELU
- **`gemv`** (line 149): naive GEMV — all threads iterate rows, serial dot product per row. O(M*N) work, no tiling. Correctness over performance.
- **`layer_norm`** (line 165): mean → variance → normalize with warp-level reduction (sequential half-warps for the final 32 threads)
- **`attention_head`** (line 232): per-head attention with KV cache append. Steps:
  1. Append K/V to cache (coalesced writes by pos*dim)
  2. Compute scores via serial dot product (not tiled — O(seq_len * head_dim) per thread)
  3. Stable softmax (max reduction → exp → sum reduction → normalize)
  4. Weighted sum of V: `O(seq_len * head_dim)` per thread
  No shared-memory tiling of QK^T — works for small config but won't scale.

### Persistent Kernel (lines 491-564)

```
persistent_agent_kernel<<<1, 256>>>(weights, k_cache, v_cache, kv_len, d_queue, d_stats)
```

Single block, 256 threads. Only block 0 does work (line 499: `if (blockIdx.x != 0) return;`).

**Sigma instrumentation:**
```
thread 0 polls *queue_state (volatile) in a spin-loop
on READY/SHUTDOWN: s_t_ready = clock64()
__syncthreads() (broadcast to all warps)
t_compute_start = clock64()      ← sigma window ends here
decode_segment(...)
t_compute_end = clock64()        ← compute window ends here

sigma = t_compute_start - s_t_ready   (152 cycles avg)
compute = t_compute_end - t_compute_start  (~1.76M cycles avg)
```

The sigma window captures: queue-poll latency + thread-0 seeing READY + __syncthreads() barrier propagation time to wake all warps.

### Naive Baseline Kernel (lines 570-576)

```c
transformer_step_kernel<<<1, 256>>>(input, output, weights, k_cache, v_cache, kv_len)
```

Same forward pass, fresh launch each step. KV cache is persistent across steps (host resets kv_len to 0 between runs).

### Host Main (lines 608-934)

**Flow:**
1. Parse CLI args (`--steps`, `--tool-latency-ms`, `--json-out`, `--verbose`)
2. Query device properties, clock rate
3. Allocate weights, KV cache, queue, stats buffer
4. **Persistent mode:** launch kernel, T=10 warm-up steps (with simulated tool latency), then reset stats and run N_steps=50 timed steps, measure sigma+compute via clock64 + host wall time
5. **Naive mode:** same 50 steps with `transformer_step_kernel` launches, CUDA events for kernel-only timing
6. Derive kappa = naive_total_ms - naive_kernel_only_ms
7. Print/write JSON

**Critical correctness detail (line 705):** Stats are reset AFTER warm-up steps so sigma excludes the first few steps where L1/caches are cold. The warm-up includes `sleep_for(tool_latency_ms)` so the persistent kernel's poll loop is exercised realistically.

**Kappa derivation (line 870-871):**
```cpp
double compute_ms_cuda_events = naive_kernel_per_step;  // kernel-only CUDA events
double kappa_ms = naive_ms_per_step - compute_ms_cuda_events;
```
This gives kappa = H2D copy + launch overhead + teardown. The naive total includes `cudaMemcpy` per step (which is present in practice — new input tokens each iteration).

---

## triton_persistent_agent_kernel.py — Analysis

This is explicitly a scaffold (docstring line 13: "This is a SCAFFOLD"). Key limitations vs the CUDA version:

- **No device-side residency:** Triton doesn't support persistent kernels with spin-loops. Instead demonstrates CUDA-graph replay as the framework-level analogue.
- **Placeholder decode kernel:** Just copies input→output elementwise. The real compute is meant to be swapped in.
- **Two modes:**
  - `run_agent_loop_with_cuda_graph`: captures decode as CUDA graph, replays each step via `g.replay()`
  - `run_agent_loop_naive`: fresh Triton launch each step

The CUDA-graph version eliminates Python/driver dispatch overhead but still pays graph-node scheduling cost on the GPU side — hence σ is ~0.22ms in the 48-op chain benchmarks vs 0.095μs for true device-side residency.

---

## benchmark.py — Cost Model Engine

### Three Modes

#### 1. `--mode simulate` (lines 48-70)
CPU-only. Hardcoded κ=1.5ms, σ=κ/20. Random d_t, ℓ_t from regime distributions. Runs cost model formulas. No GPU needed.

#### 2. `--mode gpu` (lines 268-289)
Measures κ, σ via PyTorch ops:
- **κ:** time a chain of N sequential ops (add/mul/relu alternation) with naive kernel launches per op
- **σ:** time the same op chain via CUDA-graph replay (`g.replay()`)
- Optional `--real-model`: also benchmark actual PyTorch `TransformerLayer` stack and count kernels

The synthetic op chain approximates a decode pass whose per-operator launches accumulate. 48 ops ≈ 72 real kernels (attention has extra internal ops).

#### 3. `--mode persistent-cuda` (lines 292-337)
Shells out to `persistent_kernel.exe`, reads the JSON output. Uses **true device-side sigma** (clock64) and derived kappa from the real forward pass. This is the measurement that backs the decomposition in paper §6.1 point 5.

### Cost Model Rows (lines 73-91)
```python
def _cost_model_rows(regime_key, kappa, sigma, seed=0):
    for T in T_VALUES:
        d = random(lo, hi) for T steps
        ell = random(lo, hi) for T steps
        t_naive = sum(d) + T * kappa + sum(ell)
        t_persistent = sum(d) + T * sigma + sum(ell)
        delta = T * (kappa - sigma)
        delta_over_naive = delta / t_naive
```

### Real Model Comparison (lines 94-216)
Builds `nn.Sequential([TransformerLayer()] * n_layers)` with:
```
LayerNorm → MultiheadAttention → residual → LayerNorm → Linear→GELU→Linear → residual
```
Counts actual CUDA kernels via profiler, compares synthetic κ/σ against real κ/σ, suggests correct `--ops-per-step N`.

### Verification (lines 122-136)
```python
def _count_cuda_kernels(model, x):
    # Warmup, then profile to count kernel invocations
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        model(x)
    return sum(e.count for e in prof.key_averages())
```

---

## JSON Result Lineage

### `gpu_results.json` (superseded)
- Single trivial add op per step
- κ ~ 0.02ms, σ ~ 0.016ms
- **Problem:** wildly underestimated real decode overhead (paper text: "70× too low")

### `gpu_results_v2.json` (Table 1 data)
- 48-op chain per step (add/mul/relu alternation)
- **Regime A:** κ=0.92539ms, σ=0.22134ms → Δ/T_naive at T=100: **0.53%**
- **Regime B:** κ=0.88926ms, σ=0.22040ms → Δ/T_naive at T=100: **12.08%**
- This backs the "12-16%" figure and Table 1 in the paper

### `gpu_results_v3.json` (cross-check)
- Same 48-op chain + real-model comparison
- Lower κ (0.56-0.65ms) — different GPU state/temperature
- Real model κ=0.98-1.32ms, kernel count=72 per step
- Suggests --ops-per-step=72 would match better

### `gpu_results_device_sigma.json` (primary result — §6.1 point 5)
- **Real forward pass** from persistent_kernel.cu (2 layers, dim=256, 4 heads)
- κ_fused = **1.208μs** (derived from naive_total - cuda_event_compute)
- σ = **0.09490μs** (clock64 device-side, 152 cycles/step over 50 steps)
- clock64 compute = 1.0935ms, CUDA event compute = 0.7418ms

This is the file that enables the decomposition:
```
κ_unfused (48-op proxy) = 0.889ms
κ_fused (real fused kernel) = 1.208μs     ← fusion saves 99.9%
σ (device-side residency) = 0.095μs        ← residency saves 1.11μs more
Marginal benefit beyond fusion = 1.11μs/step  ← this paper's specific contribution
```

### `device_sigma_kernel.json` (earlier run)
- Same sigma (0.09470μs — within 0.2% of primary)
- κ=0 (bug in kappa derivation: naive_ms_per_step was larger than expected)
- Not used in paper — superseded by primary file

### `sim_results.json` (Figure 2 data)
- κ=1.5ms (modeled), σ=0.075ms (κ/20)
- Regime A: Δ/T_naive 0.8-1.07%
- Regime B: Δ/T_naive 18.6-23.2%
- The "~23%" simulated vs "~12%" measured gap is analyzed in §6.1

---

## Key Measurement Methodology Details

### clock64() Calibration
```
clock_rate = 1,605,000 kHz (from cudaDeviceGetAttribute)
cycles_to_ms = cycles / 1,605,000,000
```
1 clock64 tick ≈ 0.623 ns. 152 cycle sigma = 94.7 ns = 0.0947 μs.

### Why CUDA Events Show Lower Compute Than clock64
- CUDA events record start/stop at command processor level, exclude warp-scheduling overhead and barrier stalls
- clock64 includes all wall time from decode start to end, including __syncthreads() wait time
- Ratio: 0.7418ms (event) vs 1.0935ms (clock64) = ~32% lower — consistent with warp-stall time in naive GEMV

### Why κ_fused (1.2μs) Is So Low
- The fused kernel is a single launch → no per-operator dispatch
- 1.2μs is just the driver submission overhead for one grid launch (kernel launch latency on modern NVIDIA GPUs is ~1-5μs depending on driver state)
- H2D copy (0.0012ms for 256 floats) is negligible
- Compare to unfused: 0.889ms for 48 ops = ~18.5μs per op launch, showing launch overhead compounds

---

## Cross-File Consistency Checks

### Sigma Values
| Source | Value | Agreement |
|--------|-------|-----------|
| `gpu_results_device_sigma.json` | 0.09490 μs | Primary |
| `device_sigma_kernel.json` | 0.09470 μs | 0.2% diff |
| `gpu_results_v2.json` | 220 μs | CUDA-graph proxy, 2300× larger |

### Compute Values
| Source | clock64 compute | CUDA event compute | Ratio |
|--------|----------------|-------------------|-------|
| `gpu_results_device_sigma.json` | 1.0935 ms | 0.7418 ms | 1.47× |
| `device_sigma_kernel.json` | 1.1489 ms | — | — |

The ~15% difference in clock64 compute between runs is normal — depends on memory state, warp scheduling, and input data-dependent control flow divergence in softmax.

### Model Size
All files agree: 2 layers, dim=256, 4 heads, FFN dim=1024. The paper's figures use this config exclusively.

---

## Build System

`persistent_kernel.exe` was compiled with:
```
nvcc -O3 -arch=sm_89 persistent_kernel.cu -o persistent_kernel.exe
```
- sm_89 = Ada Lovelace (RTX 40xx). RTX 4050 is AD107, sm_89.
- No `--use_fast_math` (but uses `rsqrtf` and `tanhf` which are fast intrinsics)
- No `-G` debug flag (performance measurement, not debugging)

---

## Known Issues / Limitations (in code, not paper)

1. **Naive GEMV** (line 149): serial dot product per row, no tiling, no vectorized loads. Performance is poor but consistent for measurement.
2. **No shared-memory tiling in attention** (line 257): QK^T is re-read from global memory per position. Fine at HEAD_DIM=64, would not scale.
3. **Single-block kernel** (line 499): `if (blockIdx.x != 0) return;` — only one SM used. A production version would use all SMs with warp specialization.
4. **No yield mechanism** in the persistent kernel: warps spin-poll continuously. The "cooperative yield" described in paper §4 (3) is **not implemented** — only the architecture diagram and Algorithm 1 reference it.
5. **Kappa derivation** includes H2D copy: `naive_ms_per_step` includes `cudaMemcpy(d_input, ...)` on every step, which is realistic (new token embeddings arrive each step) but inflates κ slightly vs kernel-only measures.
6. **Device sigma kernel JSON** has kappa=0 due to negative computation: the naive_total exceeded compute+expected kappa, suggesting a measurement artifact (possibly from CUDA event asynchrony).
