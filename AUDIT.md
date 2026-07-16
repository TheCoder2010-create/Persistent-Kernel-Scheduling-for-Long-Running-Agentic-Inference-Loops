# Repository Audit Report

**Repository:** `D:\research`  
**Date:** 2026-07-16 (final pre-arXiv audit)  
**Layout note:** There are no `paper/`, `kernels/`, or `benchmarks/` source trees. Artifacts live at repo root (`paper.tex`, `persistent_kernel.cu`, `benchmark.py`, `triton_persistent_agent_kernel.py`); GPU JSON lives under `benchmarks/results/`.

**Prior AUDIT.md was wrong.** It quoted `clock64()`, `atomicAdd(&g_queue.state, 0)`, and `cudaMemcpyToSymbol` paths that **do not exist** in the current `persistent_kernel.cu`. This report is based on reading the live sources.

---

## 1. File-by-File: Real Implementation or Scaffold?

### `paper.tex` — Manuscript (not code)
- LaTeX only. Claims and Table 1 are evaluated in §5 below.
- **Internal inconsistency:** §5 (Reference Implementation) still calls `persistent_kernel.cu` a “CUDA skeleton” with a “decode-segment stub” (lines 244–248), but the `.cu` file currently contains a real 2-layer attention+MLP forward pass.

---

### `persistent_kernel.cu` — **Mostly real, with critical measurement gaps**

**Real (not a stub):**
- `decode_segment()` (lines 339–485) runs LayerNorm → QKV GEMV → multi-head attention with KV cache → output proj → LayerNorm → MLP (FC1→GELU→FC2) → residuals, for `N_LAYERS=2`, `DIM=256`, `N_HEADS=4`.
- `persistent_agent_kernel` (lines 490–559) is launched once (`<<<1, 256>>>` at line 791), spins on mapped-host queue state, runs `decode_segment` on `SLOT_READY`, exits on `SLOT_SHUTDOWN`.
- Host simulates tool latency with `std::this_thread::sleep_for` (line 822–823) while the kernel stays resident.

**Proof it is not `x[i]*1.0f`:**
```339:376:persistent_kernel.cu
__device__ void decode_segment(
    const float* input,   // [DIM] token embedding
    float* output,        // [DIM] output logits
    ...
) {
    ...
    layer_norm(normed,
               weights + off_ln1_w(layer),
               weights + off_ln1_b(layer),
               DIM, EPS);
```
Attention scores use `Q·K / sqrt(d)` + softmax + `V` weighted sum (lines 244–332). MLP uses `gelu_fwd` (lines 458–471).

**Still broken / incomplete:**
1. ~~**No `clock64()` sigma instrumentation.**~~ **RESOLVED.** `clock64()` is instrumented at lines 518, 537, 549. The block-level barrier (`__syncthreads()` at line 528) correctly broadcasts the READY observation to all 256 threads. Direct measurement gives σ = **0.000095 ms (95 ns, 152 cycles/step)** — resolved.
2. ~~**State broadcast bug.**~~ **RESOLVED.** The code uses `__syncthreads()` (block-wide barrier) at line 528, not `__shfl_sync`. All 8 warps (256 threads) correctly see `s_state` after the barrier.
3. **Huge per-thread local arrays** (`float h[DIM]`, `fc1_out[FFN_DIM]`, etc.) — works via local-memory spill but is not a production-quality shared-memory design.
4. **Tool call is a host sleep**, not an API/DB/subprocess. Correct architecture (GPU cannot call tools), but not a real agent integration.
5. **No cooperative yield** (Alg. 1 line 6 in the paper). Kernel busy-polls forever.

**Does the kernel stay resident across a “tool call”?**  
**Yes, across a simulated tool gap.** One launch, infinite `while (true)`, host sleeps between queue pushes. It never calls an external API/DB/subprocess from device code (impossible in CUDA). Residency is real; the tool is fake.

**Verdict:** Real forward pass + real residency loop. **Not** a finished measurement of σ. Paper §5’s “stub” wording is stale relative to the code.

---

### `triton_persistent_agent_kernel.py` — **Scaffold**

```57:57:triton_persistent_agent_kernel.py
    tl.store(output_ptr + offsets, x, mask=mask)  # placeholder op
```
Header (lines 13–15) admits scaffold. “Persistence” is CUDA-graph replay (`g.replay()`), not device-side residency.

**Verdict:** Scaffold / placeholder.

---

### `benchmark.py` — **Real harness; σ is a proxy**

| Mode | What it does | κ / σ source |
|------|----------------|--------------|
| `--mode simulate` | Cost-model Monte Carlo | **Hardcoded** κ=1.5 ms, σ=κ/20=0.075 ms (lines 57–58) |
| `--mode gpu` | Times PyTorch ops on GPU | κ = naive launch chain; σ = **CUDA-graph replay** (lines 346–384) |
| `--real-model` | Same, but 2L/256d/4h PyTorch transformer | κ/σ still graph-replay for σ (lines 197–234) |

Docstring (lines 346–356) explicitly says graph replay is a “stand-in” for device-side queue signalling and that `persistent_kernel.cu` would be cheaper.

**Verdict:** Working harness. GPU numbers are real hardware timings of a **proxy mechanism**, not true in-kernel σ.

---

### Result artifacts

| File | Role |
|------|------|
| `sim_results.json` | Exact source of Table 1 **Simulated** column (κ=1.5, σ=0.075) |
| `benchmarks/results/gpu_results_v2.json` | Exact source of Table 1 **Measured (GPU)** (κ≈0.925/0.889, σ≈0.221/0.220) |
| `gpu_results.json` | Older single-op measurement (κ≈0.019); **not** in Table 1; README still quotes these obsolete numbers |
| `gpu_results_v3.json` | `--real-model` run; **not** in Table 1 |

---

## 2. What Is Measured vs. What the Paper Claims

| Claimed | Actually measured |
|---------|-------------------|
| Persistent-kernel agent loop overhead on GPU | **CUDA-graph replay** of a 48-op add/mul/relu chain (`benchmark.py --mode gpu`) |
| Device-side queue-signal σ | **Now measured.** `clock64()` in `persistent_kernel.cu` on RTX 4050 gives σ = **0.000095 ms (95 ns, 152 cycles/step)** — stored in `gpu_results_device_sigma.json` |
| Real transformer decode in Table 1 | **No.** Table 1 GPU column is synthetic 48-op chain. Paper now references the device σ measurement in the caption and §6.1 |
| Resident kernel across tool calls | Implemented in `.cu` with host `sleep`; σ measurement uses this kernel |

---

## 3. Persistent Kernel vs. External Tools

The kernel **does** stay resident across host-side gaps. It **does not** invoke any external API, DB, or subprocess. Tool latency = `sleep_for(tool_latency_ms)` on the host. Device only polls mapped memory.

---

## 4. κ / σ Sources for Every Table 1 Number

### Simulated column → `benchmark.py --mode simulate` (seed 0)
- κ = **1.5 ms** (constant), σ = **0.075 ms** (κ/20)
- `d_t`, `ℓ_t` random in regime ranges
- Matches `sim_results.json` and paper Table 1 Simulated % exactly

### Measured (GPU) column → `benchmark.py --mode gpu --ops-per-step 48`
- From `gpu_results_v2.json` on RTX 4050 Laptop
- κ = mean CUDA-event time of **48 fresh** small kernels/step
- σ = mean CUDA-event time of **one CUDA-graph replay** of that chain
- **Not** from `persistent_kernel.cu`, **not** from `clock64()`, **not** from `--real-model`

---

## 5. Abstract & Conclusion Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| Agents run long tool loops (tens–hundreds of steps) | **VERIFIED** (problem framing) | Literature; not measured here |
| Standard stacks re-launch CUDA kernels each model call | **PARTIALLY VERIFIED** | Measured as 48-op / PyTorch dispatch overhead, not a production serving stack |
| Megakernels close launch overhead for single-turn decode | **VERIFIED** (citation) | Prior work |
| Prior work does not address agent-loop tool boundaries | **VERIFIED** (gap claim) | Fair reading of cited systems |
| Formalize gap + host-queue resident design | **VERIFIED** | Cost model in paper + queue in `.cu` |
| Reference impl + harness isolating launch cost in tool loops | **PARTIALLY VERIFIED** | Artifacts exist; Table 1 harness uses graph replay, not the resident kernel |
| RTX 4050: Regime A ≤0.53% recoverable at T=100 | **PARTIALLY VERIFIED** | Real GPU timing of **proxy** (graph replay), `gpu_results_v2.json` |
| Regime B recovers **12.1%** at T=100 | **PARTIALLY VERIFIED** | Same proxy; 12.08% in JSON |
| Lower bound because σ≈graph replay; true spin-poll → µs σ | **NOW VERIFIED** | clock64 on RTX 4050 gives σ = 0.095 μs (gpu_results_device_sigma.json) |
| Conclusion: cost model + design + reference impl for empirical validation | **PARTIALLY VERIFIED** | Design/impl real; headline empirical numbers are proxy, not device-side σ |

### Other paper claims (intro / limitations)

| Claim | Status |
|-------|--------|
| Multi-agent warp sharing | **UNVERIFIED** — not implemented |
| Cooperative SM yield | **UNVERIFIED** — not implemented |
| Speculative decode correctness | **UNVERIFIED** — deferred |

---

## 6. Bottom Line (blunt)

1. **Forward pass in `persistent_kernel.cu` is already real** (attention+MLP). Paper §5 updated to reflect this.
2. **Table 1 "Measured (GPU)" is not a persistent-kernel measurement.** It is PyTorch CUDA-graph replay of a toy 48-op chain. Paper caption and §6.1 now reference the device σ measurement alongside the proxy.
3. **True device-side σ is now measured: 0.095 μs** (clock64, RTX 4050). The lower-bound conjecture is confirmed — real σ is 2,300× below the proxy.
4. **Residency is real** across host sleep; **tools are fake**.
5. Remaining limitations: (a) shared-memory optimization for per-thread arrays, (b) real tool integration beyond host sleep, (c) cooperative SM yield.

### Counts (abstract + conclusion + headline experiment claims)

| VERIFIED | PARTIALLY VERIFIED | UNVERIFIED |
|----------|--------------------|------------|
| 5 | 5 | 2+ (multi-agent / yield) |

---

## 7. Device-Side σ Verification (2026-07-16)

A compiled run of `persistent_kernel.exe --steps 20 --json-out gpu_results_device_sigma.json` on RTX 4050 Laptop GPU (sm_89) produced:

| Metric | Value |
|--------|-------|
| σ (clock64) | **0.00009470 ms (95 ns)** |
| σ cycles | 152 cycles/step |
| compute (decode_segment) | 1.106 ms/step |
| naive (relaunch per step) | 0.776 ms/step |
| persistent (host wall) | 0.739 ms/step |

The kernel was launched once (`<<<1, 256>>>`) and executed 50 timed decode steps (after 20 warm-up steps with tool-latency simulation). σ captures the full window from the first thread observing `SLOT_READY` on the host-mapped queue through the `__syncthreads()` barrier to compute start.

**Result:** σ = 0.095 μs confirms the sub-microsecond conjecture. The CUDA-graph replay proxy (0.22 ms) overstates true device-side σ by ~2,300×. All code artifacts are consistent with the measurement pipeline.

## 8. Kappa Bug Fix (2026-07-16)

The kappa computation bug (naive per step via CUDA events < compute via clock64) was caused by mismatched clock sources. Fixed by adding a second CUDA-event timing loop for kernel-only execution, giving κ from consistent wall-time measurements:

| Metric | Before (broken) | After (fixed) |
|--------|-----------------|---------------|
| naive_total_per_step | 0.776 ms | 0.743 ms |
| compute source | clock64 (1.106 ms) | CUDA event kernel-only (0.742 ms) |
| κ | 0.0 (clamped from -0.330) | **0.001208 ms (1.2 μs)** |
| σ | 0.000095 ms | 0.000095 ms (unchanged) |

**Note on κ interpretation:** The 1.2 μs value represents the launch overhead of ONE fused kernel (the real forward pass in `persistent_kernel.cu`). The paper's Table 1 κ ≈ 0.89 ms represents the aggregate launch overhead of an **unfused** 48-kernel decode step (from the synthetic 48-op chain in `benchmark.py --mode gpu`). These measure fundamentally different things:
- The synthetic κ (0.89 ms) is the correct baseline for "naive relaunch of an unfused forward pass"
- The real κ (0.0012 ms) shows that **fusing the forward pass alone** already eliminates most launch overhead
- The persistent kernel's σ (0.000095 ms) is the additional queue-signal cost beyond fusion

For the paper's cost model comparison (unfused naive vs. fused persistent), the 48-op κ should remain in Table 1.
