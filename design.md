# Multi-Trajectory Persistent Kernel — System Design

## 1. Why Multi-Trajectory?

The original paper's residency saving (κ_fused − σ = **5.725 μs/step**) is
too small to matter when d_t + ℓ_t ≫ 1 ms.

**Key insight:** If one persistent kernel handles N trajectories, the
launch cost κ_fused is paid **once** instead of N times per step.

### Cost Model (Original)

```
Single trajectory, per-step:
  T_naive      = d_t + κ_fused + ℓ_t
  T_persistent = d_t + σ      + ℓ_t

Saving per step = κ_fused − σ ≈ 5.7 μs
```

### Cost Model (Multi-Trajectory)

```
N trajectories, one resident kernel, per-step:

  T_persistent(N) = (d_t) + (κ_fused / N + σ) + ℓ_t

Because:
  - Launch cost κ_fused is paid once for all N trajectories
  - Queue-poll cost σ is paid per trajectory per step
  - Decode cost d_t per trajectory is unchanged
  - Tool latency ℓ_t per trajectory is unchanged

Saving per step vs N×naive-launch:
  Δ(N) = N × κ_fused − (κ_fused + N × σ)
       = (N − 1) × κ_fused − N × σ
       ≈ (N − 1) × 5.82 − N × 0.095  μs

For N ≫ 1:
  Δ(N) ≈ (N − 1) × 5.82 μs    (σ becomes negligible)
```

### Why This Matters

| N | Δ per step | vs 5 ms step | vs 1 ms step |
|---|-----------|-------------|-------------|
| 1 | 5.7 μs | 0.11% | 0.57% |
| 4 | 17.1 μs | 0.34% | 1.71% |
| 16 | 85.7 μs | 1.71% | **8.57%** |
| 64 | 370 μs | **7.4%** | **37%** |
| 128 | 742 μs | **14.8%** | **74%** |

At N ≥ 16, the saving exceeds 1% even with d_t + ℓ_t = 5 ms. At N ≥ 64
and d_t + ℓ_t ≤ 1 ms, the saving is transformative.

Fusion alone (prior art) cannot give this — each trajectory still needs
its own fused-kernel launch. Residency across tool-call boundaries is
the only way to amortize κ_fused over N trajectories.

---

## 2. Queue Architecture

### Original (Single-Trajectory)

```
Host                              GPU (Kernel)
┌─────────────────┐              ┌──────────────────────┐
│ QueueEntry[1]   │   write      │ Poll slot[0]         │
│   .state=READY  │ ──────────>  │   if READY: decode    │
│   .input[...]   │   (mapped    │   set state=DONE     │
│   .output[...]  │    memory)   │   poll again         │
└─────────────────┘              └──────────────────────┘
```

### New (Multi-Trajectory)

```
Host                              GPU (Kernel)
┌─────────────────┐              ┌──────────────────────────────┐
│ QueueEntry[N]    │   write      │ Round-robin over N slots:   │
│ [0] traj=0, READY│ ──────────> │   for i in 0..N-1:           │
│ [1] traj=1, READY│             │     if slot[i].state==READY: │
│ [2] traj=2, EMPTY│             │       decode(traj=i)         │
│ ...              │             │       slot[i].state=DONE     │
│ [N-1] traj=M, ...│             │     poll + yield             │
└─────────────────┘              └──────────────────────────────┘
```

### Queue Entry (Extended)

```c
struct QueueEntry {
    int32_t state;           // EMPTY=0, READY=1, DONE=2, SHUTDOWN=3
    int32_t trajectory_id;   // which trajectory this belongs to
    int32_t seq_len;         // KV cache length before this step
    float   input[DIM];
    float   output[DIM];
};
```

### Queue States

```
EMPTY ──(host submits)──> READY ──(kernel decodes)──> DONE ──(host reads)──> EMPTY
                              SHUTDOWN ──(kernel exits)──> [terminated]
```

### KV Cache Isolation

Each trajectory has its own KV cache in device memory:

```c
// Layout: [trajectory][layer][head][pos][dim]
// Trajectory t's KV cache starts at:
float* k_cache_t = k_cache + t * KV_CACHE_PER_TRAJ;
```

Trajectories do NOT share KV cache. The kernel selects the correct
cache via `trajectory_id` in the queue entry.

---

## 3. Kernel Design

### Poll Loop (Pseudo-code)

```
__global__ void persistent_multi_agent_kernel(
    float* weights,          // shared weights (all trajectories)
    float* k_cache,          // [N_TRAJ * N_LAYERS * KV_CACHE_PER_LAYER]
    float* v_cache,          // [N_TRAJ * N_LAYERS * KV_CACHE_PER_LAYER]
    int*   kv_len,           // [N_TRAJ * N_LAYERS]  per-trajectory lengths
    QueueEntry* queue,       // [N_TRAJ] queue slots
    int N,                   // actual trajectory count
    KernelStats* stats       // per-trajectory stats [N]
) {
    if (blockIdx.x != 0) return;

    while (true) {
        bool any_work = false;

        for (int i = 0; i < N; i++) {
            volatile int s = queue[i].state;

            if (s == SLOT_SHUTDOWN) {
                // Check ALL slots for SHUTDOWN before exiting
                bool all_shutdown = true;
                for (int j = 0; j < N; j++)
                    if (queue[j].state != SLOT_SHUTDOWN) all_shutdown = false;
                if (all_shutdown) return;
                continue;  // keep polling other slots
            }

            if (s == SLOT_READY) {
                any_work = true;
                unsigned long long t_ready = clock64();

                // Barrier: all warps sync (end of sigma window)
                __syncthreads();
                unsigned long long t_compute_start = clock64();
                unsigned long long sigma_cycles = t_compute_start - t_ready;

                int tid = queue[i].trajectory_id;

                decode_segment(
                    queue[i].input,
                    queue[i].output,
                    weights,
                    k_cache + tid * KV_CACHE_PER_TRAJ,
                    v_cache + tid * KV_CACHE_PER_TRAJ,
                    kv_len + tid * N_LAYERS,
                    &shared_workspace
                );

                unsigned long long t_compute_end = clock64();

                // Record stats for this trajectory
                if (stats) {
                    stats[tid].sigma_cycles_total   += sigma_cycles;
                    stats[tid].compute_cycles_total += (t_compute_end - t_compute_start);
                    stats[tid].n_timed_steps        += 1;
                }

                queue[i].state = SLOT_DONE;
                __threadfence();
            }
        }

        // Yield briefly if no work (avoids tight spin with zero throughput)
        if (!any_work) {
            __nanosleep(100);  // ~100 ns pause
        }
    }
}
```

### Why Round-Robin?

- **Fairness**: all trajectories make progress regardless of submission order
- **No starvation**: each poll cycle visits every slot
- **Simple**: O(N) per cycle, trivially correct
- **Adequate**: at N ≤ 128, O(N) is negligible vs decode time

### Why __nanosleep?

Without yield, an idle kernel spins at maximum SM clock consuming power
and memory bandwidth. `__nanosleep(100)` releases the SM to other
workloads while keeping the kernel resident. This is the **cooperative
yield** mechanism described in the paper.

---

## 4. Host-Side Submission Pattern

```
for each step t in 0..T-1:
    for each trajectory i in 0..N-1:
        if trajectory i has work:
            queue[i].input = encode(tool_result)
            queue[i].state = READY

    // Wait for ALL N trajectories to complete
    for each trajectory i in 0..N-1:
        spin until queue[i].state == DONE

    // Read outputs
    for each trajectory i in 0..N-1:
        output = queue[i].output

    // Simulate tool latency
    sleep(tool_latency_ms)
```

### Why Wait for All N?

In a real agent system, trajectories proceed independently. The benchmark
synchronizes at each step to make the cost model comparison fair against
N×naive launches. In production, the persistent kernel would process
whatever is ready.

---

## 5. Benchmark Regime (Target)

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-0.5B (INT4) or synthetic DIM=256 transformer |
| N sweep | 1, 2, 4, 8, 16, 32, 64, 128 |
| d_t | 100 μs–5 ms (model-dependent) |
| ℓ_t | 0.1–1 ms (local Python tools) |
| σ | 0.095 μs (device-side queue poll, unchanged) |
| κ_fused | 5.82 μs (single fused launch, from original paper) |

### Measurement Protocol

```
For each N:
  1. Warmup: 10 steps (cold caches, JIT compilation)
  2. Timed: 200 steps with clock64() instrumentation
  3. Record: σ_cycles, compute_cycles per trajectory
  4. Naive baseline: N × κ_fused per step (computed, not measured)

Output: Δ(N) = N × κ_fused − (κ_fused + N × σ)  
        Fraction = Δ(N) / (d_t + N × κ_fused + ℓ_t)
```

---

## 6. Implementation Plan

### Files to Modify

| File | Change |
|------|--------|
| `persistent_kernel.cu` | Add N-trajectory queue, round-robin poll, per-trajectory KV cache |
| `triton_persistent_agent_kernel.py` | Multi-trajectory version with host-side queue simulation |
| `benchmark.py` | Multi-trajectory regimes, N-sweep, small-model integration |
| New: `design.md` | This document |

### Build & Test

```bash
# Compile multi-trajectory CUDA kernel
nvcc -O3 -arch=sm_86 persistent_kernel.cu -o persistent_kernel.exe

# Run benchmark sweep
python benchmark.py --mode multi-trajectory --sweep-n 1,2,4,8,16,32,64,128

# Triton prototype (faster iteration)
python triton_persistent_agent_kernel.py --n-trajectories 8 --steps 50
```

---

## 7. Formulas Reference

| Symbol | Meaning | Original Value |
|--------|---------|----------------|
| κ_fused | Fused kernel launch + teardown overhead | 5.82 ± 0.27 μs |
| σ | Device-side queue polling signal | 0.095 μs (152 cycles) |
| d_t | Decode segment compute time | Model-dependent |
| ℓ_t | Tool-call latency | Benchmark parameter |
| N | Number of concurrent trajectories | Sweep parameter |
| T | Number of agent-loop steps | Sweep parameter |

### T_persistent(N) derivation

```
Per step, N trajectories:
  Host submits N queue entries
  Kernel polls N slots:           N × σ (sequential)
  Kernel decodes N segments:      N × d_t (sequential per trajectory)
  Kernel launch:                  κ_fused (paid once)

  Total GPU time = κ_fused + N × (σ + d_t)

Per trajectory per step:
  T_persistent(N) = κ_fused/N + σ + d_t

Compare to N × naive:
  T_naive(N)   = N × (κ_fused + d_t)

Saving:
  Δ = T_naive - T_persistent
    = N × κ_fused + N × d_t - κ_fused - N × σ - N × d_t
    = (N − 1) × κ_fused - N × σ
    = N × (κ_fused − σ) − κ_fused

For N → ∞:
  Δ ≈ N × κ_fused
  (each additional trajectory saves one full launch)
```

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Round-robin over N slots | Fair, simple, no starvation |
| Per-trajectory KV cache | No cross-trajectory interference |
| Shared weights | All trajectories use same model |
| __nanosleep on idle | Cooperative yield without teardown |
| clock64() per trajectory | Isolate timing per trajectory |
| Host submits all N before kernel poll | Simplifies benchmark comparison |
