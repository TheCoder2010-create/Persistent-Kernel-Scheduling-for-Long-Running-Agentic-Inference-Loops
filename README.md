<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="agent-loop-visual.svg">
    <img src="agent-loop-visual.svg" width="100%" alt="Persistent Kernel Scheduling for Agent Loops">
  </picture>
</p>

<p align="center">
  <a href="paper.pdf"><img src="https://img.shields.io/badge/paper-PDF-ef4444?style=flat-square" alt="Paper PDF"></a>
  <a href="paper.tex"><img src="https://img.shields.io/badge/source-LaTeX-3b82f6?style=flat-square" alt="LaTeX source"></a>
  <a href="persistent_kernel.cu"><img src="https://img.shields.io/badge/CUDA-persistent__kernel.cu-10b981?style=flat-square" alt="CUDA kernel"></a>
  <a href="benchmark.py"><img src="https://img.shields.io/badge/benchmark-Python-f59e0b?style=flat-square" alt="Benchmark harness"></a>
  <img src="https://img.shields.io/badge/license-MIT-8b5cf6?style=flat-square" alt="License">
</p>

---

## The Problem

**Autonomous LLM agents run long iterative loops:**

```
model call → tool call → model call → tool call → model call → ...
```

Each model call re-launches a full stack of CUDA kernels — attention, GEMMs,
softmax, layer-norm, residual adds — even though the transformer weights and
most on-GPU state persist across the entire loop. In a single 100-step agent
trajectory, this means paying **kernel tear-down and re-launch overhead on
every single step** while the GPU sits idle across externally-latent tool
calls (APIs, databases, subprocesses).

Prior megakernel work (MPK, Kog, Ada-MK, Event Tensor) solves the
per-operator launch problem for single-turn decode by fusing the entire
forward pass into one resident kernel. **But no existing system addresses the
distinct overhead structure at the tool-call boundary** — where the GPU must
survive externally-latent, data-dependent gaps between decode segments.

---

## What This Work Does

### 1. A Cost Model for Agent Loops

We formalize the agent-loop overhead with three parameters:

| Symbol | Meaning |
|--------|---------|
| **κ** | Kernel launch/teardown overhead per step |
| **σ** | Queue-signal cost (persistent design, per step) |
| **ℓ_t** | Tool-call latency (API, DB, subprocess — opaque to GPU) |

Total wall time under standard per-step launching:
```
T_naive      = Σ (d_t + κ + ℓ_t)
```
Under persistent scheduling with a host-managed work queue:
```
T_persistent = Σ (d_t + σ + ℓ_t)
```
Recoverable overhead: **Δ(T) = T · (κ − σ)**

### 2. A Host-Queue / Resident-Kernel Design

<p align="center">
  <img src="agent-loop-visual.svg" width="90%" alt="Design overview">
</p>

A three-part architecture:
- **Resident decode kernel** — launched once per agent session, not once per step
- **Host-managed work queue** — tool results arrive as queue entries; the
  kernel polls instead of relaunching
- **Cooperative yield** — during idle tool windows, SMs release to the
  scheduler for other tenants

### 3. Reference Implementation

Three artifacts in this repository:

- **`persistent_kernel.cu`** — CUDA skeleton with device-side spin-queue,
  clock64() instrumentation, and a decode-segment stub. Drop in a real
  forward pass (CUTLASS, Mirage-compiled task graph) for production use.

- **`triton_persistent_agent_kernel.py`** — Triton prototype for rapid
  iteration before dropping to raw CUDA.

- **`benchmark.py`** — Hardware-agnostic harness with simulation and GPU modes
  that measures T_naive vs. T_persistent under the cost model.

---

## Key Results

| Metric | Value |
|--------|-------|
| **σ** (device-side queue signal) | **0.095 μs** (152 clock cycles) |
| **κ_fused** (fused kernel, single launch) | **5.82 ± 0.27 μs** |
| **κ_unfused** (48-kernel-per-step baseline) | **0.851 ± 0.069 ms** |
| **Fusion saving** (κ_unfused − κ_fused) | **99.3%** of unfused overhead |
| **Residency saving** (κ_fused − σ) | **5.725 μs/step** |
| **Recovery at T=100** (vs unfused baseline) | **13.6–14.5%** |

### Honest Decomposition

Over **98%** of the headline recovery comes from **kernel fusion** — a benefit
already established by MPK, Kog, and Ada-MK for single-turn decode.

This paper's specific contribution — **keeping an already-fused kernel
resident across tool-call boundaries** — saves the remaining **5.7 μs/step**.
At this model scale, that is 0.011% of a full decode-plus-tool-call step
(where tool latency ≥ 50 ms).

> Persistent-kernel scheduling's marginal value beyond fusion is real but
> small on current hardware at this scale. Its practical significance
> depends on regimes with much smaller decode times or much higher step
> frequencies than tested here.

---

## Repository Layout

```
.
├── paper.tex                          # Main paper (self-contained, NeurIPS-style)
├── paper.pdf                          # Compiled paper
├── agent-loop-visual.svg              # Architecture diagram
├── persistent_kernel.cu               # CUDA resident kernel with clock64 timing
├── triton_persistent_agent_kernel.py  # Triton / CUDA-graph prototype
├── benchmark.py                       # Benchmark harness (simulate + GPU modes)
├── AUDIT.md                           # Measurement verification
├── requirements.txt
└── *.json                             # Raw measurement data (10 runs)
```

---

## Quick Start

### Run the Cost Model (CPU, no GPU needed)

```bash
python benchmark.py --mode simulate --regime both
```

### Build the CUDA Persistent Kernel

```bash
nvcc -O3 -arch=sm_XX persistent_kernel.cu -o persistent_kernel.exe
persistent_kernel.exe --steps 20 --json-out results.json
```

Adjust `-arch=sm_XX` to match your GPU (`sm_80` A100, `sm_86` RTX 30xx,
`sm_89` Ada, `sm_90` H100).

### GPU Benchmark

```bash
python benchmark.py --mode gpu --regime both --ops-per-step 48
```

### Compile the Paper

```bash
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

---

## Dependencies

| Tool | Required for |
|------|-------------|
| CUDA Toolkit ≥ 12.x | `persistent_kernel.cu` |
| MSVC / GCC | CUDA compilation |
| Python 3.10+ (torch, numpy, matplotlib) | `benchmark.py` |
| LaTeX (amsmath, tikz, pgfplots, natbib) | Paper compilation |

---

## Citation

```bibtex
@misc{sutar2026persistent,
  author = {Manav Sutar},
  title  = {Persistent Kernel Scheduling for Long-Running
            Agentic Inference Loops},
  year   = {2026}
}
```
