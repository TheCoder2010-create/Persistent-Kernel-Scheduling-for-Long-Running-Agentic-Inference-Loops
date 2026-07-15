# Persistent Kernel Scheduling for Long-Running Agentic Inference Loops

**Author:** Manav Sutar · Single Core Labs · Pune, India

## Problem

Standard LLM inference stacks re-launch a full stack of CUDA kernels on
every model call. In single-turn decoding that overhead is paid once.
In *agentic loops* — where the model repeatedly calls tools and processes
their results for tens to hundreds of steps — it is paid on every
iteration, even though the transformer weights and most on-GPU state
never change between steps.

Existing *megakernel* / *persistent kernel* work (MPK, AMD monokernel,
Ada-MK) fuses a forward pass into one resident kernel to eliminate
per-operator launch cost, but targets *single-turn* decoding. They do
not address the distinct overhead structure of agent loops, where:

1. The GPU sits fully idle across externally-latent tool calls.
2. The next step's control flow depends on tool *output*, not token output.

## Approach

We formalize the cost model for agent-loop inference:

```
T_naive(T)      = Σ d_t  +  T·κ  +  Σ ℓ_t
T_persistent(T) = Σ d_t  +  T·σ  +  Σ ℓ_t
Δ(T)            = T · (κ − σ)
```

where `κ` is per-step kernel-launch overhead, `σ` is the (much smaller)
cost of signalling a resident queue slot, `d_t` is decode time, and
`ℓ_t` is tool-call latency.

The key design: launch the kernel **once** at session start. Each agent
step becomes a queue push/pop rather than a `cudaLaunchKernel` call. The
kernel polls a device-visible `QueueEntry` and executes `decode_segment`
whenever a `SLOT_READY` entry appears, without ever being torn down
between steps.

See `paper.tex` (Section 3–4 and Algorithm 1) for the full cost model
and design.

## Repository Layout

```
.
├── paper.tex                          # Main paper (NeurIPS-style)
├── paper.pdf                          # Compiled paper
├── persistent_kernel.cu               # Reference CUDA implementation
├── triton_persistent_agent_kernel.py  # Triton / CUDA-graph prototype
├── benchmark.py                       # Benchmark harness (simulate + GPU modes)
├── sim_results.json                   # Canonical simulated results
├── benchmarks/
│   └── results/
│       └── gpu_results.json           # Real hardware results (RTX 4050)
└── requirements.txt                   # Python dependencies (pinned)
```

## Components

### `persistent_kernel.cu` — CUDA reference kernel

A single resident kernel launched once per agent session. The host
pushes work via `cudaMemcpyToSymbolAsync` on a separate non-blocking
stream; the kernel polls `g_queue.state` and executes `decode_segment`
when it sees `SLOT_READY`, then flips to `SLOT_DONE`.

**Build** (requires MSVC + CUDA Toolkit, tested with CUDA 13.3 / sm_89):

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
nvcc -O3 -arch=sm_89 persistent_kernel.cu -o persistent_kernel.exe
```

Adjust `-arch` for your GPU: `sm_80` for A100, `sm_86` for RTX 30xx,
`sm_89` for RTX 40xx Ada Lovelace, `sm_90` for H100.

**Run:**

```
persistent_kernel.exe
```

Expected output: 20 lines of `step N: decode segment done in X ms (no
kernel relaunch)` followed by `Resident kernel exited after 20 agent
steps, single launch.`

---

### `triton_persistent_agent_kernel.py` — Triton / CUDA-graph prototype

Higher-level Python prototype of the same idea. Because Triton on
Windows has no standalone wheel, this uses **CUDA graph replay** as the
practical equivalent of kernel residency: the decode segment is captured
once and replayed each step, eliminating Python/driver dispatch overhead.

Compares `run_agent_loop_naive` (fresh kernel dispatch every step)
against `run_agent_loop_with_cuda_graph` (single capture, repeated
replay).

**Requires:** PyTorch with CUDA; Triton bundled inside PyTorch on
Windows (no separate `pip install triton` needed).

```
python triton_persistent_agent_kernel.py --steps 50 --tool-latency-ms 20
```

---

### `benchmark.py` — Cost-model benchmark harness

Two modes:

| Mode | Requires GPU? | What it measures |
|------|--------------|-----------------|
| `simulate` | No | Models κ, σ, d_t, ℓ_t as random variables; validates cost model |
| `gpu` | Yes | Measures actual κ via CUDA events; real Δ(T)/T_naive(T) |

Two regimes:

| Regime | Label | d_t | ℓ_t |
|--------|-------|-----|-----|
| A | tool-latency-dominated | 5–15 ms | 50–200 ms |
| B | launch-overhead-dominated | 2–5 ms | 0.5–2 ms |

**Run simulate (any machine, validates against `sim_results.json`):**

```
python benchmark.py --mode simulate --regime both
```

**Run GPU benchmark (saves results):**

```
python benchmark.py --mode gpu --regime both --json-out benchmarks/results/gpu_results.json
```

---

### `paper.tex` — Main paper

NeurIPS-style, self-contained (no external `.sty` files). Requires a
TeX distribution with standard packages: `amsmath`, `amssymb`,
`geometry`, `times`, `booktabs`, `algorithm`, `algpseudocode`,
`hyperref`, `xcolor`, `caption`, `natbib`.

**Compile (MiKTeX or TeX Live):**

```
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex   # second pass for cross-references
```

## Setup

**Python environment (Python 3.13, CUDA 12.8 wheels):**

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy matplotlib
```

Or install all pinned deps:

```
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128
```

## Hardware Used

| Component | Details |
|-----------|---------|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU (sm_89, Ada Lovelace) |
| VRAM | 6 GB |
| CUDA Toolkit | 13.3 (nvcc V13.3.73) |
| Driver | 610.47 |
| PyTorch | 2.11.0+cu128 |
| OS | Windows 11 |

## Key Results (GPU, RTX 4050)

Measured kernel-launch overhead `κ ≈ 0.019 ms` and CUDA-graph replay
cost `σ ≈ 0.016 ms` — roughly **79× lower** than the 1.5 ms canonical
figure used in the simulated model. This shifts the practical benefit
toward regime B (low tool latency, launch-overhead-dominated), where
persistent scheduling recovers up to ~14.8% of total loop time at
T = 100 steps.

Full numbers: `benchmarks/results/gpu_results.json`
