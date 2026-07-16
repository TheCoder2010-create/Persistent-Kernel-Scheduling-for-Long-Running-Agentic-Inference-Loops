# Persistent Kernel Scheduling for Long-Running Agentic Inference Loops

**Author:** Manav Sutar · Single Core Labs · Pune, India  
**arXiv:** _[pending]_

## What This Paper Does

Standard LLM inference restacks CUDA kernels at every model call. In agent
loops (model → tool → model, repeated tens to hundreds of times), that
overhead is paid on every iteration. This paper:

1. **Formalizes** a cost model separating launch overhead ($\kappa$) from
   tool latency ($\ell_t$) and queue-signal cost ($\sigma$).
2. **Implements** a persistent CUDA kernel (`persistent_kernel.cu`) with
   device-side `clock64()` instrumentation that measures the true queue-signal
   cost without relying on CUDA-graph replay proxies.
3. **Decomposes** the saving into two parts:
   - **Fusion** (already established by MPK, Kog, Ada-MK): eliminates 99.9%
     of the unfused launch overhead.
   - **Residency** (this paper's mechanism): eliminates the remaining
     1.11 μs/step — real but small at this model scale.

## Key Numbers (RTX 4050 Laptop GPU)

| Metric | Value | Source |
|--------|-------|--------|
| $\sigma$ (device-side queue signal) | **0.095 μs** (152 cycles) | `clock64()` in `persistent_kernel.cu` |
| $\kappa$ (fused kernel, 1 launch) | 1.208 μs | CUDA-event timing, same forward pass |
| $\kappa$ (unfused 48-kernel baseline) | 0.889 ms | `benchmark.py --mode gpu` |
| Residency benefit beyond fusion ($\kappa_{\text{fused}} - \sigma$) | 1.11 μs/step | direct measurement |
| Recovery at T=100, regime B (vs unfused) | 12.1–16.1% | cost model with measured $\kappa$, $\sigma$ |

**Honest decomposition:** 99.9% of the headline "recovery" compares a fused
kernel against an unfused baseline (prior art). The paper's specific marginal
contribution — keeping an already-fused kernel resident rather than relaunching
it — is 1.11 μs/step, under 0.1% of a full decode+tool-call step at this
model size.

See `paper.pdf` for the full treatment, including an untested hypothesis
applying the same cost model to CPU-offloaded MoE decoding (§6.2).

## Repository Layout

```
.
├── paper.tex                          # Main paper (NeurIPS-style, self-contained)
├── paper.pdf                          # Compiled paper
├── persistent_kernel.cu               # CUDA resident kernel with clock64 timing
├── triton_persistent_agent_kernel.py  # Triton / CUDA-graph prototype
├── benchmark.py                       # Benchmark harness (simulate + GPU modes)
├── AUDIT.md                           # Repository audit + measurement verification
├── sim_results.json                   # Simulated cost-model results (Table 1)
├── gpu_results_device_sigma.json      # clock64 device-side σ (primary result)
├── benchmarks/
│   └── results/
│       ├── gpu_results.json           # Single-op measurement (superseded)
│       └── gpu_results_v2.json        # 48-op chain benchmark (Table 1 proxy)
└── requirements.txt
```

## Build & Run

### CUDA resident kernel

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
nvcc -O3 -arch=sm_89 persistent_kernel.cu -o persistent_kernel.exe
persistent_kernel.exe --steps 20 --json-out gpu_results_device_sigma.json
```

Adjust `-arch` for your GPU: `sm_80` (A100), `sm_86` (RTX 30xx), `sm_89`
(RTX 40xx Ada), `sm_90` (H100).

### Benchmark harness

```bash
# CPU-only cost-model validation
python benchmark.py --mode simulate --regime both

# GPU measurement (48-op chain)
python benchmark.py --mode gpu --regime both --json-out benchmarks/results/gpu_results.json
```

### Paper

```bash
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
```

## Dependencies

- CUDA Toolkit ≥ 12.x (for `persistent_kernel.cu`)
- MSVC Build Tools 2022 (Windows) or GCC (Linux)
- Python 3.10+ with: torch, numpy, matplotlib
- LaTeX distribution with: amsmath, amssymb, geometry, times, booktabs,
  algorithm, algpseudocode, url, caption, natbib, tikz, pgfplots

## Hardware Used

| Component | Details |
|-----------|---------|
| GPU | NVIDIA GeForce RTX 4050 Laptop (sm_89, 6 GB) |
| CUDA Toolkit | 13.3 (V13.3.73) |
| Driver | 610.47 |
| PyTorch | 2.11.0+cu128 |
| OS | Windows 11 |

## Citation

```bibtex
@misc{sutar2026persistent,
  author = {Manav Sutar},
  title  = {Persistent Kernel Scheduling for Long-Running Agentic Inference Loops},
  year   = {2026},
  note   = {arXiv preprint}
}
```
