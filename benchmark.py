"""
benchmark.py

Benchmark harness for the cost model in paper.tex, Section 3:

    T_naive(T)      = sum_t ( d_t + kappa + ell_t )
    T_persistent(T) = sum_t ( d_t + sigma + ell_t )
    Delta(T)        = T * (kappa - sigma)

Two modes:
  --mode simulate   Runs on ANY machine (no GPU required). Models kappa,
                    sigma, d_t, ell_t as configurable random variables and
                    reports Delta(T)/T_naive(T) — validates the cost model
                    and produces the scaling numbers referenced in
                    paper.tex Table 1 ("Simulated" column).

  --mode gpu        Runs ONLY on a CUDA-capable machine. Measures ACTUAL
                    kernel-launch overhead (kappa) via CUDA events by
                    timing repeated small-kernel launches, and reports
                    real Delta(T)/T_naive(T) using measured kappa instead
                    of a modeled one. Fills in paper.tex Table 1's
                    "Measured (GPU)" column.

Flags:
  --ops-per-step N  Number of synthetic kernel launches per step (default 48).
  --real-model      Also run a tiny real transformer block (2 layers, dim=256,
                    4 heads, MHA + 2-layer MLP) as the decode segment and
                    compare its kappa/sigma against the synthetic N_OPS chain.

Usage:
  python3 benchmark.py --mode simulate --regime A
  python3 benchmark.py --mode simulate --regime B
  python3 benchmark.py --mode gpu --regime A              # needs a CUDA GPU
  python3 benchmark.py --mode gpu --regime both --real-model
"""

import argparse
import json
import random
import statistics
import sys
import time


REGIMES = {
    # (d_t range ms, ell_t range ms, kappa ms) — see paper.tex Sec 6 setup.
    "A": dict(d_lo=5.0, d_hi=15.0, ell_lo=50.0, ell_hi=200.0, kappa=1.5, label="tool-latency-dominated"),
    "B": dict(d_lo=2.0, d_hi=5.0, ell_lo=0.5, ell_hi=2.0, kappa=1.5, label="launch-overhead-dominated"),
}

T_VALUES = [1, 10, 50, 100, 200]


def simulate(regime_key: str, seed: int = 0):
    random.seed(seed)
    r = REGIMES[regime_key]
    kappa = r["kappa"]
    sigma = kappa / 20.0  # queue-poll signal, per paper.tex Sec 6

    results = []
    for T in T_VALUES:
        d = [random.uniform(r["d_lo"], r["d_hi"]) for _ in range(T)]
        ell = [random.uniform(r["ell_lo"], r["ell_hi"]) for _ in range(T)]

        t_naive = sum(d) + T * kappa + sum(ell)
        t_persistent = sum(d) + T * sigma + sum(ell)
        delta = T * (kappa - sigma)
        frac = delta / t_naive if t_naive > 0 else 0.0

        results.append(dict(
            T=T,
            t_naive_ms=round(t_naive, 4),
            t_persistent_ms=round(t_persistent, 4),
            delta_ms=round(delta, 4),
            delta_over_naive=round(frac, 6),
        ))
    return dict(regime=regime_key, label=r["label"], kappa_ms=kappa, sigma_ms=sigma, results=results)


# ─────────────────────────────────────────────────────────────────────────────
# Tiny transformer block used by --real-model
# ─────────────────────────────────────────────────────────────────────────────

def _build_transformer_block(n_layers: int = 2, dim: int = 256,
                              n_heads: int = 4, mlp_ratio: int = 4,
                              device: str = "cuda"):
    """Return a small but structurally realistic transformer stack.

    Each layer contains:
      • LayerNorm  (pre-norm)
      • torch.nn.MultiheadAttention  (batch_first=True)
      • LayerNorm  (pre-norm for MLP)
      • 2-layer MLP  (dim → mlp_ratio*dim → dim)  with GELU activation

    Parameters mirror a tiny LLM decode block: dim=256, 4 heads,
    seq_len=1 (single-token decode step).  The model is eval()-mode so
    no grad bookkeeping inflates kernel counts.
    """
    import torch
    import torch.nn as nn

    class TransformerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn  = nn.MultiheadAttention(dim, n_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp   = nn.Sequential(
                nn.Linear(dim, mlp_ratio * dim),
                nn.GELU(),
                nn.Linear(mlp_ratio * dim, dim),
            )

        def forward(self, x):
            # Pre-norm attention with residual
            h = self.norm1(x)
            h, _ = self.attn(h, h, h, need_weights=False)
            x = x + h
            # Pre-norm MLP with residual
            x = x + self.mlp(self.norm2(x))
            return x

    layers = nn.Sequential(*[TransformerLayer() for _ in range(n_layers)])
    return layers.to(device).eval()


def _count_cuda_kernels(model, x) -> int:
    """Count the number of CUDA kernels dispatched by one forward pass.

    Uses torch.profiler with CUDA activity recording.  Returns the count
    of cudaLaunchKernel (and cudaLaunchKernelExC) events — i.e. the real
    n_ops that the driver sees per agent step with naive (non-graph) dispatch.
    """
    import torch
    from torch.profiler import profile, ProfilerActivity

    # One warmup pass so lazy init / cuBLAS plan selection does not count.
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            with torch.no_grad():
                _ = model(x)
    torch.cuda.synchronize()

    # Each entry in key_averages corresponds to one kernel *type*; the
    # .count field is how many times it was launched in this trace.
    total_kernels = sum(e.count for e in prof.key_averages())
    return total_kernels


def measure_kappa_real_model(n_steps: int = 500,
                              n_layers: int = 2, dim: int = 256,
                              n_heads: int = 4) -> tuple:
    """Measure kappa for a real transformer block via naive per-step dispatch.

    Each step runs a full forward pass through the n_layers transformer
    (MHA + MLP + LayerNorm × 2 per layer).  Returns (kappa_ms, kernel_count).
    """
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected; use --mode simulate instead.")

    device = torch.device("cuda")
    model = _build_transformer_block(n_layers, dim, n_heads, device=device)
    # seq_len=1 mimics single-token autoregressive decode.
    x = torch.randn(1, 1, dim, device=device)

    # Count real kernel launches before timing.
    kernel_count = _count_cuda_kernels(model, x)

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for _ in range(n_steps):
            _ = model(x)
    end.record()
    torch.cuda.synchronize()

    kappa_ms = start.elapsed_time(end) / n_steps
    return kappa_ms, kernel_count


def measure_sigma_real_model(n_steps: int = 500,
                              n_layers: int = 2, dim: int = 256,
                              n_heads: int = 4) -> float:
    """Measure sigma for a real transformer block via CUDA-graph replay.

    The entire forward pass is captured as ONE CUDA graph; replaying it
    once per step eliminates all Python/driver dispatch overhead.
    Returns sigma_ms.
    """
    import torch

    device = torch.device("cuda")
    model = _build_transformer_block(n_layers, dim, n_heads, device=device)
    x = torch.randn(1, 1, dim, device=device)

    # Warmup before capture (required — CUDA graphs need a clean stream).
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
    torch.cuda.synchronize()

    # Capture the full forward pass as a single CUDA graph.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            _ = model(x)

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_steps):
        g.replay()
    end.record()
    torch.cuda.synchronize()

    sigma_ms = start.elapsed_time(end) / n_steps
    return sigma_ms


def print_real_model_comparison(synth_kappa: float, synth_sigma: float,
                                 n_ops: int,
                                 real_kappa: float, real_sigma: float,
                                 real_kernel_count: int) -> dict:
    """Print a side-by-side comparison of synthetic vs real-model overhead."""
    kappa_pct = abs(real_kappa - synth_kappa) / real_kappa * 100
    sigma_pct = abs(real_sigma - synth_sigma) / real_sigma * 100
    kappa_ratio = synth_kappa / real_kappa
    sigma_ratio = synth_sigma / real_sigma

    print("\n" + "═" * 68)
    print("  --real-model vs --ops-per-step comparison")
    print("═" * 68)
    print(f"  {'Metric':<28} {'Synthetic (N='+str(n_ops)+')':>16}  {'Real model':>12}")
    print(f"  {'':<28} {'':>16}  {'(2L/256d/4h)':>12}")
    print("  " + "─" * 64)
    print(f"  {'kappa (naive dispatch, ms)':<28} {synth_kappa:>16.4f}  {real_kappa:>12.4f}")
    print(f"  {'sigma (graph replay, ms)':<28} {synth_sigma:>16.4f}  {real_sigma:>12.4f}")
    print(f"  {'kappa/sigma ratio':<28} {synth_kappa/synth_sigma:>16.2f}x  {real_kappa/real_sigma:>11.2f}x")
    print(f"  {'kernel launches / step':<28} {n_ops:>16}   {real_kernel_count:>11}")
    print("  " + "─" * 64)
    print(f"  kappa error vs real model : {kappa_pct:6.1f}%  "
          f"({'over' if synth_kappa > real_kappa else 'under'}-estimate, ratio {kappa_ratio:.2f}x)")
    print(f"  sigma error vs real model : {sigma_pct:6.1f}%  "
          f"({'over' if synth_sigma > real_sigma else 'under'}-estimate, ratio {sigma_ratio:.2f}x)")
    equiv_ops = round(real_kernel_count)
    print(f"\n  The real model dispatches {real_kernel_count} CUDA kernels/step.")
    if real_kernel_count <= n_ops * 1.15 and real_kernel_count >= n_ops * 0.85:
        verdict = f"--ops-per-step={n_ops} is a GOOD approximation (within 15%)."
    elif real_kernel_count < n_ops:
        verdict = (f"--ops-per-step={n_ops} OVER-counts by "
                   f"{n_ops - real_kernel_count} ops; try --ops-per-step={equiv_ops}.")
    else:
        verdict = (f"--ops-per-step={n_ops} UNDER-counts by "
                   f"{real_kernel_count - n_ops} ops; try --ops-per-step={equiv_ops}.")
    print(f"  {verdict}")
    print("═" * 68)

    return dict(
        synth_kappa_ms=round(synth_kappa, 5),
        synth_sigma_ms=round(synth_sigma, 5),
        synth_n_ops=n_ops,
        real_kappa_ms=round(real_kappa, 5),
        real_sigma_ms=round(real_sigma, 5),
        real_kernel_count=real_kernel_count,
        kappa_error_pct=round(kappa_pct, 2),
        sigma_error_pct=round(sigma_pct, 2),
        kappa_ratio=round(kappa_ratio, 4),
        sigma_ratio=round(sigma_ratio, 4),
        verdict=verdict,
    )


def _run_op_chain(x: "torch.Tensor", n_ops: int) -> None:
    """Execute a chain of n_ops sequential small kernel launches on x.

    Alternates add / mul / relu to mimic the heterogeneous mix of
    element-wise ops that appear across attention + MLP + norm passes
    in a transformer layer.  Each call dispatches a distinct CUDA
    kernel, so n_ops launches hit the driver per simulated agent step.
    """
    for i in range(n_ops):
        op = i % 3
        if op == 0:
            x.add_(0.01)
        elif op == 1:
            x.mul_(1.001)
        else:
            x.relu_()


def measure_kappa_gpu(n_steps: int = 2000, n_ops: int = 48):
    """Measure per-step kernel-launch overhead for a chain of n_ops kernels.

    Each 'step' dispatches n_ops sequential small kernels (add / mul /
    relu, alternating) — approximating the op count of a few transformer
    layers (attention + MLP + norms).  kappa is the average wall-time of
    one full chain, measured with CUDA events.

    Requires torch + a CUDA device.
    """
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected; use --mode simulate instead.")

    device = torch.device("cuda")
    # Small tensor: overhead, not compute, is what we want to measure.
    x = torch.ones(256, device=device)

    # Warmup: a few full chains so driver/JIT is hot.
    for _ in range(20):
        _run_op_chain(x, n_ops)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_steps):
        _run_op_chain(x, n_ops)  # n_ops fresh kernel launches per step
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    kappa_ms = total_ms / n_steps   # average cost of one n_ops chain
    return kappa_ms


def measure_sigma_gpu(n_steps: int = 2000, n_ops: int = 48):
    """Approximate sigma as the cost of replaying a CUDA graph of the
    full n_ops chain.

    The whole chain (n_ops kernels) is captured as ONE CUDA graph and
    replayed once per step — eliminating per-launch Python/driver
    dispatch overhead while keeping the same compute.  This is the
    framework-level stand-in for device-side queue signalling cost
    (see triton_persistent_agent_kernel.py and paper.tex Sec 4).
    True device-side polling (persistent_kernel.cu) is typically even
    cheaper than this upper bound.
    """
    import torch

    device = torch.device("cuda")
    x = torch.ones(256, device=device)

    # Warmup before graph capture.
    for _ in range(20):
        _run_op_chain(x, n_ops)
    torch.cuda.synchronize()

    # Capture the entire n_ops chain as a single CUDA graph.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _run_op_chain(x, n_ops)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_steps):
        g.replay()  # one graph replay = one agent step, no fresh launches
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    sigma_ms = total_ms / n_steps
    return sigma_ms


def run_gpu(regime_key: str, n_ops: int = 48, real_model: bool = False):
    r = REGIMES[regime_key]
    print(f"  measuring kappa (n_ops={n_ops}, naive launches)...")
    kappa = measure_kappa_gpu(n_ops=n_ops)
    print(f"  measuring sigma (n_ops={n_ops}, CUDA graph replay)...")
    sigma = measure_sigma_gpu(n_ops=n_ops)

    comparison = None
    if real_model:
        print(f"  measuring kappa (real transformer block, naive)...")
        real_kappa, kernel_count = measure_kappa_real_model()
        print(f"  measuring sigma (real transformer block, CUDA graph)...")
        real_sigma = measure_sigma_real_model()
        comparison = print_real_model_comparison(
            kappa, sigma, n_ops, real_kappa, real_sigma, kernel_count
        )

    results = []
    random.seed(0)
    for T in T_VALUES:
        d = [random.uniform(r["d_lo"], r["d_hi"]) for _ in range(T)]
        ell = [random.uniform(r["ell_lo"], r["ell_hi"]) for _ in range(T)]

        t_naive = sum(d) + T * kappa + sum(ell)
        t_persistent = sum(d) + T * sigma + sum(ell)
        delta = T * (kappa - sigma)
        frac = delta / t_naive if t_naive > 0 else 0.0

        results.append(dict(
            T=T,
            t_naive_ms=round(t_naive, 4),
            t_persistent_ms=round(t_persistent, 4),
            delta_ms=round(delta, 4),
            delta_over_naive=round(frac, 6),
        ))

    return dict(regime=regime_key, label=r["label"],
                kappa_ms=round(kappa, 5), sigma_ms=round(sigma, 5),
                n_ops_per_step=n_ops,
                real_model_comparison=comparison,
                results=results)


def print_table(payload):
    n_ops_str = (f"  ops_per_step={payload['n_ops_per_step']}"
                 if "n_ops_per_step" in payload else "")
    print(f"\nRegime {payload['regime']} ({payload['label']})  "
          f"kappa={payload['kappa_ms']:.4f} ms  sigma={payload['sigma_ms']:.4f} ms"
          f"{n_ops_str}")
    print(f"{'T':>5} {'T_naive(ms)':>14} {'T_persist(ms)':>16} "
          f"{'Delta(ms)':>12} {'Delta/T_naive':>15}")
    for row in payload["results"]:
        print(f"{row['T']:>5} {row['t_naive_ms']:>14.3f} {row['t_persistent_ms']:>16.3f} "
              f"{row['delta_ms']:>12.3f} {row['delta_over_naive']:>15.4%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["simulate", "gpu"], default="simulate")
    parser.add_argument("--regime", choices=["A", "B", "both"], default="both")
    parser.add_argument("--json-out", type=str, default=None,
                         help="optional path to write results as JSON")
    parser.add_argument("--ops-per-step", type=int, default=48,
                         help="number of sequential kernel launches per simulated "
                              "agent step (gpu mode only); default 48, representing "
                              "~attention+MLP+norm ops across a few transformer layers")
    parser.add_argument("--real-model", action="store_true",
                         help="(gpu mode only) also benchmark a tiny real transformer "
                              "block (2 layers, dim=256, 4 heads, MHA+MLP) as the "
                              "decode segment and compare its kappa/sigma against the "
                              "synthetic --ops-per-step chain")
    args = parser.parse_args()

    regimes = ["A", "B"] if args.regime == "both" else [args.regime]
    all_payloads = []

    for reg in regimes:
        if args.mode == "simulate":
            payload = simulate(reg)
        else:
            payload = run_gpu(reg, n_ops=args.ops_per_step,
                              real_model=args.real_model)
        print_table(payload)
        all_payloads.append(payload)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_payloads, f, indent=2)
        print(f"\nWrote results to {args.json_out}")
