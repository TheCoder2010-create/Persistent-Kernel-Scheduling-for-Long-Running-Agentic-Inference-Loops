"""
benchmark.py

Benchmark harness for the cost model in paper.tex, Section 3:

    T_naive(T)      = sum_t ( d_t + kappa + ell_t )
    T_persistent(T) = sum_t ( d_t + sigma + ell_t )
    Delta(T)        = T * (kappa - sigma)

Modes:
  --mode simulate        CPU-only cost-model validation (hardcoded kappa/sigma).
  --mode gpu             Measure kappa/sigma via PyTorch (CUDA-graph replay for sigma).
  --mode persistent-cuda Run persistent_kernel.exe and use its clock64 device-side
                         sigma + derived kappa (real 2L/256d/4h forward pass).
                         This is the measurement that backs paper.tex Table 1
                         "Measured (GPU)" after the device-side sigma fix.

Flags:
  --ops-per-step N       Synthetic chain length for --mode gpu (default 48).
  --real-model           (gpu mode) also benchmark a PyTorch transformer block.
  --kernel-exe PATH      Path to persistent_kernel binary (default: ./persistent_kernel.exe).
  --kernel-json PATH     Where persistent-cuda writes kernel stats JSON.

Usage:
  python benchmark.py --mode simulate --regime both
  python benchmark.py --mode gpu --regime both
  python benchmark.py --mode persistent-cuda --regime both --json-out benchmarks/results/device_sigma_results.json
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


REGIMES = {
    "A": dict(d_lo=5.0, d_hi=15.0, ell_lo=50.0, ell_hi=200.0, kappa=1.5,
              label="tool-latency-dominated"),
    "B": dict(d_lo=2.0, d_hi=5.0, ell_lo=0.5, ell_hi=2.0, kappa=1.5,
              label="launch-overhead-dominated"),
}

T_VALUES = [1, 10, 50, 100, 200]


def simulate(regime_key: str, seed: int = 0):
    random.seed(seed)
    r = REGIMES[regime_key]
    kappa = r["kappa"]
    sigma = kappa / 20.0

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
    return dict(regime=regime_key, label=r["label"], kappa_ms=kappa, sigma_ms=sigma,
                sigma_source="modeled_kappa_over_20", results=results)


def _cost_model_rows(regime_key: str, kappa: float, sigma: float, seed: int = 0):
    random.seed(seed)
    r = REGIMES[regime_key]
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
    return results


def _build_transformer_block(n_layers: int = 2, dim: int = 256,
                              n_heads: int = 4, mlp_ratio: int = 4,
                              device: str = "cuda"):
    import torch
    import torch.nn as nn

    class TransformerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, mlp_ratio * dim),
                nn.GELU(),
                nn.Linear(mlp_ratio * dim, dim),
            )

        def forward(self, x):
            h = self.norm1(x)
            h, _ = self.attn(h, h, h, need_weights=False)
            x = x + h
            x = x + self.mlp(self.norm2(x))
            return x

    return nn.Sequential(*[TransformerLayer() for _ in range(n_layers)]).to(device).eval()


def _count_cuda_kernels(model, x) -> int:
    import warnings
    import torch
    from torch.profiler import profile, ProfilerActivity

    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            with torch.no_grad():
                _ = model(x)
    torch.cuda.synchronize()
    return sum(e.count for e in prof.key_averages())


def measure_kappa_real_model(n_steps: int = 500, n_layers: int = 2,
                              dim: int = 256, n_heads: int = 4) -> tuple:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected; use --mode simulate instead.")
    device = torch.device("cuda")
    model = _build_transformer_block(n_layers, dim, n_heads, device=device)
    x = torch.randn(1, 1, dim, device=device)
    kernel_count = _count_cuda_kernels(model, x)
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        for _ in range(n_steps):
            _ = model(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_steps, kernel_count


def measure_sigma_real_model(n_steps: int = 500, n_layers: int = 2,
                              dim: int = 256, n_heads: int = 4) -> float:
    """Proxy sigma via CUDA-graph replay (NOT device-side queue poll)."""
    import torch
    device = torch.device("cuda")
    model = _build_transformer_block(n_layers, dim, n_heads, device=device)
    x = torch.randn(1, 1, dim, device=device)
    with torch.no_grad():
        for _ in range(20):
            _ = model(x)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            _ = model(x)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_steps):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_steps


def print_real_model_comparison(synth_kappa, synth_sigma, n_ops,
                                 real_kappa, real_sigma, real_kernel_count) -> dict:
    kappa_pct = abs(real_kappa - synth_kappa) / real_kappa * 100
    sigma_pct = abs(real_sigma - synth_sigma) / real_sigma * 100
    print("\n" + "=" * 68)
    print("  --real-model vs --ops-per-step comparison (graph-replay sigma PROXY)")
    print("=" * 68)
    print(f"  kappa synth={synth_kappa:.4f} real={real_kappa:.4f}")
    print(f"  sigma synth={synth_sigma:.4f} real={real_sigma:.4f} (both graph replay)")
    print(f"  kernels/step real={real_kernel_count} synth_ops={n_ops}")
    equiv_ops = round(real_kernel_count)
    if n_ops * 0.85 <= real_kernel_count <= n_ops * 1.15:
        verdict = f"--ops-per-step={n_ops} is a GOOD approximation (within 15%)."
    elif real_kernel_count < n_ops:
        verdict = (f"--ops-per-step={n_ops} OVER-counts; try --ops-per-step={equiv_ops}.")
    else:
        verdict = (f"--ops-per-step={n_ops} UNDER-counts; try --ops-per-step={equiv_ops}.")
    print(f"  {verdict}")
    return dict(
        synth_kappa_ms=round(synth_kappa, 5),
        synth_sigma_ms=round(synth_sigma, 5),
        synth_n_ops=n_ops,
        real_kappa_ms=round(real_kappa, 5),
        real_sigma_ms=round(real_sigma, 5),
        real_kernel_count=real_kernel_count,
        kappa_error_pct=round(kappa_pct, 2),
        sigma_error_pct=round(sigma_pct, 2),
        verdict=verdict,
    )


def _run_op_chain(x, n_ops: int) -> None:
    for i in range(n_ops):
        op = i % 3
        if op == 0:
            x.add_(0.01)
        elif op == 1:
            x.mul_(1.001)
        else:
            x.relu_()


def measure_kappa_gpu(n_steps: int = 2000, n_ops: int = 48):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected; use --mode simulate instead.")
    x = torch.ones(256, device="cuda")
    for _ in range(20):
        _run_op_chain(x, n_ops)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_steps):
        _run_op_chain(x, n_ops)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_steps


def measure_sigma_gpu(n_steps: int = 2000, n_ops: int = 48):
    """Proxy sigma: CUDA-graph replay (NOT clock64 device-side poll)."""
    import torch
    x = torch.ones(256, device="cuda")
    for _ in range(20):
        _run_op_chain(x, n_ops)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _run_op_chain(x, n_ops)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_steps):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_steps


def run_gpu(regime_key: str, n_ops: int = 48, real_model: bool = False):
    r = REGIMES[regime_key]
    print(f"  measuring kappa (n_ops={n_ops}, naive launches)...")
    kappa = measure_kappa_gpu(n_ops=n_ops)
    print(f"  measuring sigma (n_ops={n_ops}, CUDA graph replay PROXY)...")
    sigma = measure_sigma_gpu(n_ops=n_ops)
    comparison = None
    if real_model:
        print("  measuring kappa (real transformer block, naive)...")
        real_kappa, kernel_count = measure_kappa_real_model()
        print("  measuring sigma (real transformer, CUDA graph PROXY)...")
        real_sigma = measure_sigma_real_model()
        comparison = print_real_model_comparison(
            kappa, sigma, n_ops, real_kappa, real_sigma, kernel_count)
    return dict(
        regime=regime_key, label=r["label"],
        kappa_ms=round(kappa, 5), sigma_ms=round(sigma, 5),
        n_ops_per_step=n_ops,
        sigma_source="cuda_graph_replay_proxy",
        real_model_comparison=comparison,
        results=_cost_model_rows(regime_key, kappa, sigma),
    )


def measure_persistent_cuda(kernel_exe: str, kernel_json: str) -> dict:
    """Run persistent_kernel.exe; load clock64 device-side sigma + kappa."""
    exe = Path(kernel_exe)
    if not exe.exists():
        raise SystemExit(
            f"Kernel binary not found: {exe}\n"
            "Build with: compile.bat"
        )
    out_path = Path(kernel_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(exe), "--steps", "10", "--tool-latency-ms", "1",
           "--json-out", str(out_path)]
    print(f"  running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(exe.parent))
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"persistent_kernel failed with exit {proc.returncode}")
    if not out_path.exists():
        raise SystemExit(f"Kernel did not write {out_path}")
    with open(out_path, encoding="utf-8") as f:
        stats = json.load(f)
    for k in ("kappa_ms", "sigma_ms", "sigma_source"):
        if k not in stats:
            raise SystemExit(f"Kernel JSON missing {k}: {out_path}")
    if stats["sigma_source"] != "clock64_device_side_queue_poll":
        raise SystemExit(f"Unexpected sigma_source: {stats['sigma_source']}")
    return stats


def run_persistent_cuda(regime_key: str, kernel_stats: dict):
    r = REGIMES[regime_key]
    kappa = float(kernel_stats["kappa_ms"])
    sigma = float(kernel_stats["sigma_ms"])
    return dict(
        regime=regime_key,
        label=r["label"],
        kappa_ms=round(kappa, 8),
        sigma_ms=round(sigma, 8),
        sigma_source=kernel_stats["sigma_source"],
        kappa_source=kernel_stats.get("kappa_source", "persistent_kernel"),
        compute_ms=kernel_stats.get("compute_ms"),
        device=kernel_stats.get("device"),
        model=kernel_stats.get("model"),
        results=_cost_model_rows(regime_key, kappa, sigma),
    )


def print_table(payload):
    n_ops_str = (f"  ops_per_step={payload['n_ops_per_step']}"
                 if "n_ops_per_step" in payload else "")
    src = payload.get("sigma_source", "")
    src_str = f"  sigma_source={src}" if src else ""
    print(f"\nRegime {payload['regime']} ({payload['label']})  "
          f"kappa={payload['kappa_ms']:.6f} ms  sigma={payload['sigma_ms']:.6f} ms"
          f"{n_ops_str}{src_str}")
    print(f"{'T':>5} {'T_naive(ms)':>14} {'T_persist(ms)':>16} "
          f"{'Delta(ms)':>12} {'Delta/T_naive':>15}")
    for row in payload["results"]:
        print(f"{row['T']:>5} {row['t_naive_ms']:>14.3f} {row['t_persistent_ms']:>16.3f} "
              f"{row['delta_ms']:>12.3f} {row['delta_over_naive']:>15.4%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",
                        choices=["simulate", "gpu", "persistent-cuda"],
                        default="simulate")
    parser.add_argument("--regime", choices=["A", "B", "both"], default="both")
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--ops-per-step", type=int, default=48)
    parser.add_argument("--real-model", action="store_true")
    parser.add_argument("--kernel-exe", type=str,
                        default=str(Path(__file__).resolve().parent / "persistent_kernel.exe"))
    parser.add_argument("--kernel-json", type=str,
                        default=str(Path(__file__).resolve().parent
                                    / "benchmarks" / "results" / "device_sigma_kernel.json"))
    args = parser.parse_args()

    regimes = ["A", "B"] if args.regime == "both" else [args.regime]
    all_payloads = []
    kernel_stats = None

    if args.mode == "persistent-cuda":
        kernel_stats = measure_persistent_cuda(args.kernel_exe, args.kernel_json)
        print(f"  device-side sigma={kernel_stats['sigma_ms']:.6f} ms  "
              f"kappa={kernel_stats['kappa_ms']:.6f} ms  "
              f"(compute={kernel_stats.get('compute_ms', float('nan')):.6f} ms)")

    for reg in regimes:
        if args.mode == "simulate":
            payload = simulate(reg)
        elif args.mode == "gpu":
            payload = run_gpu(reg, n_ops=args.ops_per_step, real_model=args.real_model)
        else:
            payload = run_persistent_cuda(reg, kernel_stats)
        print_table(payload)
        all_payloads.append(payload)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(all_payloads, f, indent=2)
        print(f"\nWrote results to {args.json_out}")
