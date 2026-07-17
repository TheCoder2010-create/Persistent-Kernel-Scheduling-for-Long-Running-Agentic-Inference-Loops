"""
benchmark.py — Multi-Trajectory Persistent Agent Kernel Harness

Cost model (design.md §7):

    T_persistent(N) = kappa/N + sigma + d_t + tool_latency
    T_naive(N)      = N * (kappa + d_t + tool_latency)
    Delta(N)        = (N-1)*kappa - N*sigma

Modes:
  --mode simulate              Analytical cost model (CPU, no GPU).
  --mode gpu                   Single-trajectory kappa/sigma via PyTorch.
  --mode persistent-cuda       Run persistent_kernel.exe, parse device clock64 data.
  --mode multi-trajectory      Run persistent_kernel.exe with --n-trajectories.
  --mode small-model           Load Qwen2.5-0.5B INT4, measure real d_t.
  --mode multi-trajectory-sim  Analytical multi-trajectory cost model only.

Flags:
  --regime A|B|both            (simulate/gpu/persistent-cuda modes).
  --ops-per-step N             Synthetic chain length (default 48).
  --sweep-n N1,N2,...          Multi-trajectory sweep (default 1,2,4,8,16,32,64,128).
  --steps T                    Agent-loop steps (default 100).
  --tool-latency-ms F          Tool-call latency (default 1.0).
  --kernel-exe PATH            Path to persistent_kernel.exe.
  --json-out PATH              Write results JSON.
  --verbose                    Extra output.

Usage:
  python benchmark.py --mode simulate --regime both
  python benchmark.py --mode multi-trajectory-sim --sweep-n 1,2,4,8,16,32
  python benchmark.py --mode multi-trajectory --sweep-n 1,2,4 --steps 20
  python benchmark.py --mode small-model
"""

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Regimes (original)
# ---------------------------------------------------------------------------
REGIMES = {
    "A": dict(d_lo=5.0, d_hi=15.0, ell_lo=50.0, ell_hi=200.0, kappa=1.5,
              label="tool-latency-dominated"),
    "B": dict(d_lo=2.0, d_hi=5.0, ell_lo=0.5, ell_hi=2.0, kappa=1.5,
              label="launch-overhead-dominated"),
}

T_VALUES = [1, 10, 50, 100, 200]


# ===================================================================
#  Analytical cost model (original, single-trajectory)
# ===================================================================
def simulate(regime_key, seed=0):
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
    return dict(regime=regime_key, label=r["label"], kappa_ms=kappa,
                sigma_ms=sigma, sigma_source="modeled_kappa_over_20",
                results=results)


def _cost_model_rows(regime_key, kappa, sigma, seed=0):
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
        results.append(dict(T=T, t_naive_ms=round(t_naive, 4),
                            t_persistent_ms=round(t_persistent, 4),
                            delta_ms=round(delta, 4),
                            delta_over_naive=round(frac, 6)))
    return results


# ===================================================================
#  Multi-trajectory analytical cost model
# ===================================================================
MULTI_COLUMNS = [
    "N", "persistent_us_per_traj_step", "naive_us_per_traj_step",
    "delta_us", "saving_pct",
]


def cost_model_multi(N, kappa_us=5.82, sigma_us=0.095, decode_us=200.0,
                     tool_latency_us=500.0, T=100):
    """Multi-trajectory cost model (design.md §7).

    Returns per-step numbers for N trajectories.
    """
    kappa_ms = kappa_us / 1000.0
    sigma_ms = sigma_us / 1000.0
    decode_ms = decode_us / 1000.0
    tool_ms = tool_latency_us / 1000.0

    # Per trajectory per step
    persistent_us = kappa_us / N + sigma_us + decode_us + tool_latency_us
    naive_us = kappa_us + decode_us + tool_latency_us

    # Total over all trajectories for T steps
    total_persistent_ms = (persistent_us / 1000.0) * N * T
    total_naive_ms = (naive_us / 1000.0) * N * T

    delta = total_naive_ms - total_persistent_ms
    frac = delta / total_naive_ms if total_naive_ms > 0 else 0.0

    # Also compute the per-step saving per trajectory
    delta_per_traj_step = naive_us - persistent_us

    return {
        "N": N,
        "kappa_us": kappa_us,
        "sigma_us": sigma_us,
        "decode_us": decode_us,
        "tool_latency_us": tool_latency_us,
        "T": T,
        "persistent_us_per_traj_step": round(persistent_us, 4),
        "naive_us_per_traj_step": round(naive_us, 4),
        "delta_us_per_traj_step": round(delta_per_traj_step, 4),
        "total_persistent_ms": round(total_persistent_ms, 4),
        "total_naive_ms": round(total_naive_ms, 4),
        "delta_ms": round(delta, 4),
        "saving_pct": round(frac * 100.0, 4),
    }


# ===================================================================
#  GPU measurement helpers (single-trajectory, original)
# ===================================================================
def _build_transformer_block(n_layers=2, dim=256, n_heads=4, mlp_ratio=4,
                              device="cuda"):
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


def _count_cuda_kernels(model, x):
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


def _run_op_chain(x, n_ops):
    for i in range(n_ops):
        op = i % 3
        if op == 0:
            x.add_(0.01)
        elif op == 1:
            x.mul_(1.001)
        else:
            x.relu_()


def measure_kappa_gpu(n_steps=2000, n_ops=48):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU; use --mode simulate instead.")
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


def measure_sigma_gpu(n_steps=2000, n_ops=48):
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


def run_gpu(regime_key, n_ops=48, real_model=False):
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
        comparison = dict(
            synth_kappa_ms=round(kappa, 5),
            synth_sigma_ms=round(sigma, 5),
            synth_n_ops=n_ops,
            real_kappa_ms=round(real_kappa, 5),
            real_sigma_ms=round(real_sigma, 5),
            real_kernel_count=kernel_count,
        )
    return dict(regime=regime_key, label=r["label"],
                kappa_ms=round(kappa, 5), sigma_ms=round(sigma, 5),
                n_ops_per_step=n_ops,
                sigma_source="cuda_graph_replay_proxy",
                real_model_comparison=comparison,
                results=_cost_model_rows(regime_key, kappa, sigma))


def measure_kappa_real_model(n_steps=500, n_layers=2, dim=256, n_heads=4):
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU; use --mode simulate instead.")
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


def measure_sigma_real_model(n_steps=500, n_layers=2, dim=256, n_heads=4):
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


# ===================================================================
#  Persistent CUDA driver (single-trajectory, original)
# ===================================================================
def measure_persistent_cuda(kernel_exe, kernel_json):
    exe = Path(kernel_exe)
    if not exe.exists():
        raise SystemExit(
            f"Kernel binary not found: {exe}\nBuild with: nvcc -O3 -arch=sm_86 persistent_kernel.cu -o persistent_kernel.exe")
    out_path = Path(kernel_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(exe), "--steps", "10", "--tool-latency-ms", "1",
           "--n-trajectories", "1",
           "--json-out", str(out_path)]
    print(f"  running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(exe.parent))
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"persistent_kernel failed with exit {proc.returncode}")
    if not out_path.exists():
        raise SystemExit(f"Kernel did not write {out_path}")
    with open(out_path) as f:
        stats = json.load(f)
    for k in ("kappa_us", "sigma_us"):
        if k not in stats:
            raise SystemExit(f"Kernel JSON missing {k}: {out_path}")
    stats["kappa_ms"] = stats["kappa_us"] / 1000.0
    stats["sigma_ms"] = stats["sigma_us"] / 1000.0
    return stats


def run_persistent_cuda(regime_key, kernel_stats):
    r = REGIMES[regime_key]
    kappa = float(kernel_stats["kappa_ms"])
    sigma = float(kernel_stats["sigma_ms"])
    return dict(regime=regime_key, label=r["label"],
                kappa_ms=round(kappa, 8), sigma_ms=round(sigma, 8),
                sigma_source=kernel_stats.get("sigma_source", "clock64"),
                kappa_source=kernel_stats.get("kappa_source", "persistent_kernel"),
                compute_ms=kernel_stats.get("compute_ms"),
                device=kernel_stats.get("device"),
                model=kernel_stats.get("model"),
                results=_cost_model_rows(regime_key, kappa, sigma))


# ===================================================================
#  Multi-trajectory sweep (via persistent_kernel.exe)
# ===================================================================
def run_multi_trajectory_sweep(kernel_exe, N_list, T, tool_latency_ms,
                                json_out=None, verbose=False):
    """Sweep persistent_kernel.exe across N_list values."""
    exe = Path(kernel_exe)
    if not exe.exists():
        raise SystemExit(f"Kernel binary not found: {exe}")

    all_results = []
    for n in N_list:
        print(f"\n=== N={n} ===")
        tmp_json = Path(f"_mt_sweep_n{n}.json")
        cmd = [str(exe), "--steps", str(T), "--tool-latency-ms",
               str(tool_latency_ms), "--n-trajectories", str(n),
               "--json-out", str(tmp_json)]
        if verbose:
            cmd.append("--verbose")

        print(f"  running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              cwd=str(exe.parent))
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            print(f"  WARNING: kernel failed for N={n}, skipping")
            continue

        if tmp_json.exists():
            with open(tmp_json) as f:
                stats = json.load(f)
            all_results.append(stats)
            print(f"  sigma={stats.get('sigma_us', '?'):.4f} us  "
                  f"kappa={stats.get('kappa_us', '?'):.4f} us  "
                  f"saving={stats.get('saving_pct', '?'):.2f}%")
            if json_out is None:
                tmp_json.unlink()  # clean up temp file

    if json_out:
        with open(json_out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote {json_out}")
        # Clean up temp files
        for n in N_list:
            p = Path(f"_mt_sweep_n{n}.json")
            if p.exists():
                p.unlink()

    return all_results


# ===================================================================
#  Small model measurement (Qwen2.5-0.5B INT4)
# ===================================================================
def measure_small_model(n_steps=100, model_name="Qwen/Qwen2.5-0.5B-Instruct",
                         quantize="int4", tool_latency_s=0.001):
    """Load a small LLM, measure decode time per step (d_t).

    This gives the real d_t for the multi-trajectory regime.

    Falls back to synthetic measurement if the model cannot be loaded.
    """
    import time
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise SystemExit(
            "Need transformers + torch for small-model mode.\n"
            "Install: pip install transformers torch accelerate bitsandbytes")

    print(f"  Loading {model_name} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("  WARNING: no CUDA, falling back to CPU (slow)")

    load_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
    if quantize == "int4" and device == "cuda":
        try:
            load_kwargs["quantization_config"] = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
            }
        except Exception:
            print("  bitsandbytes not available, loading in fp16")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs)
        model.eval()
    except Exception as e:
        print(f"  Failed to load model: {e}")
        print("  Falling back to synthetic 256-dim transformer")
        model = _build_transformer_block(2, 256, 4, device=device)
        tokenizer = None

    # Measure single-step decode time
    import torch.nn as nn
    is_real = not isinstance(model, nn.Sequential)

    if is_real and tokenizer is not None:
        input_text = "The capital of France is"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        input_len = inputs.input_ids.shape[1]

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        torch.cuda.synchronize()

        # Timed
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            for _ in range(n_steps):
                _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
        d_t_us = total_ms / n_steps * 1000.0

        print(f"  {model_name} decode: {d_t_us:.2f} us/step  "
              f"(input_len={input_len}, steps={n_steps})")
    else:
        # Synthetic measurement
        x = torch.randn(1, 1, 256, device=device)
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
        total_ms = start.elapsed_time(end)
        d_t_us = total_ms / n_steps * 1000.0
        print(f"  Synthetic model decode: {d_t_us:.2f} us/step")

    return {
        "model": model_name if is_real else "synthetic_256dim",
        "quantize": quantize,
        "d_t_us": round(d_t_us, 2),
        "n_steps": n_steps,
        "d_t_ms": round(d_t_us / 1000.0, 4),
    }


# ===================================================================
#  Output formatting
# ===================================================================
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


def print_multi_table(results, d_t_us=200.0, tool_latency_us=500.0):
    """Print the multi-trajectory sweep results."""
    print(f"\n{'N':>4} {'persist(us)':>14} {'naive(us)':>12} "
          f"{'delta(us)':>12} {'saving%':>10}  notes")
    print("-" * 68)
    for r in results:
        N = r["N"]
        pers = r["persistent_us_per_traj_step"]
        naive = r["naive_us_per_traj_step"]
        delta = r["delta_us_per_traj_step"]
        pct = r["saving_pct"]
        note = ""
        if pct > 5:
            note = "** significant"
        elif pct > 1:
            note = "* notable"
        print(f"{N:>4} {pers:>14.2f} {naive:>12.2f} {delta:>12.2f} "
              f"{pct:>9.2f}%  {note}")


def print_cuda_multi_table(results):
    """Print raw CUDA multi-trajectory sweep results."""
    print(f"\n{'N':>4} {'sigma(us)':>12} {'kappa(us)':>12} "
          f"{'compute(us)':>14} {'wall/step(ms)':>16} {'delta(us)':>12} {'saving%':>10}")
    print("-" * 80)
    for r in results:
        N = r.get("n_trajectories", "?")
        sigma = r.get("sigma_us", 0)
        kappa = r.get("kappa_us", 0)
        comp = r.get("compute_us", 0)
        wall = r.get("wall_per_step_ms", 0)
        delta = r.get("delta_us_total", 0)
        pct = r.get("saving_pct", 0)
        print(f"{N:>4} {sigma:>12.4f} {kappa:>12.4f} {comp:>14.2f} "
              f"{wall:>16.4f} {delta:>12.2f} {pct:>9.2f}%")


# ===================================================================
#  CLI
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",
                        choices=["simulate", "gpu", "persistent-cuda",
                                 "multi-trajectory", "multi-trajectory-sim",
                                 "small-model"],
                        default="simulate")
    parser.add_argument("--regime", choices=["A", "B", "both"], default="both")
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--ops-per-step", type=int, default=48)
    parser.add_argument("--kernel-exe", type=str,
                        default=str(Path(__file__).resolve().parent / "persistent_kernel.exe"))
    parser.add_argument("--kernel-json", type=str,
                        default=str(Path(__file__).resolve().parent
                                    / "_kernel_stats.json"))
    parser.add_argument("--sweep-n", type=str, default="1,2,4,8,16,32,64,128")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--tool-latency-ms", type=float, default=1.0)
    parser.add_argument("--decode-us", type=float, default=200.0,
                        help="Decode time in us for analytical models")
    parser.add_argument("--kappa-us", type=float, default=5.82,
                        help="Kappa in us for analytical models")
    parser.add_argument("--sigma-us", type=float, default=0.095,
                        help="Sigma in us for analytical models")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.mode == "multi-trajectory-sim":
        N_list = [int(x.strip()) for x in args.sweep_n.split(",")]
        print(f"=== Multi-Trajectory Cost Model ===\n")
        print(f"  kappa={args.kappa_us} us  sigma={args.sigma_us} us  "
              f"decode={args.decode_us} us  tool_latency={args.tool_latency_ms} ms")
        results = []
        for n in N_list:
            r = cost_model_multi(
                n, kappa_us=args.kappa_us, sigma_us=args.sigma_us,
                decode_us=args.decode_us,
                tool_latency_us=args.tool_latency_ms * 1000.0,
                T=args.steps)
            results.append(r)
        print_multi_table(results, args.decode_us, args.tool_latency_ms * 1000.0)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nWrote {args.json_out}")

    elif args.mode == "multi-trajectory":
        N_list = [int(x.strip()) for x in args.sweep_n.split(",")]
        print(f"=== Multi-Trajectory CUDA Sweep ===\n")
        print(f"  Steps={args.steps}  Tool latency={args.tool_latency_ms} ms")
        print(f"  Sweeping N = {N_list}")
        results = run_multi_trajectory_sweep(
            args.kernel_exe, N_list, args.steps, args.tool_latency_ms,
            json_out=args.json_out, verbose=args.verbose)
        if results:
            print_cuda_multi_table(results)

    elif args.mode == "small-model":
        print(f"=== Small Model Decode Measurement ===\n")
        result = measure_small_model(
            n_steps=args.steps,
            tool_latency_s=args.tool_latency_ms / 1000.0)
        print(f"\nResult: {json.dumps(result, indent=2)}")
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Wrote {args.json_out}")

        # Also run the cost model with the measured d_t
        print(f"\n=== Projected Multi-Trajectory Benefit ===\n"
              f"  Using d_t = {result['d_t_us']} us")
        N_list = [int(x.strip()) for x in args.sweep_n.split(",")]
        for n in N_list:
            r = cost_model_multi(n, kappa_us=args.kappa_us,
                                  sigma_us=args.sigma_us,
                                  decode_us=result['d_t_us'],
                                  tool_latency_us=args.tool_latency_ms * 1000.0,
                                  T=args.steps)
            print(f"  N={n:3d}:  saving={r['saving_pct']:.2f}%  "
                  f"persistent={r['persistent_us_per_traj_step']:.1f} us/traj-step  "
                  f"delta={r['delta_us_per_traj_step']:.1f} us/traj-step")

    elif args.mode == "simulate":
        regimes = ["A", "B"] if args.regime == "both" else [args.regime]
        all_payloads = []
        for reg in regimes:
            payload = simulate(reg)
            print_table(payload)
            all_payloads.append(payload)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(all_payloads, f, indent=2)
            print(f"\nWrote {args.json_out}")

    elif args.mode == "gpu":
        regimes = ["A", "B"] if args.regime == "both" else [args.regime]
        all_payloads = []
        for reg in regimes:
            payload = run_gpu(reg, n_ops=args.ops_per_step)
            print_table(payload)
            all_payloads.append(payload)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(all_payloads, f, indent=2)
            print(f"\nWrote {args.json_out}")

    elif args.mode == "persistent-cuda":
        kernel_stats = measure_persistent_cuda(args.kernel_exe, args.kernel_json)
        print(f"  device-side sigma={kernel_stats['sigma_ms']:.6f} ms  "
              f"kappa={kernel_stats['kappa_ms']:.6f} ms  "
              f"(compute={kernel_stats.get('compute_us', '?'):.2f} us)")
        regimes = ["A", "B"] if args.regime == "both" else [args.regime]
        all_payloads = []
        for reg in regimes:
            payload = run_persistent_cuda(reg, kernel_stats)
            print_table(payload)
            all_payloads.append(payload)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(all_payloads, f, indent=2)
            print(f"\nWrote {args.json_out}")
