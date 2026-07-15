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

Usage:
  python3 benchmark.py --mode simulate --regime A
  python3 benchmark.py --mode simulate --regime B
  python3 benchmark.py --mode gpu --regime A   # needs a CUDA GPU
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


def run_gpu(regime_key: str, n_ops: int = 48):
    r = REGIMES[regime_key]
    print(f"  measuring kappa (n_ops={n_ops}, naive launches)...")
    kappa = measure_kappa_gpu(n_ops=n_ops)
    print(f"  measuring sigma (n_ops={n_ops}, CUDA graph replay)...")
    sigma = measure_sigma_gpu(n_ops=n_ops)

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
    args = parser.parse_args()

    regimes = ["A", "B"] if args.regime == "both" else [args.regime]
    all_payloads = []

    for reg in regimes:
        if args.mode == "simulate":
            payload = simulate(reg)
        else:
            payload = run_gpu(reg, n_ops=args.ops_per_step)
        print_table(payload)
        all_payloads.append(payload)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_payloads, f, indent=2)
        print(f"\nWrote results to {args.json_out}")
