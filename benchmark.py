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


def measure_kappa_gpu(n_launches: int = 2000):
    """Measure actual CUDA kernel-launch overhead via a trivial kernel,
    timed with CUDA events. Requires torch + a CUDA device."""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU detected; use --mode simulate instead.")

    device = torch.device("cuda")
    x = torch.zeros(1, device=device)

    # Warmup
    for _ in range(50):
        x.add_(1.0)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_launches):
        x.add_(1.0)  # each call dispatches a fresh kernel launch
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    kappa_ms = total_ms / n_launches
    return kappa_ms


def measure_sigma_gpu(n_polls: int = 2000):
    """
    Approximates sigma (queue-poll signal cost) as the cost of a
    CUDA-graph replay of the same trivial op, which is the practical
    stand-in for kernel residency available without hand-written CUDA
    (see triton_persistent_agent_kernel.py). True device-side queue
    polling (persistent_kernel.cu) will typically be even cheaper than
    this upper bound.
    """
    import torch

    device = torch.device("cuda")
    x = torch.zeros(1, device=device)

    for _ in range(50):
        x.add_(1.0)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        x.add_(1.0)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_polls):
        g.replay()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    sigma_ms = total_ms / n_polls
    return sigma_ms


def run_gpu(regime_key: str):
    r = REGIMES[regime_key]
    kappa = measure_kappa_gpu()
    sigma = measure_sigma_gpu()

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
                results=results)


def print_table(payload):
    print(f"\nRegime {payload['regime']} ({payload['label']})  "
          f"kappa={payload['kappa_ms']:.4f} ms  sigma={payload['sigma_ms']:.4f} ms")
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
    args = parser.parse_args()

    regimes = ["A", "B"] if args.regime == "both" else [args.regime]
    all_payloads = []

    for reg in regimes:
        if args.mode == "simulate":
            payload = simulate(reg)
        else:
            payload = run_gpu(reg)
        print_table(payload)
        all_payloads.append(payload)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_payloads, f, indent=2)
        print(f"\nWrote results to {args.json_out}")
