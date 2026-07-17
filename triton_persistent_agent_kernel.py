"""
triton_persistent_agent_kernel.py — Multi-Trajectory Prototype

Models N concurrent agent trajectories sharing one timeline, using
CUDA-graph replay as a proxy for kernel residency.  Each trajectory
has its own KV-cache state; all share the same model weights.

This is the fast-iteration scaffold for the multi-trajectory design
in design.md Section 2-3.  The real device-level residency (device-side
spin poll) lives in persistent_kernel.cu.

Usage:
  python triton_persistent_agent_kernel.py --n-trajectories 8 --steps 50
"""

import argparse
import json
import time
import math

try:
    import torch
    import triton
    import triton.language as tl
except ImportError as e:
    raise SystemExit(
        "Need torch + triton on a CUDA-capable machine.\n"
        "Install: pip install torch triton --break-system-packages"
    ) from e


# ---------------------------------------------------------------------------
# Tiny transformer block (2 layers, dim=256, 4 heads) matching the CUDA
# kernel's synthetic model so results are cross-comparable.
# ---------------------------------------------------------------------------
def make_model(device: str = "cuda"):
    import torch.nn as nn

    dim = 256
    n_heads = 4
    mlp_ratio = 4

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

    return nn.Sequential(*[TransformerLayer() for _ in range(2)]).to(device).eval()


# ---------------------------------------------------------------------------
# Per-trajectory state
# ---------------------------------------------------------------------------
class TrajectoryState:
    """Holds one trajectory's KV-cache proxy (hidden state) + timing."""

    def __init__(self, dim: int, device: str = "cuda"):
        self.hidden = torch.randn(1, 1, dim, device=device)
        self.steps_done = 0
        self.sigma_us = 0.0       # graph-replay overhead proxy
        self.decode_us = 0.0      # compute-only time
        self.step_times_us = []   # per-step wall clock

    def reset(self):
        self.steps_done = 0
        self.sigma_us = 0.0
        self.decode_us = 0.0
        self.step_times_us = []


# ---------------------------------------------------------------------------
# Host-side queue: array of N slots
# ---------------------------------------------------------------------------
SLOT_EMPTY, SLOT_READY, SLOT_DONE, SLOT_SHUTDOWN = 0, 1, 2, 3


class HostQueue:
    """Simulates the persistent-kernel queue on the host side.

    Each slot holds: state, trajectory_id, and a token embedding vector.
    The 'kernel' (run_loop) polls this array round-robin.
    """

    def __init__(self, n_trajectories: int, dim: int, device: str = "cuda"):
        self.N = n_trajectories
        self.dim = dim
        self.device = device
        self.state = [SLOT_EMPTY] * n_trajectories
        self.trajectory_id = list(range(n_trajectories))
        self.input = [torch.randn(1, 1, dim, device=device)
                      for _ in range(n_trajectories)]
        self.output = [torch.zeros(1, 1, dim, device=device)
                       for _ in range(n_trajectories)]

    def submit(self, idx: int):
        self.state[idx] = SLOT_READY

    def poll_done(self, idx: int, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while self.state[idx] != SLOT_DONE:
            if time.monotonic() > deadline:
                return False
        return True

    def reset_slot(self, idx: int):
        self.state[idx] = SLOT_EMPTY

    def all_done(self) -> bool:
        return all(s == SLOT_DONE for s in self.state)


# ---------------------------------------------------------------------------
# Multi-trajectory persistent kernel loop
# ---------------------------------------------------------------------------
def run_multi_trajectory(
    n_trajectories: int,
    n_steps: int,
    tool_latency_s: float,
    dim: int = 256,
    verbose: bool = False,
) -> dict:
    """Run N trajectories through T steps using a single 'persistent'
    timeline — one model instance, per-trajectory CUDA graphs replayed
    round-robin.

    Returns a dict with per-trajectory and aggregate timing.
    """
    device = torch.device("cuda")
    model = make_model(device)
    queue = HostQueue(n_trajectories, dim, device)

    trajectories = [TrajectoryState(dim, device)
                    for _ in range(n_trajectories)]

    # Pre-capture CUDA graph for each trajectory (graphs are weight-static,
    # input-varying via copy).
    graphs = []
    for t in range(n_trajectories):
        x = trajectories[t].hidden
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _ = model(x)
        graphs.append(g)
    torch.cuda.synchronize()
    if verbose:
        print(f"  Captured {n_trajectories} CUDA graphs")

    # ---- Step loop ------------------------------------------------------
    for step in range(n_steps):
        step_start = time.perf_counter()

        # 1. Submit all N trajectories to the queue
        for t in range(n_trajectories):
            # Refresh input token
            trajectories[t].hidden.copy_(
                torch.randn(1, 1, dim, device=device))
            queue.submit(t)

        # 2. Process round-robin (the 'kernel' poll loop)
        for t in range(n_trajectories):
            poll_start = time.perf_counter()
            # Busy-wait until READY (simulates queue-poll σ)
            while queue.state[t] != SLOT_READY:
                pass
            poll_end = time.perf_counter()

            # Replay the decode graph
            decode_start = time.perf_counter()
            graphs[t].replay()
            torch.cuda.synchronize()
            decode_end = time.perf_counter()

            # Mark done
            queue.state[t] = SLOT_DONE

            # Accumulate timing
            traj = trajectories[t]
            traj.steps_done += 1
            sigma_us = (poll_end - poll_start) * 1e6
            decode_us = (decode_end - decode_start) * 1e6
            traj.sigma_us += sigma_us
            traj.decode_us += decode_us

        step_end = time.perf_counter()
        step_wall_us = (step_end - step_start) * 1e6

        if verbose:
            print(f"  step {step:3d}: wall={step_wall_us:.1f} us  "
                  f"({n_trajectories} trajectories)")

        # 3. Tool-latency sleep
        time.sleep(tool_latency_s)

    # ---- Aggregate ------------------------------------------------------
    total_sigma_us = sum(t.sigma_us for t in trajectories)
    total_decode_us = sum(t.decode_us for t in trajectories)
    total_steps = sum(t.steps_done for t in trajectories)

    # Compute per-trajectory-per-step averages
    per_traj = []
    for t, traj in enumerate(trajectories):
        if traj.steps_done > 0:
            per_traj.append({
                "trajectory": t,
                "steps": traj.steps_done,
                "sigma_us_per_step": round(traj.sigma_us / traj.steps_done, 4),
                "decode_us_per_step": round(traj.decode_us / traj.steps_done, 4),
            })

    return {
        "n_trajectories": n_trajectories,
        "n_steps": n_steps,
        "tool_latency_s": tool_latency_s,
        "dim": dim,
        "total_sigma_us": round(total_sigma_us, 2),
        "total_decode_us": round(total_decode_us, 2),
        "per_trajectory": per_traj,
        "avg_sigma_us_per_traj_step": round(
            total_sigma_us / total_steps, 4) if total_steps > 0 else 0,
        "avg_decode_us_per_traj_step": round(
            total_decode_us / total_steps, 4) if total_steps > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Cost model (analytical, no GPU needed)
# ---------------------------------------------------------------------------
def cost_model(
    n_trajectories: int,
    kappa_us: float = 5.82,
    sigma_us: float = 0.095,
    decode_us: float = 200.0,
    tool_latency_us: float = 500.0,
    n_steps: int = 100,
) -> dict:
    """Analytical multi-trajectory cost model.

    Formulas (from design.md §7):

        T_persistent(N) = kappa/N + sigma + decode + tool_latency
        T_naive(N)      = N * (kappa + decode + tool_latency)

    All times in microseconds, returned in ms.
    """
    kappa_ms = kappa_us / 1000.0
    sigma_ms = sigma_us / 1000.0
    decode_ms = decode_us / 1000.0
    tool_ms = tool_latency_us / 1000.0

    per_step_persistent = kappa_ms / n_trajectories + sigma_ms + decode_ms + tool_ms
    per_step_naive = kappa_ms + decode_ms + tool_ms

    total_persistent = per_step_persistent * n_steps
    total_naive = per_step_naive * n_steps * n_trajectories

    # Fair comparison: both do N × T steps total work
    delta = total_naive - total_persistent
    frac = delta / total_naive if total_naive > 0 else 0.0

    return {
        "n_trajectories": n_trajectories,
        "n_steps": n_steps,
        "kappa_us": kappa_us,
        "sigma_us": sigma_us,
        "decode_us": decode_us,
        "tool_latency_us": tool_latency_us,
        "per_step_persistent_ms": round(per_step_persistent, 4),
        "per_step_naive_ms": round(per_step_naive, 4),
        "total_persistent_ms": round(total_persistent, 2),
        "total_naive_ms": round(total_naive, 2),
        "delta_ms": round(delta, 2),
        "delta_over_naive": round(frac, 6),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trajectories", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--tool-latency-ms", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--mode", choices=["gpu", "simulate"], default="gpu")
    parser.add_argument("--sweep-n", type=str, default=None,
                        help="Comma-separated N values, e.g. '1,2,4,8,16'")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "simulate":
        print("=== Analytical Cost Model ===\n")
        if args.sweep_n:
            ns = [int(x.strip()) for x in args.sweep_n.split(",")]
        else:
            ns = [args.n_trajectories]

        all_results = []
        for n in ns:
            r = cost_model(n)
            all_results.append(r)
            print(f"  N={n:3d}:  "
                  f"persistent={r['per_step_persistent_ms']:.4f} ms/step  "
                  f"naive={r['per_step_naive_ms']:.4f} ms/step  "
                  f"Δ={r['delta_ms']:.2f} ms  "
                  f"saving={r['delta_over_naive']:.4%}")

        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nWrote {args.json_out}")

    else:
        if not torch.cuda.is_available():
            raise SystemExit("No CUDA GPU. Use --mode simulate for CPU-only.")

        if args.sweep_n:
            ns = [int(x.strip()) for x in args.sweep_n.split(",")]
        else:
            ns = [args.n_trajectories]

        print(f"=== Triton Multi-Trajectory Prototype ===\n"
              f"  Tool latency={args.tool_latency_ms} ms  Steps={args.steps}\n")
        all_results = []
        for n in ns:
            print(f"--- N={n} ---")
            result = run_multi_trajectory(
                n_trajectories=n,
                n_steps=args.steps,
                tool_latency_s=args.tool_latency_ms / 1000.0,
                verbose=args.verbose,
            )
            all_results.append(result)
            avg = result["avg_decode_us_per_traj_step"]
            sigma = result["avg_sigma_us_per_traj_step"]
            print(f"  avg decode: {avg:.3f} us/step  "
                  f"avg sigma: {sigma:.4f} us/step\n")

        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"Wrote {args.json_out}")
