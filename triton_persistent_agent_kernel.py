"""
triton_persistent_agent_kernel.py

Higher-level Triton prototype of the same idea as persistent_kernel.cu:
a single resident kernel invocation that services multiple agent-loop
steps via a device-side queue, instead of one Triton kernel launch per
step. Useful for fast iteration before dropping to raw CUDA.

Requires: pip install triton torch --break-system-packages
(needs a CUDA-capable GPU; will raise a clear error otherwise — use
benchmark.py's --mode simulate on machines without a GPU).

This is a SCAFFOLD, matching paper.tex Section 4. The `decode_segment`
Triton kernel is a placeholder elementwise op; swap in a real fused
attention/MLP block for production use.
"""

import time
import argparse

try:
    import torch
    import triton
    import triton.language as tl
except ImportError as e:
    raise SystemExit(
        "This script needs torch + triton on a CUDA-capable machine.\n"
        "Install with: pip install torch triton --break-system-packages\n"
        "On a machine without a GPU, use benchmark.py --mode simulate instead."
    ) from e


SLOT_EMPTY, SLOT_READY, SLOT_DONE, SLOT_SHUTDOWN = 0, 1, 2, 3


@triton.jit
def decode_segment_kernel(
    input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    """Placeholder decode segment: replace with real fused forward pass.

    In a true megakernel design (MPK-style) this would be one kernel
    fused across the entire model's layers, resident for the whole
    session. Triton doesn't natively expose the same persistent-kernel
    residency primitives as raw CUDA (device-side spin loops across
    host round-trips), so this file demonstrates the *per-step compute*
    piece and pairs it with a host-side loop that avoids unnecessary
    framework-level relaunch overhead (module re-tracing, CUDA graph
    capture reuse) rather than a device-resident poll loop. For true
    kernel residency across host round-trips, use persistent_kernel.cu.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x, mask=mask)  # placeholder op


def run_agent_loop_with_cuda_graph(T: int, tool_latency_s: float, dim: int = 4096):
    """
    Demonstrates the framework-level analogue of kernel residency:
    capturing the decode segment once as a CUDA graph and *replaying*
    it every step, instead of re-dispatching a fresh Triton/CUDA launch
    (with associated Python/driver dispatch overhead) each time.

    This is the practical, GPU-available counterpart to the raw
    persistent-kernel design in persistent_kernel.cu: CUDA graphs don't
    give you device-side residency across host round-trips, but they
    do eliminate most per-step *launch and dispatch* overhead, which is
    the dominant recoverable term (kappa) in the paper's cost model.
    """
    device = torch.device("cuda")
    x = torch.randn(dim, device=device)
    y = torch.empty_like(x)

    grid = (triton.cdiv(dim, 1024),)

    # Warmup (required before graph capture).
    for _ in range(3):
        decode_segment_kernel[grid](x, y, dim, BLOCK_SIZE=1024)
    torch.cuda.synchronize()

    # Capture the decode segment as a CUDA graph: one "launch"
    # regardless of how many times we replay it.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        decode_segment_kernel[grid](x, y, dim, BLOCK_SIZE=1024)

    step_times = []
    for t in range(T):
        step_start = time.perf_counter()

        # "Re-encode" input in place (stand-in for tool-result -> tokens).
        x.copy_(torch.randn(dim, device=device))

        g.replay()  # <-- no new kernel launch/dispatch, just a graph replay
        torch.cuda.synchronize()

        step_end = time.perf_counter()
        step_times.append(step_end - step_start)

        # Simulate the opaque tool-call latency (ell_t in the cost model).
        time.sleep(tool_latency_s)

    return step_times


def run_agent_loop_naive(T: int, tool_latency_s: float, dim: int = 4096):
    """Naive baseline: a fresh kernel dispatch every step, no graph reuse."""
    device = torch.device("cuda")
    x = torch.randn(dim, device=device)
    y = torch.empty_like(x)
    grid = (triton.cdiv(dim, 1024),)

    step_times = []
    for t in range(T):
        step_start = time.perf_counter()
        x.copy_(torch.randn(dim, device=device))
        decode_segment_kernel[grid](x, y, dim, BLOCK_SIZE=1024)
        torch.cuda.synchronize()
        step_end = time.perf_counter()
        step_times.append(step_end - step_start)
        time.sleep(tool_latency_s)

    return step_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--tool-latency-ms", type=float, default=20.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA GPU detected. Run this on your RTX-class or A100/H100 "
            "machine, or use benchmark.py --mode simulate here instead."
        )

    tool_latency_s = args.tool_latency_ms / 1000.0

    naive_times = run_agent_loop_naive(args.steps, tool_latency_s)
    graph_times = run_agent_loop_with_cuda_graph(args.steps, tool_latency_s)

    print(f"Naive avg step time:        {sum(naive_times)/len(naive_times)*1e3:.4f} ms")
    print(f"CUDA-graph-replay avg step: {sum(graph_times)/len(graph_times)*1e3:.4f} ms")
    print(f"Recovered overhead/step:    "
          f"{(sum(naive_times)-sum(graph_times))/len(naive_times)*1e3:.4f} ms")
