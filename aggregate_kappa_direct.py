"""Aggregate N=10 independent runs of kappa_fused_direct measurement."""
import json, os, sys, math

run_files = [f"kappa_direct_run_{i}.json" for i in range(10)]
runs = []

for f in run_files:
    if not os.path.exists(f):
        print(f"WARNING: {f} not found, skipping")
        continue
    with open(f) as fh:
        runs.append(json.load(fh))

if len(runs) == 0:
    print("ERROR: no run files found")
    sys.exit(1)

n = len(runs)

def mean_std(arr):
    m = sum(arr) / len(arr)
    if len(arr) > 1:
        v = sum((x - m) ** 2 for x in arr) / (len(arr) - 1)
    else:
        v = 0.0
    return m, math.sqrt(v)

# Per-run data
real_ps = [r["real_per_step_ms"] for r in runs]
noop_ps = [r["noop_per_step_ms"] for r in runs]
kf      = [r["kappa_fused_ms"] for r in runs]
cc      = [r["compute_cuda_event_ms"] for r in runs]
c64_cyc = [r["clock64_compute_cycles_per_step"] for r in runs]
c64_ms  = [r["clock64_compute_ms_at_base_clock"] for r in runs]

real_md = [r["per_launch_real_ms"]["median"] for r in runs]
noop_md = [r["per_launch_noop_ms"]["median"] for r in runs]

kf_mean, kf_std = mean_std(kf)

result = {
    "n_runs": n,
    "n_loop_launches": runs[0]["n_loop_launches"],
    "n_per_launch_real": runs[0]["n_per_launch_real"],
    "n_per_launch_noop": runs[0]["n_per_launch_noop"],
    "device": runs[0]["device"],
    "sm": runs[0]["sm"],
    "clock_rate_khz": runs[0]["clock_rate_khz"],
    "model": runs[0]["model"],
    "real_per_step_ms": dict(zip(["mean","std","min","max","unit"],
                                 [*mean_std(real_ps), min(real_ps), max(real_ps), "ms"])),
    "noop_per_step_ms": dict(zip(["mean","std","min","max","unit"],
                                 [*mean_std(noop_ps), min(noop_ps), max(noop_ps), "ms"])),
    "kappa_fused_ms": {
        "mean": kf_mean, "std": kf_std,
        "min": min(kf), "max": max(kf),
        "unit": "ms",
        "measurement": "noop_loop_single_event_pair_div_N",
    },
    "kappa_fused_us": kf_mean * 1000.0,
    "compute_from_real_minus_kappa_ms": dict(zip(["mean","std","unit"],
                                                  [*mean_std(cc), "ms"])),
    "clock64_compute_ms_at_base": dict(zip(["mean","std","unit"],
                                            [*mean_std(c64_ms), "ms"])),
    "clock64_compute_cycles_per_step": dict(zip(["mean","std","unit"],
                                                  [*mean_std(c64_cyc), "cycles"])),
    "per_launch_real_median_ms": dict(zip(["mean","std","min","max","unit"],
                                           [*mean_std(real_md), min(real_md), max(real_md), "ms"])),
    "per_launch_noop_median_ms": dict(zip(["mean","std","min","max","unit"],
                                           [*mean_std(noop_md), min(noop_md), max(noop_md), "ms"])),
    "per_run_results": [
        {"run": i,
         "real_per_step_ms": real_ps[i],
         "noop_per_step_ms": noop_ps[i],
         "kappa_fused_ms": kf[i],
         "compute_cuda_event_ms": cc[i],
         "clock64_cycles_per_step": c64_cyc[i],
         "clock64_compute_ms_at_base": c64_ms[i],
         "real_per_launch_median_ms": real_md[i],
         "noop_per_launch_median_ms": noop_md[i]}
        for i in range(n)
    ],
}

with open("kappa_fused_direct.json", "w") as f:
    json.dump(result, f, indent=2)

noop_md_mean, noop_md_std = mean_std(noop_md)

print(f"Aggregated {n} runs -> kappa_fused_direct.json")
print()
print(f"  kappa_fused (loop/N):        {kf_mean*1000:.3f} +- {kf_std*1000:.3f} us")
print(f"  range:                       {min(kf)*1000:.3f} - {max(kf)*1000:.3f} us")
print(f"  std/mean:                    {kf_std / kf_mean:.4f}")
print(f"  noop per-launch median:      {noop_md_mean*1000:.3f} +- {noop_md_std*1000:.3f} us")
print(f"  range:                       {min(noop_md)*1000:.3f} - {max(noop_md)*1000:.3f} us")
print(f"  std/mean:                    {noop_md_std / noop_md_mean:.4f}")
print(f"  real per_step:               {mean_std(real_ps)[0]*1000:.3f} +- {mean_std(real_ps)[1]*1000:.3f} us")
print(f"  compute (CUDA event):        {mean_std(cc)[0]*1000:.3f} +- {mean_std(cc)[1]*1000:.3f} us")
