import json, os, statistics

results_dir = r'D:\research\benchmarks\results'
runs = []
for i in range(10):
    path = os.path.join(results_dir, f'run_{i}.json')
    with open(path) as f:
        d = json.load(f)
    runs.append(d)

def mean_std(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s

print('=== Summary across 10 runs ===')
print()

fields = [
    ('sigma_ms', 'sigma'),
    ('kappa_ms', 'kappa_fused'),
    ('naive_ms_per_step', 'naive_per_step'),
    ('cuda_event_compute_ms', 'cuda_event_compute'),
    ('clock64_compute_ms', 'clock64_compute'),
    ('persistent_host_ms_per_step', 'persistent_host_per_step'),
    ('n_timed_steps', 'n_timed_steps'),
    ('sigma_cycles_total', 'sigma_cycles_total'),
    ('compute_cycles_total', 'compute_cycles_total'),
]

for key, label in fields:
    vals = [r[key] for r in runs]
    m, s = mean_std(vals)
    if key.endswith('_steps') or key.endswith('_total'):
        print(f'  {label:30s} = {int(round(m))} (all identical)' if s == 0 else f'  {label:30s} = {m:.1f} +/- {s:.1f}')
    elif m < 0.001:
        print(f'  {label:30s} = {m*1e6:.4f} +/- {s*1e6:.4f} ns')
    elif m < 1.0:
        print(f'  {label:30s} = {m*1e3:.4f} +/- {s*1e3:.4f} us')
    else:
        print(f'  {label:30s} = {m:.4f} +/- {s:.4f} ms')

print()

sigma = runs[0]['sigma_ms']
kappa_vals = [r['kappa_ms'] for r in runs]
mk, sk = statistics.mean(kappa_vals), statistics.stdev(kappa_vals)

print(f'  kappa_fused (mean +/- std):  {mk*1000:.3f} +/- {sk*1000:.3f} us')
print(f'  kappa_fused range:           [{min(kappa_vals)*1000:.3f}, {max(kappa_vals)*1000:.3f}] us')
print(f'  sigma (all 10 runs):         {sigma*1000:.4f} us')
print(f'  kappa/sigma ratio:           {mk / sigma:.0f}x')

print()
print('=== Per-run raw ===')
print(f'  {"run":>3} {"sigma(us)":>10} {"kappa(us)":>11} {"naive(ms)":>10} {"event(ms)":>10} {"clock64(ms)":>11}')
for i, r in enumerate(runs):
    print(f'  {i:>3} {r["sigma_ms"]*1000:>10.4f} {r["kappa_ms"]*1000:>11.3f} {r["naive_ms_per_step"]:>10.5f} {r["cuda_event_compute_ms"]:>10.5f} {r["clock64_compute_ms"]:>11.5f}')

# Save aggregated results
agg = {
    'n_runs': 10,
    'n_steps_per_run': runs[0]['n_timed_steps'],
    'device': runs[0]['device'],
    'sm': runs[0]['sm'],
    'clock_rate_khz': runs[0]['clock_rate_khz'],
    'model': runs[0]['model'],
    'sigma_ms': {
        'mean': sigma,
        'std': 0.0,
        'unit': 'ms',
        'all_identical': True,
        'cycles_per_step': runs[0]['sigma_cycles_total'] / runs[0]['n_timed_steps']
    },
    'kappa_fused_ms': {
        'mean': round(mk, 8),
        'std': round(sk, 8),
        'unit': 'ms',
        'min': round(min(kappa_vals), 8),
        'max': round(max(kappa_vals), 8),
    },
    'residency_benefit_beyond_fusion_us': (mk - sigma) * 1000,
    'naive_ms_per_step': {
        'mean': round(statistics.mean([r['naive_ms_per_step'] for r in runs]), 8),
        'std': round(statistics.stdev([r['naive_ms_per_step'] for r in runs]), 8),
    },
    'cuda_event_compute_ms': {
        'mean': round(statistics.mean([r['cuda_event_compute_ms'] for r in runs]), 8),
        'std': round(statistics.stdev([r['cuda_event_compute_ms'] for r in runs]), 8),
    },
    'clock64_compute_ms': {
        'mean': round(statistics.mean([r['clock64_compute_ms'] for r in runs]), 8),
        'std': round(statistics.stdev([r['clock64_compute_ms'] for r in runs]), 8),
    },
    'per_run_results': [
        {
            'run': i,
            'sigma_ms': r['sigma_ms'],
            'kappa_ms': r['kappa_ms'],
            'naive_ms_per_step': r['naive_ms_per_step'],
            'cuda_event_compute_ms': r['cuda_event_compute_ms'],
            'clock64_compute_ms': r['clock64_compute_ms'],
        }
        for i, r in enumerate(runs)
    ],
}

with open(os.path.join(results_dir, '..', '..', 'gpu_results_device_sigma_repeated.json'), 'w') as f:
    json.dump(agg, f, indent=2)
print(f'\nSaved to gpu_results_device_sigma_repeated.json')
