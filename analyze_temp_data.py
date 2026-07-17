"""Analyze temperature-logged benchmark data already collected."""
import json, os, statistics

RESULTS_DIR = r'D:\research\benchmarks\results'
ROOT_DIR = r'D:\research'

def fmt_mean_std(vals):
    m = statistics.mean(vals); s = statistics.stdev(vals)
    return m, s

def analyze(label, runs, metric_keys, temp_key='temp_before', clock_key='sm_clock_mhz_before'):
    print(f'\n{"="*70}')
    print(f'{label}')
    print(f'{"="*70}')
    temps_before = [r[temp_key] for r in runs]
    temps_after = [r['temp_after'] for r in runs]
    clocks_before = [r[clock_key] for r in runs]
    print(f'Temperature: {min(temps_before)}-{max(temps_before)}C before, {min(temps_after)}-{max(temps_after)}C after')
    print(f'SM clock:    {min(clocks_before)}-{max(clocks_before)} MHz before')
    print(f'  runs at base=1605MHz: {sum(1 for c in clocks_before if c == 1605)}')
    print(f'  runs boosted (>1605): {sum(1 for c in clocks_before if c > 1605)}')
    if any(c < 1605 for c in clocks_before):
        print(f'  runs throttled (<1605): {sum(1 for c in clocks_before if c < 1605)}')
    print()

    for mk, label in metric_keys:
        vals = [r[mk] for r in runs]
        m, s = fmt_mean_std(vals)
        unit = 'us' if m < 1 else 'ms'
        scale = 1000 if m < 1 else 1
        print(f'  {label}: {m*scale:.4f} +/- {s*scale:.4f} {unit}  (CV={s/m*100:.1f}%)')
        print(f'    range: [{min(vals)*scale:.4f}, {max(vals)*scale:.4f}] {unit}')

        # Split by clock: boosted vs base
        base_vals = [v for v, c in zip(vals, clocks_before) if c == 1605]
        boost_vals = [v for v, c in zip(vals, clocks_before) if c > 1605]
        if base_vals and boost_vals:
            bm = statistics.mean(base_vals)*scale; bv = statistics.mean(boost_vals)*scale
            print(f'    mean at base 1605MHz: {bm:.4f} {unit}')
            print(f'    mean when boosted:    {bv:.4f} {unit}')
            print(f'    boosted/base ratio:   {bv/bm:.4f}')

        # Sort by clock and compare extremes
        sorted_by_clock = sorted(zip(clocks_before, vals), key=lambda x: x[0])
        low3 = statistics.mean([v for _, v in sorted_by_clock[:3]])*scale
        high3 = statistics.mean([v for _, v in sorted_by_clock[-3:]])*scale
        print(f'    mean of 3 lowest-clock runs:  {low3:.4f} {unit}')
        print(f'    mean of 3 highest-clock runs: {high3:.4f} {unit}')

        # Correlation
        if len(set(clocks_before)) > 1:
            corr = statistics.correlation(clocks_before, vals)
            print(f'    correlation with clock: {corr:+.4f}')
        if len(set(temps_before)) > 1:
            corr = statistics.correlation(temps_before, vals)
            print(f'    correlation with temp:  {corr:+.4f}')
        print()


# ===== PART 1: 48-op chain =====
gpu_runs = []
for i in range(10):
    path = os.path.join(RESULTS_DIR, f'temp_48op_run_{i}.json')
    if os.path.exists(path):
        with open(path) as f:
            payload = json.load(f)
        gpu_runs.append({
            'run': i,
            'temp_before': None, 'temp_after': None,
            'sm_clock_mhz_before': None, 'sm_clock_mhz_after': None,
            'kappa_ms_A': payload[0]['kappa_ms'],
            'sigma_ms_A': payload[0]['sigma_ms'],
            'kappa_ms_B': payload[1]['kappa_ms'],
            'sigma_ms_B': payload[1]['sigma_ms'],
        })

# Check if temp data exists (from the logging script)
temp_data = {}
for i in range(10):
    tpath = os.path.join(ROOT_DIR, 'gpu_temp_logged_results.json')
    if os.path.exists(tpath):
        with open(tpath) as f:
            td = json.load(f)
        for r in td['48op_chain_runs']:
            if r['run'] == i:
                gpu_runs[i]['temp_before'] = r['temp_before']
                gpu_runs[i]['temp_after'] = r['temp_after']
                gpu_runs[i]['sm_clock_mhz_before'] = r['sm_clock_mhz_before']
                gpu_runs[i]['sm_clock_mhz_after'] = r['sm_clock_mhz_after']

# If temp data not logged yet, run the logging
if gpu_runs[0]['temp_before'] is None:
    print("No temperature data found. Run run_with_temp_logging.py first.")
else:
    analyze('48-OP CHAIN (10 runs)', gpu_runs,
            [('kappa_ms_A', 'kappa regime A'), ('sigma_ms_A', 'sigma regime A'),
             ('kappa_ms_B', 'kappa regime B'), ('sigma_ms_B', 'sigma regime B')])


# ===== PART 2: persistent_kernel.exe =====
kernel_runs = []
for i in range(10):
    path = os.path.join(RESULTS_DIR, f'temp_kernel_run_{i}.json')
    if os.path.exists(path):
        with open(path) as f:
            stats = json.load(f)
        kernel_runs.append({
            'run': i,
            'temp_before': None, 'temp_after': None,
            'sm_clock_mhz_before': None, 'sm_clock_mhz_after': None,
            'sigma_ms': stats['sigma_ms'],
            'kappa_ms': stats['kappa_ms'],
            'compute_ms': stats['clock64_compute_ms'],
        })

if os.path.exists(tpath):
    with open(tpath) as f:
        td = json.load(f)
    for r in td['persistent_kernel_runs']:
        for kr in kernel_runs:
            if kr['run'] == r['run']:
                kr['temp_before'] = r['temp_before']
                kr['temp_after'] = r['temp_after']
                kr['sm_clock_mhz_before'] = r['sm_clock_mhz_before']
                kr['sm_clock_mhz_after'] = r['sm_clock_mhz_after']

if kernel_runs and kernel_runs[0]['temp_before'] is not None:
    analyze('PERSISTENT KERNEL (10 runs)', kernel_runs,
            [('sigma_ms', 'sigma (clock64)'), ('kappa_ms', 'kappa_fused (derived)'),
             ('compute_ms', 'clock64 compute')])
else:
    print('\nPersistent kernel data not available. Re-run run_with_temp_logging.py.')
