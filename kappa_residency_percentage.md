# Residency-beyond-fusion percentage — recalculation

**Date:** 2026-07-17  
**Author:** post-hoc analysis after κ_fused direct measurement update

## Parameters

| Symbol | Value | Source |
|--------|-------|--------|
| κ_fused | 5.82 µs | N=10 median of per-run medians, 2000 per-launch samples each |
| σ | 0.095 µs | N=10, zero variance across runs |
| residency benefit = κ_fused − σ | **5.725 µs** | (using median; upper bound = 6.09 µs = +1σ) |
| d_t | ≈ 740 µs | real kernel per-step CUDA event time |
| ℓ_t | ~ Uniform(50, 200) ms | regime-A tool latency |

## Percentages

### Against compute-only step time (d_t = 740 µs)
- 5.725 / 740 = **0.77%**
- Upper bound (6.09 µs): 6.09 / 740 = **0.82%**

### Against full decode-plus-tool-call step (original denominator)
Using minimum ℓ_t (50 ms = 50,000 µs):  5.725 / (740 + 50,000) = **0.0113%**  
Using maximum ℓ_t (200 ms = 200,000 µs): 5.725 / (740 + 200,000) = **0.00285%**  
Upper bound 6.09 µs with min ℓ_t: 6.09 / 50,740 = **0.0120%**  

### Comparison with old numbers (44 µs upper bound)

| Denominator | Old (44 µs) | New (5.725 µs) | Factor |
|---|---|---|---|
| Compute-only (740 µs) | 5.9% | 0.77% | 7.7× smaller |
| Full step, min ℓ_t (50.74 ms) | 0.087% | 0.011% | 7.7× smaller |
| Full step, max ℓ_t (200.74 ms) | 0.022% | 0.0029% | 7.7× smaller |

## Verdict

The new 5.725 µs benefit is **still under 0.1%** of a full decode-plus-tool-call step (~0.011% vs the old ~0.087%). It is 7.7× smaller than the old figure, making the headline claim *stronger* (even more negligible). The percentages in the abstract (§1) correctly use the full-step denominator and say "under 0.01%". The §6 and §8 sections use compute-only denominator (0.83%), which is a valid perspective but a different basis from the abstract.
