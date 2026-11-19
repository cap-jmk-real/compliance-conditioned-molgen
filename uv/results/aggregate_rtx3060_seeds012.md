# Multi-seed aggregation (3 seeds, band 400-450 nm)

Sources: uv/results/seeds_rtx3060/eval_seed0.json, uv/results/seeds_rtx3060/eval_seed1.json, uv/results/seeds_rtx3060/eval_seed2.json

## Per-arm metrics (mean +/- sd across seeds)

| metric | plain | steered |
|---|---|---|
| validity | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| uniqueness | 1.0000 +/- 0.0001 | 0.9999 +/- 0.0001 |
| novelty | 0.9979 +/- 0.0002 | 0.9986 +/- 0.0003 |
| structural-screen pass rate | 0.9630 +/- 0.0015 | 0.9992 +/- 0.0005 |
| in-band fraction | 0.1847 +/- 0.0064 | 0.2523 +/- 0.0037 |
| mean predicted abs (nm) | 365.6367 +/- 1.8127 | 381.4067 +/- 0.1901 |

## Steering effect (steered minus plain in-band fraction)

| seed | plain | steered | diff |
|---|---|---|---|
| 0 | 0.1864 | 0.2523 | +0.0659 |
| 1 | 0.1901 | 0.2486 | +0.0585 |
| 2 | 0.1776 | 0.2559 | +0.0783 |

- mean diff across 3 seeds: +0.0676 (sd 0.0100)
- standardised effect d_z = mean/sd = 6.75
- paired t = 11.697, df = 2, two-sided p = 0.0072
- seed-level 95% t-interval on the mean diff (t_crit 4.3027): [+0.0427, +0.0924]
- CAUTION: n=3 is very small; this paired test is badly underpowered. Read it as a direction-and-size check, not a significance claim. Here the effect is positive in every seed above.

## Per-seed within-run bootstrap CI (sampling at fixed weights)

| seed | diff point | CI95 low | CI95 high |
|---|---|---|---|
| 0 | 0.0658 | 0.0582 | 0.0738 |
| 1 | 0.0585 | 0.0506 | 0.0664 |
| 2 | 0.0783 | 0.0702 | 0.0861 |


wrote uv/results/aggregate_rtx3060_seeds012.json
