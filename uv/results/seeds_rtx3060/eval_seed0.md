# UV-Vis steered vs plain generation

Target band: 400 to 450 nm. Samples per arm: 20000. Device: cuda.

| metric | plain | steered |
|---|---|---|
| validity | 1.0 | 1.0 |
| uniqueness | 1.0 | 0.9998 |
| novelty | 0.9981 | 0.9989 |
| rule pass rate | 0.9614 | 0.9986 |
| in-band fraction | 0.1864 | 0.2523 |
| mean predicted abs (nm) | 366.28 | 381.22 |
| samples | 20000 | 20000 |
| valid | 20000 | 20000 |
| unique | 19999 | 19996 |
| scored | 19999 | 19996 |

## Regressor held-out performance

| metric | value |
|---|---|
| MAE (nm) | 24.04 |
| RMSE (nm) | 38.19 |
| R2 | 0.8734 |
| train structures | 5203 |
| test structures | 643 |

## Steering effect (bootstrap)

Difference in in-band fraction, steered minus plain: 0.0658.
95 percent bootstrap CI: [0.0582, 0.0738] from 2000 resamples.
Resamples with a positive difference: 1.0.
CI excludes zero: True.
