# UV-Vis steered vs plain generation

Target band: 400 to 450 nm. Samples per arm: 20000. Device: cuda.

| metric | plain | steered |
|---|---|---|
| validity | 1.0 | 1.0 |
| uniqueness | 0.9999 | 0.9998 |
| novelty | 0.9989 | 0.999 |
| rule pass rate | 0.9635 | 0.9991 |
| in-band fraction | 0.1867 | 0.2371 |
| mean predicted abs (nm) | 365.69 | 376.92 |
| samples | 20000 | 20000 |
| valid | 20000 | 20000 |
| unique | 19997 | 19995 |
| scored | 19997 | 19995 |

## Regressor held-out performance

| metric | value |
|---|---|
| MAE (nm) | 24.04 |
| RMSE (nm) | 38.19 |
| R2 | 0.8734 |
| train structures | 5203 |
| test structures | 643 |

## Steering effect (bootstrap)

Difference in in-band fraction, steered minus plain: 0.0504.
95 percent bootstrap CI: [0.0425, 0.0582] from 2000 resamples.
Resamples with a positive difference: 1.0.
CI excludes zero: True.
