# UV-Vis steered vs plain generation

Target band: 400 to 450 nm. Samples per arm: 20000. Device: cuda.

| metric | plain | steered |
|---|---|---|
| validity | 1.0 | 1.0 |
| uniqueness | 1.0 | 1.0 |
| novelty | 0.9978 | 0.9984 |
| rule pass rate | 0.9634 | 0.9995 |
| in-band fraction | 0.1776 | 0.2559 |
| mean predicted abs (nm) | 363.59 | 381.6 |
| samples | 20000 | 20000 |
| valid | 20000 | 20000 |
| unique | 19999 | 19999 |
| scored | 19999 | 19999 |

## Regressor held-out performance

| metric | value |
|---|---|
| MAE (nm) | 24.04 |
| RMSE (nm) | 38.19 |
| R2 | 0.8734 |
| train structures | 5203 |
| test structures | 643 |

## Steering effect (bootstrap)

Difference in in-band fraction, steered minus plain: 0.0783.
95 percent bootstrap CI: [0.0702, 0.0861] from 2000 resamples.
Resamples with a positive difference: 1.0.
CI excludes zero: True.
