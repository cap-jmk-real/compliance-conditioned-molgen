# Steering meta-search results

Target band 400 to 450 nm (surrogate = fixed seed-0 UV predictor). Rungs (steps:n_eval): [(10, 300), (30, 500)].
Constraints: validity >= 0.95, uniqueness >= 0.9, novelty >= 0.9, rule pass >= 0.95, internal diversity >= 0.9 x plain anchor; holdout gap (surrogate minus holdout in-band) <= 0.1.

Holdout predictor: morgan_count_r3_1024+maccs+conj_descriptors_extratrees, test MAE 18.5 nm, R2 0.928 (own split; same database as the surrogate).

## Confirmation stage (fresh samples)

| trial | method | hparams | in-band (surr) | in-band (holdout) | gap | valid | uniq | novel | rule | int_div | kl nats | feasible | holdout ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain-7d8b64 | plain |  | 0.208 | 0.182 | 0.026 | 1.000 | 1.000 | 0.998 | 0.964 | 0.885 | n/a | yes | yes |
| prefix-fbc1c7 | prefix |  | 0.269 | 0.225 | 0.044 | 1.000 | 1.000 | 1.000 | 0.999 | 0.886 | n/a | yes | yes |
| reinvent-6313d7 | reinvent | batch=128 bucket=0 init=plain lr=0.0003 sigma=10.0 soft_width=25.0 topk_frac=1.0 | 0.312 | 0.369 | -0.057 | 1.000 | 1.000 | 0.999 | 0.974 | 0.885 | 2.74 | yes | yes |
| reinvent-9020ab | reinvent | batch=64 bucket=0 init=steered lr=0.0003 sigma=10.0 soft_width=0.0 topk_frac=1.0 | 0.281 | 0.238 | 0.043 | 1.000 | 1.000 | 0.999 | 0.999 | 0.884 | 2.48 | yes | yes |

Winner (best feasible, holdout-ok, by surrogate in-band): reinvent-6313d7

Pilot-to-final rank correlation (Spearman, rung0 vs rung1, n=5): 0.200

## rung0

| trial | method | steps | n_eval | in-band | valid | uniq | novel | rule | int_div |
|---|---|---|---|---|---|---|---|---|---|
| reinvent-6313d7 | reinvent | 10 | 300 | 0.297 | 1.000 | 1.000 | 1.000 | 0.977 | 0.883 |
| reinvent-99e794 | reinvent | 10 | 300 | 0.273 | 1.000 | 1.000 | 0.997 | 0.960 | 0.885 |
| guidance-23286f | guidance | 0 | 300 | 0.260 | 1.000 | 0.910 | 1.000 | 1.000 | 0.878 |
| reinvent-9020ab | reinvent | 10 | 300 | 0.260 | 1.000 | 1.000 | 1.000 | 1.000 | 0.890 |
| selftrain-639dcd | selftrain | 10 | 300 | 0.253 | 1.000 | 1.000 | 0.997 | 0.997 | 0.890 |
| guidance-7d16f4 | guidance | 0 | 300 | 0.243 | 1.000 | 1.000 | 0.993 | 1.000 | 0.890 |
| prefix-fbc1c7 | prefix | 0 | 300 | 0.240 | 1.000 | 1.000 | 1.000 | 0.997 | 0.887 |
| guidance-ecb358 | guidance | 0 | 300 | 0.230 | 1.000 | 0.900 | 1.000 | 1.000 | 0.893 |
| plain-7d8b64 | plain | 0 | 300 | 0.227 | 1.000 | 1.000 | 1.000 | 0.943 | 0.886 |
| selftrain-91b363 | selftrain | 10 | 300 | 0.193 | 1.000 | 1.000 | 1.000 | 0.963 | 0.888 |
| guidance-f29160 | guidance | 0 | 300 | 0.000 | 1.000 | 0.010 | 1.000 | 1.000 | 0.854 |

## rung1

| trial | method | steps | n_eval | in-band | valid | uniq | novel | rule | int_div |
|---|---|---|---|---|---|---|---|---|---|
| reinvent-6313d7 | reinvent | 30 | 500 | 0.322 | 1.000 | 1.000 | 1.000 | 0.972 | 0.882 |
| reinvent-9020ab | reinvent | 30 | 500 | 0.302 | 1.000 | 1.000 | 0.998 | 1.000 | 0.884 |
| selftrain-639dcd | selftrain | 30 | 500 | 0.300 | 1.000 | 1.000 | 1.000 | 1.000 | 0.887 |
| reinvent-99e794 | reinvent | 30 | 500 | 0.270 | 1.000 | 1.000 | 0.998 | 0.966 | 0.881 |
| prefix-fbc1c7 | prefix | 0 | 500 | 0.264 | 1.000 | 1.000 | 1.000 | 1.000 | 0.883 |
| guidance-23286f | guidance | 0 | 500 | 0.239 | 1.000 | 0.886 | 1.000 | 0.993 | 0.885 |
| plain-7d8b64 | plain | 0 | 500 | 0.208 | 1.000 | 1.000 | 0.998 | 0.960 | 0.886 |

## Trial hyper-parameters

| trial | hparams |
|---|---|
| plain-7d8b64 | plain  |
| prefix-fbc1c7 | prefix  |
| guidance-7d16f4 | guidance kind=plain temperature=1.0 w=0.25 |
| guidance-23286f | guidance kind=bandout temperature=0.9 w=3.0 |
| guidance-f29160 | guidance kind=plain temperature=1.0 w=2.0 |
| guidance-ecb358 | guidance kind=bandout temperature=1.0 w=3.0 |
| reinvent-9020ab | reinvent batch=64 bucket=0 init=steered lr=0.0003 sigma=10.0 soft_width=0.0 topk_frac=1.0 |
| reinvent-6313d7 | reinvent batch=128 bucket=0 init=plain lr=0.0003 sigma=10.0 soft_width=25.0 topk_frac=1.0 |
| reinvent-99e794 | reinvent batch=128 bucket=0 init=plain lr=0.0001 sigma=30.0 soft_width=25.0 topk_frac=1.0 |
| selftrain-91b363 | selftrain batch=128 gen_per_round=1000 init=plain lr=0.0001 replay_frac=0.75 steps_per_round=50 |
| selftrain-639dcd | selftrain batch=128 gen_per_round=2000 init=steered lr=0.0003 replay_frac=0.25 steps_per_round=25 |
