# Steering meta-search, round 2

Band 400 to 450 nm, surrogate = fixed seed-0 UV predictor. Gates (not relaxed): validity >= 0.95, uniqueness >= 0.9, novelty >= 0.9, rule pass >= 0.95, internal diversity >= 0.9 x plain, holdout gap <= 0.1. Selection (not gate) margin 0.01.

## Confirmation (fresh samples)

| trial | step | in-band (surr) | in-band (holdout) | gap | valid | uniq | novel | rule | int_div | kl | passes all | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reinvent-403657 | 1200 | 0.725 | 0.713 | 0.012 | 1.000 | 0.922 | 1.000 | 0.989 | 0.809 | 13.62 | YES |  |
| reinvent-8060b0 | 350 | 0.508 | 0.625 | -0.116 | 1.000 | 0.955 | 1.000 | 0.983 | 0.840 | n/a | YES |  |
| reinvent-00c23f | 250 | 0.520 | 0.653 | -0.134 | 1.000 | 0.916 | 1.000 | 0.996 | 0.836 | n/a | YES |  |
| reinvent-4f6fa1 | 200 | 0.468 | 0.236 | 0.232 | 1.000 | 0.918 | 0.999 | 1.000 | 0.844 | n/a | no |  holdout gap |
| reinvent-14d2ac | 300 | 0.439 | 0.490 | -0.051 | 1.000 | 0.939 | 1.000 | 0.984 | 0.852 | n/a | YES |  |
| reinvent-c2ab5e | 100 | 0.428 | 0.229 | 0.199 | 1.000 | 0.921 | 0.999 | 1.000 | 0.855 | n/a | no |  holdout gap |
| reinvent-780401 | 200 | 0.446 | 0.246 | 0.200 | 1.000 | 0.940 | 0.999 | 0.999 | 0.849 | n/a | no |  holdout gap |
| reinvent-500b20 | 300 | 0.433 | 0.220 | 0.213 | 1.000 | 0.923 | 0.998 | 1.000 | 0.838 | n/a | no |  holdout gap |
| plain-7d8b64 | 0 | 0.180 | 0.198 | -0.017 | 1.000 | 1.000 | 0.999 | 0.963 | 0.886 | n/a | YES |  |
| prefix-fbc1c7 | 0 | 0.243 | 0.232 | 0.011 | 1.000 | 1.000 | 0.999 | 0.999 | 0.886 | n/a | YES |  |

Winner: {'trial': 'reinvent-403657', 'step': 1200, 'spec': {'method': 'reinvent', 'family': 'gru_selfies', 'init': 'plain', 'sigma': 40.0, 'lr': 0.0001, 'batch': 128, 'soft_width': 25.0, 'topk_frac': 1.0, 'bucket': 3, 'scaffold_bucket': 0, 'entropy_beta': 0.0, 'ad_threshold': 0.25, 'post_w': 0.0, 'eval_temp': 0.9}}

## Frontier over all evaluated snapshots (n per snapshot as set in config)

Best surrogate in-band fraction at each uniqueness level, among snapshots meeting all other gates and the holdout gap.

| uniqueness >= | trial | step | in-band (surr) | in-band (holdout) | uniq | int_div |
|---|---|---|---|---|---|---|
| 0.7 | reinvent-403657 | 1150 | 0.735 | 0.780 | 0.906 | 0.805 |
| 0.75 | reinvent-403657 | 1150 | 0.735 | 0.780 | 0.906 | 0.805 |
| 0.8 | reinvent-403657 | 1150 | 0.735 | 0.780 | 0.906 | 0.805 |
| 0.85 | reinvent-403657 | 1150 | 0.735 | 0.780 | 0.906 | 0.805 |
| 0.9 | reinvent-403657 | 1150 | 0.735 | 0.780 | 0.906 | 0.805 |
| 0.95 | reinvent-8060b0 | 350 | 0.511 | 0.615 | 0.957 | 0.838 |

## Trial hyper-parameters (confirmed trials)

| trial | hparams |
|---|---|
| reinvent-403657 | ad_threshold=0.25 batch=128 bucket=3 entropy_beta=0.0 eval_temp=0.9 init=plain lr=0.0001 post_w=0.0 scaffold_bucket=0 sigma=40.0 soft_width=25.0 topk_frac=1.0 |
| reinvent-8060b0 | ad_threshold=0.25 batch=256 bucket=0 entropy_beta=2.0 eval_temp=1.0 init=plain lr=0.0001 post_w=0.0 scaffold_bucket=0 sigma=30.0 soft_width=25.0 topk_frac=1.0 |
| reinvent-00c23f | ad_threshold=0.25 batch=128 bucket=3 entropy_beta=1.0 eval_temp=1.0 init=plain lr=0.0002 post_w=0.1 scaffold_bucket=25 sigma=50.0 soft_width=25.0 topk_frac=1.0 |
| reinvent-4f6fa1 | ad_threshold=0.25 batch=256 bucket=0 entropy_beta=1.0 eval_temp=0.9 init=steered lr=0.0001 post_w=0.1 scaffold_bucket=0 sigma=40.0 soft_width=15.0 topk_frac=1.0 |
| reinvent-14d2ac | ad_threshold=0.35 batch=256 bucket=0 entropy_beta=1.0 eval_temp=1.0 init=plain lr=0.0001 post_w=0.0 scaffold_bucket=0 sigma=30.0 soft_width=25.0 topk_frac=1.0 |
| reinvent-c2ab5e | ad_threshold=0.35 batch=256 bucket=0 entropy_beta=1.0 eval_temp=1.0 init=steered lr=0.0002 post_w=0.0 scaffold_bucket=0 sigma=30.0 soft_width=25.0 topk_frac=1.0 |
| reinvent-780401 | ad_threshold=0.35 batch=256 bucket=0 entropy_beta=1.0 eval_temp=1.0 init=steered lr=0.0001 post_w=0.0 scaffold_bucket=0 sigma=30.0 soft_width=25.0 topk_frac=1.0 |
| reinvent-500b20 | ad_threshold=0.35 batch=128 bucket=3 entropy_beta=1.0 eval_temp=0.9 init=steered lr=0.0001 post_w=0.1 scaffold_bucket=25 sigma=30.0 soft_width=15.0 topk_frac=0.5 |
