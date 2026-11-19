# Steering meta-search, round 2

Band 400 to 450 nm, surrogate = fixed seed-0 UV predictor. Gates (not relaxed): validity >= 0.95, uniqueness >= 0.9, novelty >= 0.9, rule pass >= 0.95, internal diversity >= 0.9 x plain, holdout gap <= 0.1. Selection (not gate) margin 0.01.

## Confirmation (fresh samples)

| trial | step | in-band (surr) | in-band (holdout) | gap | valid | uniq | novel | rule | int_div | kl | passes all | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reinvent-be70a9 | 1150 | 0.620 | 0.190 | 0.430 | 1.000 | 0.964 | 1.000 | 1.000 | 0.836 | 7.20 | no |  holdout gap |
| reinvent-8c31d6 | 1200 | 0.622 | 0.145 | 0.477 | 1.000 | 0.956 | 0.999 | 0.999 | 0.816 | 7.17 | no |  holdout gap |
| reinvent-a29221 | 1150 | 0.588 | 0.526 | 0.061 | 1.000 | 0.985 | 0.999 | 1.000 | 0.808 | 6.26 | YES |  |
| reinvent-0b32e1 | 1200 | 0.587 | 0.329 | 0.258 | 1.000 | 0.917 | 1.000 | 0.969 | 0.820 | 12.01 | no |  holdout gap |
| reinvent-4bafab | 400 | 0.531 | 0.561 | -0.030 | 1.000 | 0.939 | 1.000 | 0.999 | 0.843 | n/a | YES |  |
| reinvent-5fbebd | 700 | 0.513 | 0.364 | 0.149 | 1.000 | 0.924 | 0.999 | 1.000 | 0.826 | n/a | no |  holdout gap |
| reinvent-947193 | 400 | 0.514 | 0.549 | -0.035 | 1.000 | 0.922 | 1.000 | 0.997 | 0.838 | n/a | YES |  |
| reinvent-583a2d | 700 | 0.512 | 0.482 | 0.030 | 1.000 | 0.974 | 0.999 | 1.000 | 0.824 | n/a | YES |  |
| plain-7d8b64 | 0 | 0.186 | 0.191 | -0.006 | 1.000 | 1.000 | 0.999 | 0.963 | 0.885 | n/a | YES |  |
| prefix-fbc1c7 | 0 | 0.256 | 0.231 | 0.025 | 1.000 | 1.000 | 0.999 | 0.999 | 0.885 | n/a | YES |  |

Winner: {'trial': 'reinvent-a29221', 'step': 1150, 'spec': {'method': 'reinvent', 'family': 'gru_selfies', 'init': 'steered', 'sigma': 20.0, 'lr': 0.0001, 'batch': 256, 'soft_width': 15.0, 'topk_frac': 0.5, 'bucket': 1, 'scaffold_bucket': 0, 'entropy_beta': 5.0, 'ad_threshold': 0.35, 'post_w': 0.25, 'eval_temp': 1.1}}

## Frontier over all evaluated snapshots (n per snapshot as set in config)

Best surrogate in-band fraction at each uniqueness level, among snapshots meeting all other gates and the holdout gap.

| uniqueness >= | trial | step | in-band (surr) | in-band (holdout) | uniq | int_div |
|---|---|---|---|---|---|---|
| 0.7 | reinvent-e72611 | 1200 | 0.717 | 0.722 | 0.731 | 0.797 |
| 0.75 | reinvent-50c3d1 | 700 | 0.685 | 0.755 | 0.796 | 0.804 |
| 0.8 | reinvent-50c3d1 | 600 | 0.624 | 0.720 | 0.846 | 0.817 |
| 0.85 | reinvent-a29221 | 900 | 0.604 | 0.542 | 0.976 | 0.800 |
| 0.9 | reinvent-a29221 | 900 | 0.604 | 0.542 | 0.976 | 0.800 |
| 0.95 | reinvent-a29221 | 900 | 0.604 | 0.542 | 0.976 | 0.800 |

## Trial hyper-parameters (confirmed trials)

| trial | hparams |
|---|---|
| reinvent-be70a9 | ad_threshold=0.0 batch=128 bucket=3 entropy_beta=0.0 eval_temp=1.1 init=steered lr=0.0001 post_w=0.25 scaffold_bucket=0 sigma=30.0 soft_width=25.0 topk_frac=1.0 |
| reinvent-8c31d6 | ad_threshold=0.0 batch=128 bucket=3 entropy_beta=0.0 eval_temp=1.0 init=steered lr=0.0001 post_w=0.0 scaffold_bucket=0 sigma=30.0 soft_width=25.0 topk_frac=1.0 |
| reinvent-a29221 | ad_threshold=0.35 batch=256 bucket=1 entropy_beta=5.0 eval_temp=1.1 init=steered lr=0.0001 post_w=0.25 scaffold_bucket=0 sigma=20.0 soft_width=15.0 topk_frac=0.5 |
| reinvent-0b32e1 | ad_threshold=0.0 batch=256 bucket=0 entropy_beta=20.0 eval_temp=1.1 init=plain lr=0.0002 post_w=-0.25 scaffold_bucket=0 sigma=50.0 soft_width=40.0 topk_frac=1.0 |
| reinvent-4bafab | ad_threshold=0.35 batch=256 bucket=0 entropy_beta=5.0 eval_temp=1.1 init=plain lr=0.0002 post_w=0.25 scaffold_bucket=10 sigma=50.0 soft_width=15.0 topk_frac=1.0 |
| reinvent-5fbebd | ad_threshold=0.35 batch=256 bucket=3 entropy_beta=0.0 eval_temp=1.0 init=steered lr=0.0001 post_w=0.0 scaffold_bucket=50 sigma=30.0 soft_width=40.0 topk_frac=0.5 |
| reinvent-947193 | ad_threshold=0.35 batch=256 bucket=1 entropy_beta=5.0 eval_temp=1.0 init=plain lr=0.0002 post_w=0.25 scaffold_bucket=10 sigma=30.0 soft_width=40.0 topk_frac=1.0 |
| reinvent-583a2d | ad_threshold=0.5 batch=128 bucket=1 entropy_beta=0.0 eval_temp=1.0 init=steered lr=0.0001 post_w=0.0 scaffold_bucket=0 sigma=30.0 soft_width=25.0 topk_frac=1.0 |
