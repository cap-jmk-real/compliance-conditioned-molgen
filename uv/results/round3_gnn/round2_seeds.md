# Multi-seed confirmation

Config `reinvent-a29221` at 1150 steps, 10000 fresh samples per prior seed. Fixed seed-0 surrogate. Gates unchanged.

| prior seed | in-band (surr) | in-band (holdout) | gap | valid | uniq | novel | rule | int_div | plain in-band | plain int_div | passes all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.581 | 0.502 | 0.079 | 1.000 | 0.989 | 0.999 | 0.998 | 0.813 | 0.189 | 0.886 | YES |
| 1 | 0.607 | 0.597 | 0.010 | 1.000 | 0.981 | 0.999 | 1.000 | 0.792 | 0.190 | 0.886 | no: int_div |
| 2 | 0.607 | 0.530 | 0.077 | 1.000 | 0.987 | 1.000 | 1.000 | 0.792 | 0.177 | 0.887 | no: int_div |
| mean +- sd | 0.598 +- 0.015 | 0.543 +- 0.049 | 0.055 +- 0.040 | 1.000 +- 0.000 | 0.986 +- 0.005 | 1.000 +- 0.000 | 0.999 +- 0.001 | 0.799 +- 0.012 | 0.186 +- 0.007 | 0.886 +- 0.001 | not all |
