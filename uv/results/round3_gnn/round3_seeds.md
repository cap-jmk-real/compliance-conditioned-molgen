# Multi-seed confirmation

Config `reinvent-403657` at 1200 steps, 10000 fresh samples per prior seed. Fixed seed-0 surrogate. Gates unchanged.

| prior seed | in-band (surr) | in-band (holdout) | gap | valid | uniq | novel | rule | int_div | plain in-band | plain int_div | passes all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.667 | 0.782 | -0.115 | 1.000 | 0.926 | 1.000 | 0.943 | 0.808 | 0.189 | 0.886 | no: rule_pass_rate |
| 1 | 0.736 | 0.781 | -0.045 | 1.000 | 0.877 | 1.000 | 0.982 | 0.793 | 0.190 | 0.886 | no: uniqueness,int_div |
| 2 | 0.748 | 0.621 | 0.127 | 1.000 | 0.912 | 1.000 | 0.987 | 0.799 | 0.177 | 0.887 | no:  holdout gap |
| mean +- sd | 0.717 +- 0.044 | 0.728 +- 0.093 | -0.011 +- 0.125 | 1.000 +- 0.000 | 0.905 +- 0.025 | 1.000 +- 0.000 | 0.971 +- 0.024 | 0.800 +- 0.008 | 0.186 +- 0.007 | 0.886 +- 0.001 | not all |
