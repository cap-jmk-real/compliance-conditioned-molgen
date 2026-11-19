# RTX 3060 multi-seed runs: hardware, software and timing log

Extracted from the training logs of the three-seed runs (seeds 0, 1, 2; both arms).
Host-specific paths are omitted; every value below is copied from the logs or from a
read-only query on the training machine (2026-11-18).

## Hardware and software

| item | value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 12288 MiB, 1 GPU (`nvidia-smi`) |
| NVIDIA driver | 550.163.01 |
| Python | 3.13.5 |
| PyTorch | 2.6.0+cu124 (CUDA 12.4) |
| scikit-learn | 1.9.1 |
| NumPy | 2.5.5 |
| RDKit | 2026.09.2 |
| cuDNN deterministic mode | not forced (see `uv/train_generator.py:set_seed`) |

## Final training step per run (from the training log)

| seed | arm | steps | epochs | elapsed (min) | steps/s | final loss |
|---|---|---|---|---|---|---|
| 0 | plain | 301584 | 618 | 139.1 | 36.14 | 0.6173 |
| 0 | steered | 301584 | 618 | 143.1 | 35.12 | 0.5988 |
| 1 | plain | 301584 | 618 | 139.3 | 36.08 | 0.6276 |
| 1 | steered | 301584 | 618 | 141.2 | 35.60 | 0.6067 |
| 2 | plain | 301584 | 618 | 138.7 | 36.25 | 0.6224 |
| 2 | steered | 301584 | 618 | 144.9 | 34.70 | 0.6053 |

Raw log lines (`epoch=617` is the zero-based index of the 618th epoch):

```
seed 0 plain   step 301584 | of 301584 | eta 0.0 min | elapsed 139.1 min | 36.14 steps/s | epoch=617 | loss=0.6173
seed 0 steered step 301584 | of 301584 | eta 0.0 min | elapsed 143.1 min | 35.12 steps/s | epoch=617 | loss=0.5988
seed 1 plain   step 301584 | of 301584 | eta 0.0 min | elapsed 139.3 min | 36.08 steps/s | epoch=617 | loss=0.6276
seed 1 steered step 301584 | of 301584 | eta 0.0 min | elapsed 141.2 min | 35.60 steps/s | epoch=617 | loss=0.6067
seed 2 plain   step 301584 | of 301584 | eta 0.0 min | elapsed 138.7 min | 36.25 steps/s | epoch=617 | loss=0.6224
seed 2 steered step 301584 | of 301584 | eta 0.0 min | elapsed 144.9 min | 34.70 steps/s | epoch=617 | loss=0.6053
```

The surrogate's held-out metrics (MAE 24.04 nm, RMSE 38.19 nm, R2 0.8734, 5203/643
train/test, 600 trees) are identical in all three `eval_seed*.json` files and in the A100
artifact `uv/results/eval_results.json`.
