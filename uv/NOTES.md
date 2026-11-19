# UV-Vis experiment notes

## Safety scope

This experiment is optical absorption in ordinary organic and dye chemistry. The
`smarts_rules.py` families are a screening metric only. "Pass the screen" means the molecule
matches none of those flagged families. The rules are never optimised toward and never used
as a reward. No controlled substances or precursors are a target. No synthesis, reactions or
routes appear anywhere.

## Steering design

Both arms are the same GRU language model with the same hyper-parameters and the same number
of training steps. The steered arm prepends two control tokens to every sequence right after
`<bos>`:

- band token: `<BAND_IN>` if the predicted absorption maximum is inside the target band,
  else `<BAND_OUT>`
- screen token: `<CLEAN>` if the molecule passes the structural screen, else `<FLAG>`

At sampling time the steered arm is seeded with `<BAND_IN><CLEAN>`, so it is asked for
molecules in the target band that also pass the screen. The plain arm is sampled
unconditionally. Any difference in the band fraction between the arms is the steering effect,
measured by the same predictor and the same screen on both arms' outputs.

## What is measured vs. predicted

The absorption maximum of a generated molecule is a prediction from the step-2 model, not an
experiment. The predictor's held-out error (reported by `train_predictor.py`) bounds how far
those predictions can be trusted. The band fraction and screen pass rate are therefore a
comparison between the two arms under the same measuring stick, not absolute physical claims.

## Run modes and instrumentation

`uv/run.py` keeps child output unbuffered (it runs each step with `python -u` and sets
`PYTHONUNBUFFERED=1`), so progress prints as the run goes.

- `--calibration` runs a small fixed workload through `uv/calibrate.py`. It prints the
  detected device and GPU name, the torch and CUDA versions, the workload it ran
  (dataset rows, epochs, batch size, model parameter count), wall-clock per stage,
  training steps per second and generation samples per second. It writes
  `calibration.json` to the checkpoint root and projects the full-run wall-clock. On a
  box with no GPU these are CPU rates, so the projection is a CPU projection. Rerun on
  the A100 to size the real run.
- `--full` runs the complete pipeline: the full UV-Vis dataset and the full ZINC250k
  corpus, both generator arms with identical architecture, data and budget (only the
  two prefix control tokens differ), then evaluation. It reads `calibration.json` and
  sizes the generator epochs so the run lands near `--target-hours` (default 4.5),
  between 3 and 6 hours on an A100. Pass `--gen-epochs` to set the budget by hand.
  Checkpoints go to the checkpoint root and the run resumes from the newest one.
- Evaluation reports, per arm, validity, uniqueness, novelty, rule pass rate, in-band
  fraction and mean predicted value, plus the regressor held-out MAE, RMSE and R2 and
  all sample counts. It adds a bootstrap 95 percent confidence interval on the
  difference in in-band fraction between the arms, so the comparison stays honest about
  whether any difference is real.

## Multi-seed runs (seeds 1 and 2) and the fixed surrogate

Seed 0 is the single run reported in the preprint. To quantify training variance (the
reviewer's main request), `colab_run_seeds.py` adds independent training seeds. For each
seed it trains both arms (plain, steered) with EXACTLY the seed-0 hyperparameters and
step budget (618 epochs / 301,584 steps per arm, batch 512, lr 1e-3, emb 128, hidden
256, 2 layers, band 400-450 nm) and evaluates on 20,000 samples per arm with a 2,000-
resample bootstrap, identical to the paper.

Decision: the surrogate property model AND the corpus are held FIXED at seed 0. Both are
built deterministically (sklearn split seeds 42/7, numpy seed 0, the full ZINC250k in
release order with no subsampling at `--full` scale), so they are identical across all
seeds and are built once and shared. The ONLY things that vary with `--seed` are the
generator's weight initialisation (torch/cuda) and the data-shuffle order (numpy). This
is deliberate: a per-seed surrogate would mix surrogate variance into the measured
steering effect, so the arms must be compared under one surrogate. λmax is in all cases
a surrogate prediction, not a measurement.

`--seed` in `train_generator.py` seeds python `random`, numpy, torch and cuda, and
offsets the per-epoch shuffle seed (`seed * 100000 + epoch`), so seed 0 reproduces the
original shuffle exactly. The seed also goes into the checkpoint directory name
(`uv-generator-<arm>-seed<N>`). Same seed in, same run out (verified on CPU: identical
final loss on a rerun).

Aggregate with `python -m uv.aggregate_seeds uv/results/eval_results.json
<drive>/eval_seed1.json <drive>/eval_seed2.json`. It reports mean +/- sd per metric per
arm and a paired check on the steered-minus-plain in-band fraction. With n=3 that paired
test is badly underpowered and is reported as a direction-and-size check, not a
significance claim.
