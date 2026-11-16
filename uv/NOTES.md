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
