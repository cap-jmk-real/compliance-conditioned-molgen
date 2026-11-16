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
