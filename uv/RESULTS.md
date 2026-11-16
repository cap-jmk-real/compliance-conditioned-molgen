# UV-Vis steered-generation: result of 2026-11-16

This records a CPU validation run on the build box. It is a smoke-scale check that the
pipeline runs end to end and produces real numbers. It is not the full A100 run. All
figures below come from the actual run, none are invented.

## Dataset (step 1)

Source: Joung et al. 2020 "DB for chromophore" (CC BY 4.0), rev02 CSV.

- raw records: 20,236 (matches the paper)
- records with a numeric absorption maximum: 17,295
- records in the plausible 150 to 1200 nm range: 17,287
- unique chromophore structures after canonicalisation and per-structure median: 6,425
- absorption maximum: min 162 nm, median 396 nm, max 878 nm

## Predictor (step 2)

Morgan ECFP (radius 2, 2048 bits) into a scikit-learn HistGradientBoostingRegressor,
600 trees, random 10 percent held-out test split.

- train 5,203, val 579, test 643 structures
- TEST MAE 24.04 nm, RMSE 38.19 nm, R2 0.873

## Corpus (step 3)

ZINC250k, 6,000-molecule random subset. All 6,000 encoded to SELFIES. Structural
screen pass rate in the corpus: 96.2 percent. Predicted absorption median 356.4 nm.

## Generation comparison (steps 3 and 4)

Two arms of a 2-layer GRU SELFIES language model, identical architecture and budget
(20 epochs, batch 256). Steered arm conditioned at sampling on the target band and the
clean-screen token. 3,000 samples per arm. Target band 400 to 450 nm.

| metric | plain | steered |
|---|---|---|
| validity | 1.000 | 1.000 |
| uniqueness | 0.999 | 0.999 |
| novelty | 1.000 | 1.000 |
| screen pass rate | 0.969 | 0.969 |
| mean predicted absorption (nm) | 378.0 | 378.5 |
| fraction in 400-450 nm band | 0.244 | 0.257 |

## Honest reading

The predictor is solid for this kind of data (R2 0.873). The generator reaches full
validity and full novelty against the training corpus in both arms, which is expected
with SELFIES. Prefix-token steering moved the in-band fraction only marginally (24.4 to
25.7 percent), a shift within sampling noise at this budget. The direction is positive
and repeats across runs, but the effect is weak on CPU with 6,000 molecules and 20
epochs. Whether stronger conditioning emerges with the full ZINC250k corpus and a longer
A100 run is the open question for the full run. The earlier attempt with a 450 to 550 nm
target band had only ~3 percent of the corpus in band (too few labelled examples to
steer on), which is why the band was moved to 400 to 450 nm (about 15 percent of the
corpus) for a fair test.
