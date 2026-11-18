# Second UV-Vis dataset: does the steering effect replicate?

The main result uses one experimental absorption-maximum dataset (Joung et al. 2020) to
train the surrogate that both labels the corpus and scores the samples. This folder repeats
the plain vs steered in-band comparison (band 400 to 450 nm, 3 seeds, paired t-test, same
gates) with a second, independent surrogate trained on a different dataset.

## Dataset

Beard, Sivaraman, Vazquez-Mayagoitia, Vishwanath, Cole (2019), "Comparative dataset of
experimental and computational attributes of UV/vis absorption spectra", Scientific Data 6,
307, [10.1038/s41597-019-0306-0](https://doi.org/10.1038/s41597-019-0306-0). Data record:
figshare [10.6084/m9.figshare.7619672.v2](https://doi.org/10.6084/m9.figshare.7619672.v2),
licence CC BY 4.0 (figshare record metadata and the open-access article).

Why this one: it is a different experimental lambda_max collection, text-mined with
ChemDataExtractor from 402,034 documents, while the main dataset is a manual literature
compilation. The licence is explicit. Only 228 of its 7,430 clean structures (3.1 %) also
occur in the main dataset (measured by `prepare_beard2019 --compare`, canonical SMILES
join), and on those the two sources differ by a median of 18 nm.

Considered and not used: nablaColors (Zenodo 18061300, CC BY 4.0) and ChemFluor (figshare
12110619, no licence stated on the record). The first is a re-compilation that includes the
main dataset, so it is not independent; the second has no stated licence.

Known weakness, stated up front: text-mined values are noisier than curated ones. The
surrogate trained on it reaches a held-out MAE of 73.6 nm and R2 of 0.32 (main dataset:
24.0 nm, R2 0.87). The experiment is therefore a test of whether steering toward a weaker,
different oracle still shifts the samples. The in-band fraction under the main dataset's
surrogate is reported alongside as a check that does not depend on the oracle that produced
the labels.

## Protocol

Identical to the main result, with these stated differences:

- The surrogate is trained on this dataset (same features, model, 600 trees, same split
  seeds). Cleaning mirrors `uv/prepare_uvvis.py`: numeric lambda_max, 150 to 1200 nm,
  canonical SMILES, median per structure.
- The corpus is the same 249,455 ZINC250k molecules in the same order, relabelled with the
  new surrogate (`relabel_corpus.py` asserts that only `abs_pred` changed).
- The plain arm does not read the labels, so its training data are identical to the main
  run. The trained plain checkpoints of seeds 0, 1, 2 are reused. Only the steered arm is
  retrained (618 epochs, batch 512, lr 1e-3, 301,584 steps, seeds 0, 1, 2).
- Evaluation: 20,000 samples per arm, same definitions and rounding as `uv/evaluate.py`.
  `evaluate_ds2.py` adds internal diversity (1 - mean pairwise Tanimoto, up to 1,000
  molecules, as in `uv/steer_eval.py`) so every gate can be checked: validity >= 0.95,
  uniqueness >= 0.90, novelty >= 0.90, rule pass >= 0.95, internal diversity >= 0.9 x plain.
- The structural screen is a metric only. It is never a conditioning target and never a
  reward. Nothing here concerns synthesis or routes.

## Run

```bash
# 1. dataset, surrogate, relabelled corpus (CPU, a few minutes). --local-csv is the
#    data/CSV/paper_allDB.csv member of the figshare data.zip; without it the zip is
#    downloaded (382 MB, md5 and sha256 checked).
python -m uv.dataset2.run_dataset2 --root ckpt-ds2 prepare \
    --main-corpus CKPT/shared/corpus.csv --main-vocab CKPT/shared/vocab.json \
    --main-dataset CKPT/shared/uvvis_dataset.csv

# 2. steered arm per seed (GPU, about 2.4 h each on an RTX 3060)
python -m uv.dataset2.run_dataset2 --root ckpt-ds2 train --seed 0

# 3. evaluation per seed, then the aggregate
python -m uv.dataset2.run_dataset2 --root ckpt-ds2 evaluate --seed 0 \
    --plain-ckpt 0=PLAIN_SEED0.pt 1=PLAIN_SEED1.pt 2=PLAIN_SEED2.pt \
    --main-predictor CKPT/shared/uvvis_predictor.joblib
python -m uv.dataset2.run_dataset2 --root ckpt-ds2 report --seeds 0 1 2
```

`CKPT` is the main run's checkpoint root. `uv/engine/run_dataset2.sh` runs the whole chain
on the engine behind the meta-search and `uv/engine/backup_dataset2.sh` backs the results up.
Results are written to `<root>/results_ds2.md` and `<root>/aggregate_ds2.json`.
Tests: `python -m pytest tests/test_dataset2.py`.
