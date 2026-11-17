# UV-Vis steered molecular generation (track 25)

This folder holds a small, self-contained experiment that is ordinary materials and dye
chemistry. The question is narrow: can a small molecular language model be steered toward
a target optical property, the first UV-Vis absorption maximum in nm, while also passing a
structural-pattern screen, compared with an identical unsteered model.

The structural rules in `../smarts_rules.py` are used here only as a screening metric. A
molecule "passes" the screen when it matches none of the flagged structural families. The
rules are never a generation target and never a reward. There is no synthesis, no reaction
planning, no route prediction and no toxicology anywhere in this folder.

## Pipeline

1. `prepare_uvvis.py` builds a clean dataset of SMILES plus first absorption maximum (nm)
   from Joung et al. 2020 (Scientific Data 7, 295, figshare "DB for chromophore", CC BY 4.0).
2. `train_predictor.py` fits a UV-Vis absorption-maximum predictor (Morgan ECFP radius 2,
   2048 bits, into a scikit-learn HistGradientBoostingRegressor). Reports MAE, RMSE, R2 on a
   held-out test split.
3. `prepare_corpus.py` builds a SELFIES corpus from ZINC250k and labels each molecule with
   the predicted absorption maximum and the structural-screen result.
4. `train_generator.py` trains one arm of a small GRU SELFIES language model. Two arms share
   the architecture and training budget: `plain` and `steered`. The steered arm prepends a
   band token and a screen token to every sequence.
5. `evaluate.py` samples both arms over a fixed count and reports validity, uniqueness,
   novelty, structural-screen pass rate, mean predicted absorption maximum, and the fraction
   inside the target band. Writes a results JSON and a markdown table.
6. `run.py` is the single entrypoint with `--calibration`, `--smoke` and full modes.

## Data sources (verified 2026-11-16)

- UV-Vis: Joung et al. 2020, figshare item 12045567 (CC BY 4.0). The figshare REST and
  ndownloader hosts returned HTTP 403 from the build box network (Cloudflare). A
  byte-identical GitHub-raw mirror of the published rev02 CSV is used as the fallback and
  is pinned by commit. Raw rows in rev02: 20,236, matching the paper.
- Corpus: ZINC250k from the Aspuru-Guzik `chemical_vae` release
  (`raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/.../250k_rndm_zinc_drugs_clean_3.csv`,
  HTTP 200). 249,455 molecules.

## Colab recipe (A100)

```bash
# 1. clone
!git clone https://github.com/cap-jmk-real/compliance-conditioned-molgen.git
%cd compliance-conditioned-molgen

# 2. install
!pip install -q rdkit selfies scikit-learn pandas joblib torch

# 3. mount Drive for checkpoints
from google.colab import drive
drive.mount('/content/drive')

# 4. calibration: short fixed workload, reports wall-clock and throughput
!python -m uv.run --calibration --ckpt-root /content/drive/MyDrive/t25-checkpoints

# 5. (optional) record the Colab unit readings around the calibration run
!python colab_burn_rate.py record \
    --log /content/drive/MyDrive/t25-checkpoints/burn_rate.json \
    --accelerator A100 --units-before <before> --units-after <after> --minutes <mins>

# 6. full pipeline (downloads data itself, checkpoints to Drive, resumes on restart)
!python -m uv.run --ckpt-root /content/drive/MyDrive/t25-checkpoints \
    --pred-trees 600 --corpus-n 20000 --gen-epochs 20 --n-samples 5000
```

Every training script checkpoints to `--ckpt-root` (default
`/content/drive/MyDrive/t25-checkpoints`), writes checkpoints during training, and resumes
from the latest one automatically. Re-running any step after a disconnect continues from the
last checkpoint. The target band defaults to 450 to 550 nm and is set with `--band-lo` and
`--band-hi`.

## Model choices

- Predictor: Morgan fingerprint plus HistGradientBoostingRegressor, chosen for a clean
  install and a few-minute fit. Chemprop is supported as an optional import check
  (`--try-chemprop`) but the committed result uses the fingerprint model.
- Generator: a 2-layer GRU language model over SELFIES tokens. Small enough to train many
  epochs in minutes on an A100, large enough to reach high validity on ZINC-like molecules.
  SELFIES guarantees syntactic validity of the decoded strings, so validity reflects RDKit
  parse success.


## Steering meta-search (exp/steering-metasearch)

Stronger steering options (`uv/steering.py`: logit guidance, REINVENT-style fine-tuning,
surrogate-filtered self-training) and a successive-halving meta-search over them
(`python -m uv.metasearch ...`, cost projection with `python -m uv.metasearch_cost ...`,
second UV predictor in `uv/holdout_predictor.py`). Read `uv/STEERING_PLAN.md` first. This
does not change the existing pipeline or the preprint's method.
