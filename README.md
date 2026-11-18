# compliance-conditioned-molgen

Research track 25 of Julian M. Kleber: how far can EU chemical-inventory lists be written as machine-readable structures, and can a small molecular language model be steered toward an optical property while passing a structural-pattern screen.

[![CI](https://github.com/cap-jmk-real/compliance-conditioned-molgen/actions/workflows/ci.yml/badge.svg)](https://github.com/cap-jmk-real/compliance-conditioned-molgen/actions/workflows/ci.yml)

## What this is

Two parts share one repository.

1. Coverage measurement (repository root). Three EU lists are resolved to structures through PubChem: the ECHA Candidate List of substances of very high concern, the REACH Annex XIV authorisation list and the REACH Annex XVII restriction list, plus Appendices 1, 2, 4, 5 and 6 of Annex XVII. The scripts count how many entries can be written as one structure and how many group entries a SMARTS rule could express. Everything runs on CPU from committed tables and a committed PubChem cache.
2. UV-Vis steered generation (`uv/`). A small SELFIES language model is steered toward a target UV-Vis absorption band (400 to 450 nm) and compared with an identical unsteered model. The structural-pattern rules are a screening metric only.

Safety scope. This is ordinary optical chemistry on dyes and materials. Controlled substances and drug precursors are a blocklist only. They are never a conditioning target and never a reward. The repository contains no synthesis routes, no retrosynthesis and no hazardous or controlled structures, and the documentation does not add any.

A map of the repository is in [docs/layout.md](docs/layout.md). The documentation site is built with MkDocs Material from `docs/` and `mkdocs.yml`.

## Quickstart

Python 3.13. The root `requirements.txt` is pinned to versions with 3.13 wheels. This installs the CPU build of PyTorch and runs the unit tests.

<!-- ci:run -->
```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
grep -vE '^torch([[:space:]]|$)' uv/requirements.txt > uv-requirements-no-torch.txt
pip install -r uv-requirements-no-torch.txt
python -m pytest tests
```

## Reproducing the coverage numbers

The parsed tables in `data/` and the PubChem cache in `cache/` are committed, so no network is needed. The three scripts rewrite `COVERAGE.md` and the CSV files with identical content, which `git diff` confirms.

<!-- ci:run -->
```bash
python resolve_pubchem.py
python classify_coverage.py
python smarts_feasibility.py
git diff --exit-code COVERAGE.md data
```

The result, as written in `COVERAGE.md`:

| List | Entries | Usable single structures | Of those, confirmed by CAS and name |
|---|---|---|---|
| Candidate List | 253 | 123 (49%) | 111 (44%) |
| Annex XIV | 59 | 24 (41%) | 22 (37%) |
| Annex XVII | 79 | 28 (35%) | 25 (32%) |
| Annex XVII appendices 1 to 6 | 1,612 | 282 (17%) | 229 (14%) |

"Usable" means one single-component organic structure with common elements and no disagreement between the CAS and the name lookup. After removing duplicates across lists there are 313 distinct usable structures, 271 of them confirmed by both lookups. Every Annex XVII entry is conditional on use, concentration or product type, so for that list a structure says which family a molecule belongs to and never whether a restriction applies. The SMARTS analysis of group entries is in `SMARTS.md`.

Re-fetching the sources (`ingest_echa.py`, `ingest_eurlex.py`, `ingest_ec_inventory.py`, then `parse_lists.py` and `parse_eurlex.py`) needs a display and a Chrome binary at `/usr/bin/google-chrome` for the ECHA pages. See `DATA.md` for why. Raw downloads go to `raw/` and are not committed.

## UV-Vis smoke run and full run

| Run | Command | Checks | Needs |
|---|---|---|---|
| Unit tests | `python -m pytest tests` | Steering methods, meta-search driver and SMARTS rules on small fixtures. No paper numbers. | CPU |
| Smoke | `python uv/run.py --smoke ...` | The whole pipeline end to end on a few hundred molecules and one epoch: dataset, predictor, corpus, both generator arms, evaluation. Writes `uv/artifacts/eval_smoke.json`. It checks that the pipeline runs and produces the expected files. It does not reproduce paper numbers. | CPU, two source CSVs |
| Full | `python -m uv.run --ckpt-root ... --pred-trees 600 --corpus-n 20000 --gen-epochs 20 --n-samples 5000` (Colab A100 recipe in `uv/README.md`) | Produces the numbers in `uv/RESULTS.md`. | GPU |

The smoke run needs two source CSVs in `.ci-cache/`. The ZINC250k file is downloaded below. The UV-Vis file is Joung et al. 2020, rev02, figshare item 12045567 (CC BY 4.0), saved as `.ci-cache/chromophore_rev02.csv`. The GitHub mirror that `uv/prepare_uvvis.py` used as its fallback (`howardwang1997/AI4PS`) returned HTTP 404 on 2026-11-18, and the figshare download host returned 403 from the machine used to write this. In CI the file comes from the Actions cache that `ci.yml` also uses.

<!-- ci:run -->
```bash
mkdir -p .ci-cache
test -s .ci-cache/zinc250k.csv || curl -fL --retry 5 --retry-delay 5 --retry-all-errors -o .ci-cache/zinc250k.csv "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
test -s .ci-cache/chromophore_rev02.csv || { echo "missing .ci-cache/chromophore_rev02.csv, see README" >&2; exit 1; }
python uv/run.py --smoke --ckpt-root /tmp/t25-ckpt --uvvis-local .ci-cache/chromophore_rev02.csv --zinc-local .ci-cache/zinc250k.csv
test -s uv/artifacts/eval_smoke.json
```

Results on file. `uv/RESULTS.md` records a CPU validation run of 2026-11-16: predictor test MAE 24.04 nm and R2 0.873, and an in-band fraction (400 to 450 nm) of 0.244 for the plain arm and 0.257 for the steered arm at 3,000 samples per arm. The file calls this a smoke-scale check, with a difference within sampling noise. Stronger steering methods and a successive-halving meta-search are in `uv/steering.py` and `uv/metasearch.py`, with the plan in `uv/STEERING_PLAN.md`.

Every command in a code block marked for CI in this README and in `docs/reproduce.md` is executed by `tools/run_readme_blocks.py` in the `docs-sbom` workflow, so the documented commands cannot drift from what CI runs.

## Limits

- Structural classes for groups and variable-composition substances come from keyword rules on entry names. Every non-single entry of the three main lists was read. For the appendices a sample was read.
- PubChem can map a CAS number to a wrong or arbitrary record. The cross-check against the name lookup and an element check catch some of this.
- Hydrates and anhydrous forms of one substance count as several structures.
- The coverage is a measurement of the lists as published on 16 Nov 2026. It is not legal advice and does not decide whether any use of a substance is permitted.
- UV-Vis absorption of generated molecules is a surrogate prediction, not a measurement.

## Citation

Cite with the metadata in [CITATION.cff](CITATION.cff). Author: Julian M. Kleber, ORCID [0000-0001-5518-0932](https://orcid.org/0000-0001-5518-0932). Contact: julian.m.kleber@gmail.com.

## Licence

This repository has no licence file yet, so no licence is granted. Data licences are listed in [DATA.md](DATA.md) and `uv/README.md`.
