# Reproduce the results

## 1. Tests

<!-- ci:run -->
```bash
python -m pytest tests
```

The tests cover the steering methods, the meta-search driver and the SMARTS rules on small fixtures. They do not check paper numbers.

## 2. Coverage numbers from committed data

The three scripts read the committed tables and the PubChem cache and rewrite `COVERAGE.md` and the CSV files with identical content, so `git diff` shows no change.

<!-- ci:run -->
```bash
python resolve_pubchem.py
python classify_coverage.py
python smarts_feasibility.py
git diff --exit-code COVERAGE.md data
```

The numbers are in `COVERAGE.md` and are summarised in the README.

## 3. UV-Vis smoke run

The smoke run executes the full pipeline on a few hundred molecules and one epoch. It checks that the pipeline runs and writes `uv/artifacts/eval_smoke.json`. It does not reproduce paper numbers. It needs two source CSVs in `.ci-cache/`, as described in the README.

<!-- ci:run -->
```bash
mkdir -p .ci-cache
test -s .ci-cache/zinc250k.csv || curl -fL --retry 5 --retry-delay 5 --retry-all-errors -o .ci-cache/zinc250k.csv "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
test -s .ci-cache/chromophore_rev02.csv || { echo "missing .ci-cache/chromophore_rev02.csv, see README" >&2; exit 1; }
python uv/run.py --smoke --ckpt-root /tmp/t25-ckpt --uvvis-local .ci-cache/chromophore_rev02.csv --zinc-local .ci-cache/zinc250k.csv
test -s uv/artifacts/eval_smoke.json
```

## 4. Full run

The full pipeline needs a GPU. The Colab A100 recipe is in `uv/README.md`. The numbers it produces are written to `uv/artifacts/` and summarised in `uv/RESULTS.md`. CI does not run it.

## What each level checks

| Level | Checks |
| --- | --- |
| Tests | Code behaviour on fixtures. No paper numbers. |
| Coverage rerun | That the committed coverage tables follow from the committed data and cache. |
| Smoke run | That the UV-Vis pipeline runs end to end. No accuracy claim. |
| Full run | The UV-Vis numbers themselves. |
