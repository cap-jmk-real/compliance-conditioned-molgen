# Repository layout

| Path | Contents |
| --- | --- |
| `ingest_echa.py`, `ingest_eurlex.py`, `ingest_ec_inventory.py` | Fetch and cache the list sources, write `raw/fetch_meta.json`. |
| `parse_lists.py`, `parse_eurlex.py` | Turn cached pages into `data/*.csv`. |
| `resolve_pubchem.py` | CAS and name lookups in PubChem, cached in `cache/pubchem_cache.json.gz`. |
| `classify_coverage.py` | One structural class per entry, usability flags, writes `COVERAGE.md` and `data/entry_classification.csv`. |
| `smarts_feasibility.py`, `smarts_rules.py` | Which non-single entries a SMARTS rule could express, and five worked rules with tests. |
| `colab_checkpoint.py` | Checkpoint and progress helpers for Colab runs. |
| `data/`, `cache/`, `raw/` | Parsed tables, PubChem cache, and fetch metadata. Raw downloads are not committed. |
| `uv/` | UV-Vis steered generation: `prepare_uvvis.py`, `train_predictor.py`, `prepare_corpus.py`, `train_generator.py`, `evaluate.py`, `run.py`, plus `steering.py`, `metasearch.py`, `metasearch_cost.py`, `holdout_predictor.py` and `engine/` launchers. |
| `uv/artifacts/`, `uv/results/` | Committed predictor metrics, vocabulary, evaluation results and the meta-search CPU smoke output. |
| `tests/` | Unit tests for SMARTS rules, steering and meta-search. |
| `DATA.md`, `COVERAGE.md`, `SMARTS.md` | Sources and terms, measured coverage, SMARTS analysis. |
| `uv/README.md`, `uv/RESULTS.md`, `uv/NOTES.md`, `uv/STEERING_PLAN.md` | UV-Vis pipeline, result of the 2026-11-16 CPU validation run, design notes, steering plan. |
| `NOTES.md` | Where the track came from and the safety rule. |
| `.github/workflows/` | `ci.yml` (tests and UV-Vis smoke) and `docs-sbom.yml` (docs build, SBOM, README commands). |
| `tools/run_readme_blocks.py` | Executes the CI-marked commands in the README and docs. |
