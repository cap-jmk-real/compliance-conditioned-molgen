# Track 25: structure-resolution coverage of EU chemical inventory lists

This folder measures how many entries of three EU chemical lists can be written as one machine-readable structure. It is the first milestone of track 25 (compliance-conditioned molecular generation, plan in `docs/25-compliance-conditioned-molecular-generation.md`). There is no model and no training here. Everything runs on CPU.

Lists measured:

- ECHA Candidate List of substances of very high concern (253 entries)
- REACH Annex XIV, the authorisation list (59 entries)
- REACH Annex XVII, the restriction list (79 entries), plus Appendices 1, 2, 4, 5 and 6 of Annex XVII (1,612 CMR index entries that entries 28 to 30 refer to)

Structures come from PubChem, looked up by CAS number and separately by name, with InChIKey as the key. Sources, dates, terms and the one gap (the EC inventory bulk file) are in `DATA.md`.

## Result in short

Share of entries that resolve to one single-component organic structure, with no disagreement between the CAS and name lookups:

| List | Entries | Usable single structures | Of those, confirmed by CAS and name |
|---|---|---|---|
| Candidate List | 253 | 123 (49%) | 111 (44%) |
| Annex XIV | 59 | 24 (41%) | 22 (37%) |
| Annex XVII | 79 | 28 (35%) | 25 (32%) |
| Annex XVII appendices 1 to 6 | 1,612 | 282 (17%) | 229 (14%) |

After removing duplicates across lists, that is 313 distinct usable structures, 271 of them confirmed by both lookups. Every Annex XVII entry is conditional on use, concentration or product type, so for that list a structure says which family a molecule belongs to and never whether the restriction applies. Full tables: `COVERAGE.md`. SMARTS analysis of the group entries: `SMARTS.md`.

## Run it

```
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
DISPLAY=:0 python ingest_echa.py      # headed Chrome, see DATA.md for why
python ingest_eurlex.py
DISPLAY=:0 python ingest_ec_inventory.py   # records the gap if the file is still missing
python parse_lists.py
python parse_eurlex.py
python resolve_pubchem.py            # offline from cache/pubchem_cache.json.gz
python classify_coverage.py          # writes data/entry_classification.csv and COVERAGE.md
python smarts_feasibility.py         # writes data/group_smarts_feasibility.csv
python -m pytest tests
```

The ECHA scripts need a display and a Chrome binary at `/usr/bin/google-chrome`. Raw downloads go to `raw/` and are not committed. A fresh checkout can skip the ingestion and parsing steps because the parsed tables in `data/` and the PubChem cache are committed.

## Files

| File | Purpose |
|---|---|
| `ingest_echa.py`, `ingest_eurlex.py`, `ingest_ec_inventory.py` | fetch and cache the sources, write `raw/fetch_meta.json` |
| `parse_lists.py`, `parse_eurlex.py` | turn cached pages into `data/*.csv` |
| `resolve_pubchem.py` | CAS and name lookups in PubChem, cached |
| `classify_coverage.py` | one structural class per entry, usability flags, `COVERAGE.md` |
| `smarts_feasibility.py` | which non-single entries a SMARTS rule could express |
| `smarts_rules.py`, `tests/test_smarts_rules.py` | five worked SMARTS rules with match and non-match tests |
| `DATA.md`, `COVERAGE.md`, `SMARTS.md` | sources, measured coverage, SMARTS analysis |

## Limits

- The structural classes for groups and variable-composition substances come from keyword rules on entry names. I read every non-single entry of the three main lists. For the appendices I read a sample.
- PubChem can map a CAS number to a wrong or arbitrary record. The cross-check against the name lookup and an element check catch some of this. Entries confirmed by only one lookup carry more risk.
- Hydrates and anhydrous forms of one substance count as several structures here.
- This is a measurement of the lists as published on 16 Nov 2026. It is not legal advice and does not decide whether any use of a substance is permitted.

## UV-Vis steered generation (step 2 of the track)

The `uv/` folder adds a small, runnable experiment: steer a SELFIES molecular language
model toward a target UV-Vis absorption band while passing the structural-pattern screen
in `smarts_rules.py`, compared with an identical unsteered model. It is ordinary dye and
materials optics. No synthesis, reactions, routes or toxicology. The SMARTS rules are a
screening metric only, never a target or a reward. See `uv/README.md` for the Colab A100
recipe and `uv/RESULTS.md` for the measured validation result.
