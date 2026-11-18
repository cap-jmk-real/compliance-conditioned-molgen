# Data and licence

## Sources

Sources, retrieval times and terms are listed in `DATA.md`. In short:

- ECHA lists (Candidate List, Annex XIV, Annex XVII): official public pages. ECHA's legal notice was not reachable when the data was fetched, so terms of re-use are not established for the ECHA tables.
- REACH Regulation consolidated text on EUR-Lex: CC BY 4.0.
- PubChem PUG REST: NCBI places no restrictions on its molecular data, but some records carry depositor rights.
- UV-Vis dataset: Joung et al. 2020, Scientific Data 7, 295, figshare item 12045567, CC BY 4.0.
- ZINC250k: from the Aspuru-Guzik `chemical_vae` release.

The EC inventory bulk file is a documented gap, see `DATA.md`.

## What the repository contains

Parsed tables in `data/`, the PubChem response cache, the cleaned UV-Vis dataset `uv/data/uvvis_dataset.csv` and small result files. It contains no synthesis routes and no hazardous or controlled structures beyond what appears as list entries in the public ECHA tables.

## Licence

- Source code (scripts and tools): MIT, see `LICENSE`.
- Manuscript text and figures of the accompanying paper: CC BY 4.0, see `LICENSE-text`.
- Third-party data keep their own terms (see `DATA.md`).
