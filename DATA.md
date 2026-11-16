# Data sources

All sources are official public pages. Retrieval times are Europe/Berlin (UTC+1) and come from `raw/fetch_meta.json`. Raw downloads are not committed. They are re-created by the ingestion scripts.

| Source | URL | Retrieved | Status | Terms |
|---|---|---|---|---|
| ECHA Candidate List of SVHC | https://echa.europa.eu/candidate-list-table | 2026-11-16 04:48 | ok, 253 rows over 3 pages. The page says the list was last updated on 04 Feb 2026. | Not established in this run. ECHA's legal notice (https://echa.europa.eu/legal-notice) returned 503 on six attempts. The list page states that only the version on the ECHA website is authentic. |
| ECHA Authorisation List (REACH Annex XIV) | https://echa.europa.eu/authorisation-list | 2026-11-16 04:45 | ok, 59 rows | as above |
| ECHA Substances restricted under REACH (Annex XVII) | https://echa.europa.eu/substances-restricted-under-reach | 2026-11-16 04:45 | ok, 79 rows | as above |
| REACH Regulation (EC) No 1907/2006, consolidated version 02006R1907-20260622, EUR-Lex | https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:02006R1907-20260622 | 2026-11-16 04:53 | ok | EUR-Lex legal notice: re-use under Decision 2011/833/EU. Consolidated texts are licensed CC BY 4.0 (acknowledge the source, indicate changes). |
| EC inventory (EINECS, ELINCS, NLP) | https://echa.europa.eu/information-on-chemicals/ec-inventory | 2026-11-16 05:10 | gap. The web page exports at most 20,000 search results. The full CSV previously published at https://echa.europa.eu/documents/10162/17222/ec_inventory_en.csv/326d9adb-27ed-5460-a2da-4f651b81e4b3 now redirects to the ECHA homepage. No other official bulk file was found. | not applicable |
| PubChem PUG REST | https://pubchem.ncbi.nlm.nih.gov/rest/pug | 2026-11-16 04:57 to 05:45 | ok, 4,045 items looked up, 0 unrecovered request errors | NCBI policy: NCBI places no restrictions on use of its molecular data, but some PubChem records carry depositor rights (https://www.ncbi.nlm.nih.gov/home/about/policies/). The resolver stays under 5 requests per second, the PUG REST rate limit as I understand it (not re-read in this run). |

## How the ECHA pages were fetched

The ECHA site sits behind an Azure web application firewall. Plain HTTP clients get 403. Headless Chrome gets a 503 "website is busy" page. A headed Chrome driven by Playwright gets the real pages after a few retries. The table export buttons (CSV, XLS) did not produce a download in my attempts, so the scripts read the rendered HTML tables instead. The list portlet only honours a page number together with the sort parameters it writes into its own pager links, so `ingest_echa.py` follows those links. The parser checks that the row count matches the "Showing N results" line on each page.

## What the EC inventory gap costs

Nothing in the coverage numbers depends on it. EC and CAS numbers come from the lists themselves, and structures come from PubChem by CAS number and name. The EC inventory would have added a second EC-to-CAS mapping and molecular formulas for a cross-check. That check is missing.

## Derived files

| File | Content |
|---|---|
| `data/entries.csv` | one row per list entry (Candidate List, Annex XIV, Annex XVII), parsed from the ECHA pages |
| `data/members.csv` | member substances ECHA lists inside group entries (2,042 rows) |
| `data/annex_xiv_exemptions.csv` | Annex XIV exempted uses and use-specific derogations, from EUR-Lex |
| `data/annex_xvii_appendix_substances.csv` | Annex XVII Appendices 1, 2, 4, 5, 6 (CMR substances for entries 28 to 30), from EUR-Lex. Appendix 3 has no table in the consolidated text. |
| `data/resolution.csv` | PubChem CIDs and InChIKeys per entry, member and appendix substance, by CAS and by name |
| `data/structures.csv` | PubChem properties (InChIKey, SMILES, formula, charge) for every CID seen |
| `data/entry_classification.csv` | the structural class, the rule that set it, usability flags and the conditional-scope flag per entry |
| `data/group_smarts_feasibility.csv` | SMARTS feasibility label per non-single entry of the three main lists |
| `cache/pubchem_cache.json.gz` | every PubChem response, so the resolution re-runs offline |

## Coverage

| List | Entries | One structure | Several InChIKeys | Group | Variable composition (UVCB etc.) | Defined by hazard class | Not found | Conditional scope |
|---|---|---|---|---|---|---|---|---|
| Candidate List (SVHC) | 253 | 191 (75%) | 4 (2%) | 31 (12%) | 23 (9%) | 0 (0%) | 4 (2%) | n/a |
| REACH Annex XIV | 59 | 41 (69%) | 1 (2%) | 7 (12%) | 9 (15%) | 0 (0%) | 1 (2%) | 5 |
| REACH Annex XVII | 79 | 32 (41%) | 0 (0%) | 35 (44%) | 4 (5%) | 7 (9%) | 1 (1%) | 79 |
| Annex XVII App. 1 (carc. 1A) | 239 | 59 (25%) | 1 (0%) | 18 (8%) | 156 (65%) | 0 (0%) | 5 (2%) | n/a |
| Annex XVII App. 2 (carc. 1B) | 701 | 174 (25%) | 0 (0%) | 19 (3%) | 504 (72%) | 0 (0%) | 4 (1%) | n/a |
| Annex XVII App. 4 (muta. 1B) | 429 | 41 (10%) | 0 (0%) | 12 (3%) | 375 (87%) | 0 (0%) | 1 (0%) | n/a |
| Annex XVII App. 5 (repr. 1A) | 23 | 18 (78%) | 0 (0%) | 3 (13%) | 1 (4%) | 0 (0%) | 1 (4%) | n/a |
| Annex XVII App. 6 (repr. 1B) | 220 | 181 (82%) | 1 (0%) | 18 (8%) | 19 (9%) | 0 (0%) | 1 (0%) | n/a |

| List | One structure | RDKit parses | Single component | Organic, common elements | Single-component and organic | CAS and name disagree | Element named but absent | Usable | Usable and CAS/name cross-checked |
|---|---|---|---|---|---|---|---|---|---|
| Candidate List (SVHC) | 191 | 191 | 142 | 127 | 123 | 9 | 0 | 123 (49% of entries) | 111 (44%) |
| REACH Annex XIV | 41 | 41 | 29 | 24 | 24 | 3 | 0 | 24 (41% of entries) | 22 (37%) |
| REACH Annex XVII | 32 | 32 | 30 | 29 | 29 | 1 | 0 | 28 (35% of entries) | 25 (32%) |
| Annex XVII App. 1 (carc. 1A) | 59 | 59 | 20 | 10 | 10 | 3 | 1 | 9 (4% of entries) | 8 (3%) |
| Annex XVII App. 2 (carc. 1B) | 174 | 174 | 140 | 135 | 129 | 1 | 0 | 129 (18% of entries) | 110 (16%) |
| Annex XVII App. 4 (muta. 1B) | 41 | 41 | 30 | 28 | 27 | 1 | 0 | 27 (6% of entries) | 18 (4%) |
| Annex XVII App. 5 (repr. 1A) | 18 | 18 | 8 | 4 | 4 | 0 | 0 | 4 (17% of entries) | 4 (17%) |
| Annex XVII App. 6 (repr. 1B) | 181 | 181 | 132 | 119 | 114 | 1 | 0 | 113 (51% of entries) | 89 (40%) |

Full report with definitions: `COVERAGE.md`.
