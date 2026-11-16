# Which group entries could be SMARTS rules

This note covers the entries of the Candidate List, Annex XIV and Annex XVII that do not resolve to exactly one structure. The per-entry labels are in `data/group_smarts_feasibility.csv`, produced by `smarts_feasibility.py`. The labels come from keyword rules on the entry name. I then read every labelled name and added rules where the first pass was wrong. They are a reading of the entry text, not a legal interpretation.

## Counts

| List | Entries not resolving to one structure | single_not_in_pubchem | element_rule | parent_plus_forms | family_pattern | partial | partial_pah | not_structural | unlabelled |
|---|---|---|---|---|---|---|---|---|---|
| Candidate List | 62 | 2 | 0 | 27 | 2 | 1 | 0 | 30 | 0 |
| Annex XIV | 18 | 1 | 0 | 4 | 2 | 1 | 0 | 10 | 0 |
| Annex XVII | 47 | 0 | 7 | 12 | 9 | 5 | 2 | 12 | 0 |

Labels:

- `element_rule`: "X compounds" or "X and its compounds" (cadmium, nickel, lead, arsenic, mercury), organotin, inorganic ammonium salts. A one-atom pattern such as `[Cd]` or an element-carbon bond such as `[Sn]~[#6]` expresses the structural part.
- `parent_plus_forms`: one parent structure with its salts, isomers, stereoisomers or hydrates. A parent-skeleton pattern with charges and stereochemistry left open covers it.
- `family_pattern`: a functional family, for example PFAS, phthalate esters, diisocyanates, polychlorinated terphenyls, polybrominated biphenyls, cyclic siloxanes.
- `partial`: only part of the entry is structural. Azo colourants are restricted by the amines they release on reductive cleavage. "Related substances" and "releasers" are defined by what they degrade to. Chromium VI needs an oxidation state, which SMARTS cannot state.
- `partial_pah`: the PAH entries name eight specific PAHs. A fused-ring pattern would match far more than the entry covers, so the right encoding is the explicit list.
- `not_structural`: UVCB substances, reaction masses, mineral fibres, polymers, ethoxylates of variable chain length and hazard-class entries. These need composition data or CLP classifications, which a structure does not carry.
- `single_not_in_pubchem`: one defined substance that PubChem did not return.

## Worked examples

`smarts_rules.py` holds five commented rules. `tests/test_smarts_rules.py` checks each against structures that must match and structures that must not (35 cases, all passing with RDKit 2026.9.2).

| Rule | Covers | Pattern idea | Known limits |
|---|---|---|---|
| `ortho_phthalate_diester` | Annex XVII 51 and 52, Annex XIV 4 to 7, Candidate List phthalates | benzene-1,2-dicarboxylic acid with both carboxyls esterified on carbon | Wider than any single entry, which names specific esters. Terephthalates and monoesters do not match. |
| `pfas_oecd_2021` | Annex XVII 82, and the PFAS idea behind 68, 73, 79 | at least one sp3 carbon with no H, Cl, Br or I and at least two F (OECD 2021 definition) | Very broad: trifluoroacetic acid and CF3-containing drugs match. Entry 82 applies only to firefighting foams, which no structure rule can say. |
| `pfca_c9_c14` | Annex XVII 68 | carboxyl followed by at least eight perfluorinated carbons, plus a chain-length count in code for the C14 upper bound and branches | SMARTS alone cannot express the upper bound. Precursors ("related substances") are not covered. |
| `organotin` | Annex XVII 20 | `[Sn]~[#6]` | Inorganic tin does not match, as intended. |
| `benzidine_and_salts` | Annex XVII 13 | biphenyl-4,4'-diamine with neutral or protonated amines | Substituted benzidines also match unless ring hydrogens are constrained. |

## What this means for a rule layer

Most group entries in the Candidate List and Annex XIV are a parent plus its salts or isomers. Those are easy. Annex XVII is harder: its group entries are dominated by element-defined groups, which are trivial to match but describe mostly inorganic chemistry, and by entries whose scope is set by use, concentration or degradation. A structure-only checker can flag "this molecule belongs to a restricted family". It cannot decide whether a restriction applies.
