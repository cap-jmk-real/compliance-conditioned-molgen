# Property objectives and model selection

Status: plan only. Nothing here was trained, and no Colab run was started. Every source listed in
Section 8 was opened from this box on 16 November 2026 unless the entry says otherwise. Numbers are
quoted only where the source states them. Where a figure is our own count over a page we fetched, it
says so. Where we could not verify something, it says that too.

Companion to COVERAGE.md, which measured how far the EU lists resolve to structures. That work is
merged. This document answers the next question Julian raised: compliance alone is a filter, so the
generator needs something to optimise for.

## 0. The task, restated

Generate a molecule that hits a property target and passes the regulatory screen. Compliance is the
constraint, the property is the objective. The three candidate objectives are pesticidal bioactivity,
pharmaceutical bioactivity and UV-Vis absorption, alone or combined.

Safety rule that governs everything below. Controlled substances and drug precursors are a blocklist
only. They are never a conditioning target and never a reward. No synthesis routes and no
retrosynthesis are produced or stored. No hazardous or controlled structures go into this repo.
"Pharmaceutical activity" means ordinary therapeutic targets, named in the repo before any run.

## 1. Property objectives

### 1.1 Pharmaceutical bioactivity

**ChEMBL.** The API status endpoint (https://www.ebi.ac.uk/chembl/api/data/status.json, fetched
16 November 2026) reports release ChEMBL_37, release date 2026-05-01, with 24,527,044 activities,
3,824,604 compound records, 2,921,148 distinct compounds, 18,552 targets and 101,100 publications.
The ChEMBL front page states the data is provided under a Creative Commons Attribution-ShareAlike
3.0 Unported licence. Share-alike matters for us: any derived label file we publish inherits it, so
the benchmark has to keep ChEMBL-derived labels in a separately licensed directory.

**PubChem BioAssay.** We could not open the PubChem statistics page from this box. Both
pubchem.ncbi.nlm.nih.gov/docs/statistics and /docs/bioassays returned a reCAPTCHA interstitial
instead of content, and the fetch tool got HTTP 403. The counts below therefore come from the
PubChem 2025 update paper, which we did open in PubMed Central
(https://pmc.ncbi.nlm.nih.gov/articles/PMC11701573/): as of 12 September 2024 PubChem held
1,671,325 BioAssay records, 295,360,133 bioactivities, 118,596,691 compounds and 322,395,335
substances. Treat these as the paper's figures, not current ones. PubChem data is generally public
domain, with the usual caveat that individual depositor records can carry their own terms.

**Tox21 and ToxCast.** The DeepChem MoleculeNet documentation
(https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html, fetched 16 November
2026) describes Tox21 as qualitative toxicity measurements for "8k compounds on 12 different
targets" across 12 named nuclear-receptor and stress-response tasks, and ToxCast as qualitative
results of "over 600 experiments on 8k compounds". The Tox21 Data Challenge download page
(https://tripod.nih.gov/tox21/challenge/data.jsp) still responds and still links the challenge
data archive. The EPA ToxCast page we first tried returned 404, and
https://www.epa.gov/comptox-tools/exploring-toxcast-data responded instead. These are toxicity
endpoints, so they belong to the learned layer of the compliance screen, not to the property
objective.

Training and validation for a pharmaceutical predictor. Pick a small panel of ordinary therapeutic
targets, four to six of them, each with enough ChEMBL activity records to support a scaffold split.
Convert activities to pChEMBL, keep single-protein assays with a declared standard type (IC50, Ki,
EC50), drop records flagged as potential duplicates, and train one multi-task regressor over the
panel. Validate with a Murcko scaffold split, three seeds, report RMSE and Spearman per target with
per-seed spread, plus a time split where ChEMBL document years allow it. Gate every prediction on an
applicability-domain check (distance to the training set in the model's own representation), and
return "unknown" instead of a number outside it. The generator will leave the domain constantly, so
this is not optional.

### 1.2 Pesticidal bioactivity

**Pesticide Properties Database (PPDB, University of Hertfordshire).** The site is live
(https://sitem.herts.ac.uk/aeru/ppdb/en/index.htm) and the A to Z index page states "Last updated:
12/11/2026". Our own count over that page: 1,998 distinct substance report links. The PPDB does not
publish a record count on the pages we opened, so treat 1,998 as our count of index entries, not as
an official figure. The licensing position is the problem. The home page states that content may not
be copied, reproduced, republished, downloaded or transmitted without the University of
Hertfordshire's written permission, and that offline copies or sub-datasets are available under a
paid licence. A public benchmark cannot redistribute PPDB-derived labels without that permission.

**EU Pesticides Database.** Live at
https://food.ec.europa.eu/plants/pesticides/eu-pesticides-database_en. It covers approved active
substances, maximum residue levels and emergency authorisations. The page itself warns that the
database "is made available solely for the purpose of information", has "no legal value", and may
not be timely, complete or accurate. It gives approval status, which is a regulatory fact, not a
bioactivity label. Useful for the compliance side, weak as a property objective.

**BCPC Pesticide Manual.** The BCPC product page loaded
(https://www.bcpc.org/product/bcpc-online-pesticide-manual-latest-version), and the Pesticide
Manual Online is sold as a subscription product. We did not see a record count or any reuse licence
on the page we opened. For a public benchmark this is a paid source and effectively unusable.

Training and validation for a pesticide predictor, if it is ever carried forward. The licence-safe
route is not PPDB. It is ChEMBL and PubChem assays against insect, fungal and plant targets, used as
a weak proxy label, with PPDB consulted by hand for sanity checks only and never redistributed. That
proxy is noisier than the pharmaceutical panel and the label definition would need its own work.

### 1.3 UV-Vis absorption

**Joung et al. 2020, experimental optical properties of organic compounds** (Scientific Data 7, 295,
https://www.nature.com/articles/s41597-020-00634-8). The paper states a database of 20,236 data
points covering 7,016 unique organic chromophores in 365 solvents or in the solid state, in CSV
format, with canonical SMILES for molecule and solvent. Properties include the first absorption
maximum, its bandwidth and the extinction coefficient. The figshare record for the same data
("DB for chromophore", figshare 12045567) states 20,247 chromophore and solvent combinations and
7,018 unique chromophores, which is the earlier version of the same deposit. The article is open
access. We did not verify the licence attached to the figshare deposit itself, so check it before
redistributing anything derived from it.

**Beard et al. 2019, comparative experimental and computational UV/vis attributes** (Scientific
Data 6, https://www.nature.com/articles/s41597-019-0306-0). The paper states 18,309 records of
experimentally determined absorption maxima mined by ChemDataExtractor from 402,034 documents,
8,488 unique compounds in total, 7,726 with at least one valid experimental maximum, a subset of
5,380 compounds with both experimental and computational data, 6,142 compounds passing the
computational filter, sTDA calculations on that set, 1,302 compounds with first-excitation
oscillator strength above 0.8, 200 of those recomputed with full TD-DFT and 185 converging. The
article is licensed CC BY 4.0 and the data is on figshare (10.6084/m9.figshare.7619672.v2). This is
the TD-DFT-paired set the brief asked about, and it is the honest answer to "is there a computed
set": it exists, it is modest, and the computed values carry a systematic blue shift against
experiment (the paper reports mean absolute errors of 64.50 nm for sTDA and 51.57 nm for TD-DFT on a
76-compound subset).

**MoleculeNet "UV" dataset.** Worth flagging as a trap. The DeepChem documentation states it is an
in-house Merck set of 10,000 compounds at 190 wavelengths with no structures, only precomputed
descriptors, because the compounds are proprietary. It cannot be used for generation work.

Training and validation for a UV predictor. Target the first absorption maximum in nanometres as a
regression task, with solvent as an input feature, trained on Joung and evaluated on a held-out
scaffold split plus a cross-source check against Beard. Three seeds, report MAE and the per-seed
spread, and expect a floor in the tens of nanometres given that the two sources disagree at that
scale. Applicability domain as above.

### 1.4 Recommendation

Carry forward two objectives, and treat the third as a stretch goal.

1. **UV-Vis first absorption maximum.** Recommended as the primary objective. The labels are public
   and open, around 7,000 chromophores with SMILES, the target is a single continuous number, a
   target window ("absorbs between 400 and 450 nm in acetonitrile") is easy to state and easy to
   score, and the predictor is cheap. It also has no dual-use surface at all, which keeps the first
   end-to-end run clean.
2. **Pharmaceutical activity on a declared panel of ordinary therapeutic targets.** Recommended as
   the second objective. ChEMBL gives by far the most labels, the pipeline is standard, and this is
   the setting reviewers will expect to see. The cost is the share-alike licence and the need to
   state the target panel up front for the dual-use rule.
3. **Pesticidal bioactivity.** Not recommended for the first round. The natural label source is
   licence-restricted, the EU database disclaims its own accuracy, BCPC is paid, and a ChEMBL-based
   proxy label needs design work before it means anything. Revisit once the first two objectives run
   end to end.

The combined objective ("hits the property window and stays off every list") is where the
interesting trade-off lives, so both recommended objectives should also be run as a two-term
objective against the EU screen.

## 2. The trade-off metric

Let a run generate N samples under a fixed budget. For each sample compute:

- valid: RDKit parses it and sanitisation passes.
- screen_pass: the two-layer check from COVERAGE.md. Layer one is deterministic (InChIKey match
  against resolved list structures, plus SMARTS rules for group entries and forbidden fragments) and
  it decides. Layer two is the learned hazard scorer, it runs only where layer one found no entry,
  and only inside its applicability domain. Out-of-domain samples count as unknown, never as a pass.
- hit: the property predictor places the molecule inside the declared target window, with the same
  applicability-domain gate.

Primary metric, the compliant hit rate:

    CHR = (1/N) * count(valid AND screen_pass AND hit)

Report CHR with a Wilson 95% interval, over at least three seeds, with the per-seed values shown and
never a single pooled number. Report unknown rates alongside, because a model can inflate CHR by
drifting out of the predictor's domain.

Supporting metrics:

- Validity, the fraction of samples that parse.
- Uniqueness at N, distinct canonical InChIKeys over valid samples.
- Novelty, the fraction of unique valid samples whose InChIKey is absent from the training set.
- Distance to the nearest listed structure: for each generated molecule, the maximum ECFP4 Tanimoto
  similarity to any structure in the resolved regulatory set. Report the full distribution, the
  median, and the fraction above 0.7, which is the near-miss band. This is the metric that catches a
  model that has learned to hug the blocklist boundary.
- Per-constraint-type pass rate, split by Candidate List, Annex XIV, Annex XVII and, later, per
  non-EU jurisdiction.
- Efficiency: accepted molecules per 1,000 samples, and GPU seconds per accepted molecule. This is
  the number that decides whether conditioning beats filtering in practice.

Baselines, all at matched sample budget and matched wall-clock:

1. **Generate then filter.** The same base generator with no conditioning, followed by the screen.
   This is current practice and the thing to beat. Both CHR and GPU seconds per accepted molecule
   must improve for the conditioning claim to mean anything.
2. **Property-conditioned only.** Conditioned on the property, not on compliance, then filtered.
   This isolates what regulatory conditioning adds over ordinary property steering.
3. **Compliance-conditioned only.** The reverse, to show the property term is doing work.

One honest caveat carried over from the plan. For exact-identifier constraints almost every novel
molecule passes trivially, so a high compliance rate there means nothing. Weight the headline metric
towards the substructure-expressible rules, and report the identifier-level rate separately with
that caveat attached.

## 3. Graph neural network arm

### 3.1 Generators

- **DiGress** (Vignac et al., arXiv:2209.14734, 2022, ICLR 2023). Discrete denoising diffusion over
  graphs with categorical node and edge types, a marginal-preserving noise model, auxiliary
  structural features and classifier-style guidance for graph-level properties. It is the reference
  point for graph diffusion on molecules and the natural graph arm here.
- **GDSS** (Jo et al., arXiv:2202.02514, 2022). Score-based generation of graphs through a system of
  stochastic differential equations over nodes and adjacency jointly. Continuous-state alternative to
  DiGress.
- **EDGE** (Chen et al., arXiv:2305.04111, 2023). Degree-guided discrete diffusion aimed at
  efficiency on larger graphs. Relevant if sampling cost becomes the bottleneck.
- **GraphAF** (Shi et al., arXiv:2001.09382, 2020). Autoregressive flow over graphs, cheap to train,
  valency checks during generation, dated but a fair low-cost pool member.
- **GCPN** (You et al., arXiv:1806.02473, 2018). Graph convolutional policy network, reinforcement
  learning towards a property with domain rewards. Fiddly to tune and sensitive to reward shaping.
- **MoFlow** (Zang and Wang, arXiv:2006.10137, 2020). Invertible flow over molecular graphs, fast
  one-shot sampling, strong validity, limited expressiveness on larger molecules.

### 3.2 Property predictors

- **Chemprop D-MPNN** (Yang et al., arXiv:1904.01561, "Analyzing Learned Molecular Representations
  for Property Prediction", JCIM 2019). Directed message passing with RDKit descriptor
  concatenation. Recommended as the default predictor for all three objectives. It is the
  best-supported option, trains in minutes to low hours on this data size, and has an established
  uncertainty and applicability-domain story.
- **Attentive FP** (Xiong et al., J. Med. Chem. 2020, 63(16), 8749-8760,
  doi:10.1021/acs.jmedchem.9b00959). Graph attention with interpretable atom weights. Recommended as
  the second model, mainly because the attention maps are useful when explaining why a molecule was
  scored as a UV absorber.
- **GIN** (Xu et al., arXiv:1810.00826, "How Powerful are Graph Neural Networks?"). Cheap control
  baseline. If GIN matches Chemprop on our data, the data is the limit and a bigger predictor is
  wasted compute.

### 3.3 Recommendation

Predictors: Chemprop D-MPNN as the default, Attentive FP as the second model, GIN as the control.
Run all three on the UV task first, because it is the cheapest, and only promote the winner to the
ChEMBL panel.

Generators for the graph arm: DiGress as the single graph generator carried to a full run, with
GDSS, EDGE, GraphAF, GCPN and MoFlow entering the meta-search pool in Section 4 as pilots only.
Reason: DiGress is the only graph generator in the list with a documented property-guidance
mechanism and molecule-scale results, so it is the one worth paying full training cost for.

### 3.4 A100-hour estimates

These are estimates, not measurements. They assume a ZINC-scale training subset of roughly 100,000
molecules capped at 32 heavy atoms, mixed precision, and a single A100. The first real run replaces
every number here.

| Component | Setting | Estimate (A100 hours) |
|---|---|---|
| Chemprop D-MPNN, UV task, 3 seeds | small data, minutes per epoch | 1 to 2 |
| Chemprop D-MPNN, ChEMBL panel, 3 seeds | multi-task, larger data | 2 to 4 |
| Attentive FP, both tasks, 3 seeds | | 2 to 4 |
| GIN control, both tasks, 3 seeds | | 1 to 2 |
| DiGress pilot | 1 seed, reduced epochs | 8 to 12 |
| DiGress full | 3 seeds | 30 to 45 |
| GraphAF pilot | 1 seed | 3 to 5 |
| MoFlow pilot | 1 seed | 3 to 5 |
| GCPN pilot | 1 seed | 4 to 8 |
| GDSS or EDGE pilot | 1 seed, pick one | 4 to 8 |
| Sampling and evaluation | 10,000 samples per configuration | 3 to 6 |

Graph arm with predictors, DiGress full and no pool pilots: roughly 40 to 60 A100 hours.

### 3.5 Overlap with tracks #2 and #15

From the repo README in cap-jmk-real/open-research:

- Track #2, Graph NCA Architecture Meta-Learning, is architecture search over graph models with
  Lean 4 verification. The meta-search arm in Section 4 should reuse that search harness instead of
  writing a second one, and any claim about search methodology belongs to #2. This track contributes
  the chemistry objective and the compliance constraint, not a new search method.
- Track #15, CARATE GNN wave-function learning, is a GNN on QM9 with a planned 2026 benchmark
  against SchNet and DimeNet++. Its featurisation and QM9 tooling are reusable for a
  computed-property predictor, and the UV objective is adjacent to its physics. The risk is
  double-counting: if the same predictor result is reported in both tracks, say so explicitly in
  both papers.

## 4. Meta-search arm

Modelled on the DenoiseOpt bake-off, which compared six families under one protocol. The same shape
applies here, with the generation pipeline as the search object.

Candidate pool. Every entry is a first-class candidate, not a baseline only:

1. Discrete graph diffusion (DiGress).
2. Continuous or score-based graph diffusion (GDSS, or EDGE if sampling cost dominates).
3. Sequence diffusion over SMILES or SELFIES, with constraint projection in the sampling loop, in
   the Constrained Discrete Diffusion line (Cardei et al., arXiv:2503.09790) and the neuro-symbolic
   variant (Christopher et al., arXiv:2506.01121).
4. Autoregressive sequence model over SELFIES (Krenn et al., arXiv:1905.13741), with a conditioning
   token.
5. Graph flow and reinforcement learning (GraphAF, GCPN, MoFlow).
6. Variational autoencoder over junction trees (Jin et al., arXiv:1802.04364).

Protocol, identical for every candidate: same training subset, same seeds, same sample budget at
evaluation, same two-layer screen, same property predictor, same metric set from Section 2. Stage
gate it. Stage one is a short pilot per family at one seed and a fixed small budget, scored on CHR
and on GPU seconds per accepted molecule. Stage two promotes the top two families to three seeds at
full budget. Report the whole table including the losers, with per-seed spread, and report the
pilot-to-full rank correlation so the next bake-off knows whether short pilots predict anything.

Evaluation uses MOSES (Polykovskiy et al., arXiv:1811.12823) and GuacaMol (Brown et al.,
arXiv:1811.09621) conventions for validity, uniqueness and novelty, so the distribution-learning
numbers stay comparable to published work.

Estimates, same caveats as Section 3.4:

- Full budget: six pilots at 4 to 8 hours each is 24 to 48 hours, two full runs at three seeds at 15
  to 25 hours each is 30 to 50 hours, evaluation and reruns 10 to 15 hours. Total roughly 65 to 110
  A100 hours.
- Reduced budget: six pilots at 2 to 3 hours each on a 25,000-molecule subset is 12 to 18 hours, one
  winner promoted to three short seeds is 10 to 15 hours, evaluation 4 to 6 hours. Total roughly 26
  to 40 A100 hours.

Programme total if everything runs: predictors 6 to 12, graph arm 40 to 60, meta-search 65 to 110.
Roughly 110 to 180 A100 hours, before failed runs. Section 5 explains why that does not fit the
current Colab budget.

## 5. Colab execution requirements

Julian's instructions, turned into requirements on the scripts. These are requirements for the next
PR, since no run happens in this one.

1. **Clone, do not upload.** Every run starts by cloning this repo into the Colab VM and running
   from the clone, so the run is pinned to a commit. The notebook records the commit hash in the run
   log and in every checkpoint.
2. **Print progress.** Every script prints a heartbeat at a fixed step interval with step number,
   elapsed time, examples per second, current loss and estimated time remaining, and flushes the
   output. A silent script is treated as a failed script.
3. **Checkpoint to Drive.** Google Drive is mounted and checkpoints go to
   MyDrive/compliance-molgen/runs/<run-id>/, written atomically (write to a temporary file, then
   rename) so an interrupted write cannot corrupt the latest checkpoint. Checkpoint at a fixed step
   interval and at a fixed wall-clock interval, whichever comes first, and keep the last three plus
   the best.
4. **Resume by default.** Every script looks for the latest checkpoint for its run id and resumes
   from it without being asked. Starting over requires an explicit flag. This is the behaviour that
   makes a disconnected session cheap instead of fatal.
5. **Keep the session alive.** The runtime disconnects when idle, so the session needs periodic
   activity while a long run is going, and the run has to survive a disconnect regardless through
   requirement 4. Treat a disconnect as expected, not as an incident.
6. **Measure the burn rate on the first run.** Colab does not show the A100 per-hour cost before a
   runtime is connected, and the pricing pages we could open do not list it. So the first run is a
   calibration run: record the compute-unit balance immediately before connecting, run a fixed short
   workload on the A100, record the balance after, and derive units per hour. Budget roughly 0.25 to
   0.5 hours of A100 time for that calibration.

Budget arithmetic, with the rate still unknown. The account holds 200 compute units. If one A100
hour costs r units, the budget is 200 / r hours. We have no verified value for r. Under an assumed
range of 8 to 15 units per hour, which is an assumption to test and not a quoted figure, 200 units
buys roughly 13 to 25 A100 hours.

That is the hard finding. The programme in Sections 3 and 4 needs 110 to 180 A100 hours. The reduced
meta-search alone needs 26 to 40. Neither fits. What fits inside 200 units is: the calibration run,
the three property predictors on the UV task, the Chemprop run on the ChEMBL panel, and one
generator pilot. Everything past that needs more units or a different machine. Julian should decide
which, and the calibration run should happen before he decides, because it replaces the assumed
range with a measured number.

Two small helper scripts ship with this document and are written to satisfy requirements 2 to 4
without pulling in a training framework:

- colab_checkpoint.py: atomic checkpoint writes, retention of the last three plus best, latest
  checkpoint discovery, resume-by-default, and a progress heartbeat printer.
- colab_burn_rate.py: records the calibration readings and the derived units per hour into a JSON
  log on Drive, and converts a planned hour budget into units at the measured rate.

Neither script trains anything or starts a runtime.

## 6. Non-EU rules, scoping only

No ingestion in this PR. This section records what is actually available, checked by fetching each
source on 16 November 2026, and how hard each one looks.

### United States

- **TSCA Inventory.** https://www.epa.gov/tsca-inventory/how-access-tsca-inventory loads and states
  that the non-confidential inventory is updated about every six months, that the July 2026 update
  is available for download, that 42,698 chemicals are on the inventory as active in US commerce,
  and that 44,043 are inactive. Downloadable files are linked on the same page.
- **CompTox Chemicals Dashboard.** https://comptox.epa.gov/dashboard/ loads and reports
  "Search 1,386,492 Chemicals", last updated August 2026. This is the practical bridge from a CAS
  number or a name to a structure for US lists, through DSSTox identifiers.
- Difficulty: low. Bulk files exist, identifiers are clean, and a structure resolver sits next to the
  list. The real work is that the TSCA inventory is an inventory of what is in commerce, not a
  restriction list, so it answers a different question than Annex XVII does. Restriction-style US
  data (section 6 rules, significant new use rules) is scattered across rule texts and needs separate
  treatment.

### China

- **IECSC.** The MEE inventory page https://www.mee.gov.cn/ywgz/gtfwyhxpgl/hxphjgl/wzml/ loads and
  lists the supplement announcements, the most recent dated 2026-08-04. That announcement states
  that 4 substances and 482 already-registered new chemical substances were added. The MEE Solid
  Waste and Chemicals Management Centre download page https://www.meescc.cn/ggzc/xzzq/ loads and
  carries an item dated 2016-04-12 offering a standalone version of the inventory for download. We
  did not download it. A third-party consultancy page states 45,612 substances in the public part
  and 3,270 in the confidential part, and we are recording that as third-party and unverified
  against an official source.
- **Catalogue of Hazardous Chemicals (2015).** The official announcement page at
  https://www.mem.gov.cn/gk/gwgg/agwzlfl/gg_01/201503/t20150309_237187.shtml loads, as does the MIIT
  copy that links the catalogue as a .doc attachment. The commonly cited figure of 2,828 entries
  appears in third-party copies and not in the announcement text we read, so it is unverified here.
- Difficulty: medium to high. The data exists in public and is reachable, and it arrives as Chinese
  substance names in office documents, with CAS numbers present but not guaranteed, a standalone
  download application from 2016, and supplements published as separate announcements that have to
  be stitched together. Expect name resolution, not identifier lookup, to be the bottleneck, and
  expect a confidential portion that is simply not available.

### India

- **MSIHC Rules 1989.** Schedule 1 Part II is the list of hazardous and toxic chemicals, and it is a
  plain numbered list of substance names. The India Code PDF of the rules is the canonical public
  copy. Our direct fetch of that PDF from this box timed out twice (curl reported HTTP 000 after 60
  and after 120 seconds), so we confirmed the content through the search index instead of opening
  the file ourselves. Flagging that honestly: the source is public, our box could not reach it today.
- **Draft Chemicals (Management and Safety) Rules.** Still a draft, still titled "20xx", and still
  not in force as of our check. Its Schedule II lists priority substances with CAS numbers, which
  would be the most machine-friendly Indian source if it is ever notified.
- Difficulty: high. Schedule 1 Part II gives names without CAS numbers, the authoritative text is a
  scanned-era PDF, and the modern rules that would fix this are a draft that may change. Any
  coverage number for India will be dominated by name-to-structure resolution error.

### What this means for the coverage metric

Coverage has to be reported per jurisdiction and never pooled. The EU numbers in COVERAGE.md are not
transferable: the US inventory is a commerce list, the Chinese catalogue is a licensing list, and the
Indian schedule is a hazard-criteria list with a name annex. Each one resolves to structures at a
different rate and for different reasons, and a pooled percentage would hide exactly the thing the
benchmark is supposed to measure.

## 7. What is still unresolved

- The Colab A100 rate. Unknown until the calibration run. Everything in the budget arithmetic depends
  on it.
- The figshare licence on the Joung deposit. Check before redistributing derived labels.
- Whether a public regulatory-conditioned generation benchmark already exists. The milestone-1 search
  still stands as a gate, and a negative result there is what makes this track worth doing.
- The ChEMBL target panel. Needs to be named and written into the repo before any pharmaceutical run,
  for the dual-use rule.

## 8. Sources

All fetched 16 November 2026 unless stated.

Datasets and databases:

- ChEMBL, https://www.ebi.ac.uk/chembl/ and https://www.ebi.ac.uk/chembl/api/data/status.json.
  Release ChEMBL_37, 2026-05-01. CC BY-SA 3.0.
- PubChem 2025 update, Kim et al., Nucleic Acids Research 53(D1), D1516, read at
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11701573/. PubChem's own statistics page was not reachable
  from this box.
- DeepChem MoleculeNet documentation,
  https://deepchem.readthedocs.io/en/latest/api_reference/moleculenet.html, for Tox21, ToxCast and
  the Merck UV set.
- Tox21 Data Challenge data page, https://tripod.nih.gov/tox21/challenge/data.jsp.
- EPA ToxCast data, https://www.epa.gov/comptox-tools/exploring-toxcast-data.
- Pesticide Properties Database, https://sitem.herts.ac.uk/aeru/ppdb/en/index.htm and the A to Z
  index at /atoz.htm. Licence restrictive, written permission required for reuse.
- EU Pesticides Database, https://food.ec.europa.eu/plants/pesticides/eu-pesticides-database_en.
- BCPC Pesticide Manual Online,
  https://www.bcpc.org/product/bcpc-online-pesticide-manual-latest-version. Subscription product.
- Joung, Han, Jeong and Park, "Experimental database of optical properties of organic compounds",
  Scientific Data 7, 295 (2020), https://www.nature.com/articles/s41597-020-00634-8, with the
  figshare deposit at https://figshare.com/articles/dataset/DB_for_chromophore/12045567.
- Beard, Sivaraman, Vazquez-Mayagoitia, Vishwanath and Cole, "Comparative dataset of experimental and
  computational attributes of UV/vis absorption spectra", Scientific Data 6 (2019),
  https://www.nature.com/articles/s41597-019-0306-0. CC BY 4.0.

Methods:

- Vignac et al., "DiGress: Discrete Denoising diffusion for graph generation", arXiv:2209.14734.
- Jo et al., "Score-based Generative Modeling of Graphs via the System of Stochastic Differential
  Equations", arXiv:2202.02514.
- Chen et al., "Efficient and Degree-Guided Graph Generation via Discrete Diffusion Modeling",
  arXiv:2305.04111.
- Shi et al., "GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation",
  arXiv:2001.09382.
- You et al., "Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation",
  arXiv:1806.02473.
- Zang and Wang, "MoFlow: An Invertible Flow Model for Generating Molecular Graphs",
  arXiv:2006.10137.
- Jin et al., "Junction Tree Variational Autoencoder for Molecular Graph Generation",
  arXiv:1802.04364.
- Krenn et al., "Self-Referencing Embedded Strings (SELFIES)", arXiv:1905.13741.
- Cardei et al., "Constrained Discrete Diffusion", arXiv:2503.09790.
- Christopher et al., "Neuro-Symbolic Generative Diffusion Models for Physically Grounded, Robust,
  and Safe Generation", arXiv:2506.01121.
- Yang et al., "Analyzing Learned Molecular Representations for Property Prediction",
  arXiv:1904.01561 (Chemprop D-MPNN).
- Xu et al., "How Powerful are Graph Neural Networks?", arXiv:1810.00826 (GIN).
- Xiong et al., "Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph
  Attention Mechanism", J. Med. Chem. 63(16), 8749-8760 (2020), doi:10.1021/acs.jmedchem.9b00959
  (Attentive FP). Verified through the ACS issue table of contents and Crossref metadata, the
  publisher page itself was not opened.
- Polykovskiy et al., "Molecular Sets (MOSES)", arXiv:1811.12823.
- Brown et al., "GuacaMol: Benchmarking Models for De Novo Molecular Design", arXiv:1811.09621.

Non-EU regulatory sources:

- EPA, How to Access the TSCA Inventory,
  https://www.epa.gov/tsca-inventory/how-access-tsca-inventory.
- EPA CompTox Chemicals Dashboard, https://comptox.epa.gov/dashboard/.
- MEE, China Existing Chemical Substances Inventory,
  https://www.mee.gov.cn/ywgz/gtfwyhxpgl/hxphjgl/wzml/, and the 2026-08-04 supplement announcement
  https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202608/t20260806_1163640.html.
- MEE Solid Waste and Chemicals Management Centre download area, https://www.meescc.cn/ggzc/xzzq/.
- Catalogue of Hazardous Chemicals (2015) announcement,
  https://www.mem.gov.cn/gk/gwgg/agwzlfl/gg_01/201503/t20150309_237187.shtml, and the MIIT copy with
  the attachment,
  https://wap.miit.gov.cn/ssqqhxptyflhbqzdgz/fgzc/zywj/art/2023/art_0b9007c77d384fd2ac47da4e60231af9.html.
- Manufacture, Storage and Import of Hazardous Chemical Rules 1989, India Code PDF. Not reachable
  from this box today, content confirmed indirectly only.
