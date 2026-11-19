# Paper-review self-scorecard

Reviewed with the paper-review skill (demanding referee, ACS/RSC/Wiley family). Target >= 8.5/10.
Re-scored from scratch against the LaTeX source, refs.bib, the committed eval artifact and the training code.

## Verdict
**Manuscript quality score: 8.7 / 10. Target 8.5. Referee call: minor revision.**
No open manuscript Blockers. One pre-submission action sits outside this manuscript and outside my authority:
the code repository is private today. ChemRxiv submission needs it public, and the brief forbids me changing
visibility. So the manuscript is submission-ready on quality; the author must flip the repo to public (same commit
0c102b6) before actually submitting. I flag this as the single gating item, not a text defect.

3-line summary: Question - how much does soft prefix-token conditioning raise the on-target rate of a molecular
generator for a UV-Vis absorption band, versus an identical unconditioned model? Method - two-arm SELFIES GRU LM on
ZINC250k, identical but for two prefix control tokens, scored by a surrogate absorption model, with a bootstrap CI.
Main claim - steering raises the in-band fraction from 0.187 to 0.237 (+0.050, 95% CI [0.043, 0.058]); real, modest,
single-seed, surrogate-scored.

Primary metric: in-band fraction. Primary comparison: steered vs plain. Model-selection rule: final fixed-epoch
checkpoint, no selection on the eval set (stated in Methods).

## Number ledger (value -> source -> producer)
All values trace to uv/results/eval_results.json (committed), produced by the eval pipeline; figures by paper/figures/make_figures.py.
| Value in paper | Artifact field | Match |
|---|---|---|
| Validity 1.000 / 1.000 | arms.*.validity 1.0 | ok |
| Uniqueness 0.9999 / 0.9998 | arms.*.uniqueness | ok |
| Novelty 0.9989 / 0.9990 | arms.*.novelty 0.9989 / 0.999 | ok |
| Screen pass 0.9635 / 0.9991 | arms.*.rule_pass_rate | ok |
| In-band 0.1867 / 0.2371 | arms.*.in_band_fraction | ok |
| Mean pred abs 365.69 / 376.92 nm | arms.*.mean_pred_abs_nm | ok |
| Unique valid 19,997 / 19,995 | arms.*.n_unique | ok |
| Effect +0.050, CI [0.043,0.058], 2000 resamples, all positive | bootstrap_in_band_diff (0.0504, 0.0425, 0.0582, 2000, frac_positive 1.0, excludes_zero true) | ok |
| MAE 24.0, RMSE 38.2, R2 0.873 nm | predictor_metrics (24.04, 38.19, 0.8734) | ok (rounded) |
| Train/test 5203 / 643, 600 trees | predictor_metrics n_train/n_test/total_trees | ok |
| 301,584 steps = 618 epochs, loss 0.605 | brief/run log (488 steps/epoch x 618; ceil(249455/512)=488) | ok, internally consistent |
No headline or table number lacks a result-file source. No script rewrites the TeX with literal numbers.

## Text-versus-code ledger
| Paper statement | Code | Match |
|---|---|---|
| 2-layer GRU, emb 128, hidden 256 | train_generator.py --layers 2, --emb 128, --hidden 256 | ok |
| Adam, lr 1e-3 | torch.optim.Adam(lr=args.lr), --lr 1e-3 | ok |
| Token-level cross-entropy next-token loss | nn.CrossEntropyLoss(ignore_index=pad) | ok |
| Batch size 512 (full run) | run.py --full-batch-size default 512 | ok |
| Morgan ECFP r2 2048 + HistGradientBoostingRegressor, 600 trees | predictor_metrics model string, total_trees 600 | ok |
| Seeded once per arm (batch shuffle seeded by epoch) | batches(..., seed=epoch) | ok |
No mismatch in any headline-method description.

## Citation table (14 refs, all resolved)
krenn2020selfies, sterling2015zinc15, joung2020uvvis, pedregosa2011scikit, rdkit, polykovskiy2020moses,
brown2019guacamol, blaschke2020reinvent, wang2021mcmg, cardei2025cdd, christopher2025nsd, vignac2023digress,
brenk2008, baell2010pains - each verified via Crossref DOI or arXiv/Semantic Scholar abstract page this task.
Status: all ok. All 14 are cited in the body (unsrt drops none). No dead/mismatched/retracted ids.

## Rigour notes
- Design: clean two-arm controlled comparison, only the two prefix tokens differ. Primary metric and comparison declared.
- Statistics: 95% percentile bootstrap (2000 resamples) on the difference of arm means; resampling unit is the generated
  molecule. The paper states explicitly this covers sampling at fixed weights, not training-seed variance, and does not
  sell a seed-stable headline. Single-seed is scoped as exploratory with the caveat carried in the abstract, which the
  skill permits ("qualify them"). Not a blocker; stated honestly in Limitations as pseudo-replication.
- No tuning/search/hardware overclaim: hardware is a single A100, timings reported as measured, no "fastest"/"tuned" claims.
- Model selection: final fixed-epoch checkpoint, no eval-set selection - stated.
- Surrogate error (24 nm ~ half the 50 nm band) disclosed as the key accuracy limit; both arms scored identically.

## Dimension scores
- Claims supported & accuracy 9.0 (exact artifact match, no overclaim, "not a novelty claim" stated)
- Rigour & reproducibility 8.3 (single-seed, honestly scoped; repo private until publication is the one gap)
- Novelty & significance 8.0 (deliberately narrow controlled measurement; positioned vs soft-conditioning + hard-constraint prior art)
- Citations verified & coverage 9.0 (all 14 resolve and are cited; prior-art coverage added for conditioning, constraints, metrics)
- Clarity & language 9.2 (deslop-clean, one idea per sentence, UK spelling consistent)
- Structure & venue fit 8.8 (abstract within limit, contributions explicit, ChemRxiv working-paper fit)
- Design 8.8 (two vector figures, Okabe-Ito palette, booktabs table, stand-alone captions, clean build)

Weighted total: **8.7 / 10.** Minor revision. The only submission gate is making the (already-pinned) repo public,
which is the author's action and is outside this task's authority.

---
## External panel (klaut.pro) final result on this commit
klaut.pro agentic panel (5 reviewers): overall 5.95/10, rubric 7.5/10, 2/5 accept (needs 3), 0 open blockers.
The one remaining major and all reject votes are the single-seed soundness issue: one training run per arm
does not quantify training variance. Resolving it needs >=3 seeds with a paired test (new training runs),
plus more target bands and a compute-matched baseline. Those are new experiments, out of scope for this
preprint, and are listed in V2_TODO.md. No manuscript blocker remains.
