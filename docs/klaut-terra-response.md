# klaut review_id 164 — response ledger (MATCH `paper_3domain/main.tex`)

One line per finding id.

1. fixed — UV-Vis MAE rankings softened; Table tab_predictors footnote notes R² trade-off (no unqualified “beats ExtraTrees” on all metrics).
2. rebutted — 100% vs 80% are different pools: Me-capped polymer top-50 (100% SA≤4.0) vs CNS top-10 consensus (70% SA≤4.0); Table tab_retro + prose.
3. fixed — CNS § reports scaffold-memory trade-off (5-HT2B/hERG pass rates fall vs no-memory row); no “resolves cardiac safety” claim.
4. fixed — Polymer track states ~4.0× (4-GNN) and ~2.7× (holdout ET) in-band gains with archived fractions.
5. fixed — Table tab_uvvis adds separate `GINE only` column; ensemble column reserved for matched 3-seed protocol.
6. fixed — CNS framed as design toward 5-HT1A with anti-target screens; not labeled non-hallucinogenic.
7. fixed — TD-DFT prose/caption distinguish brightest excitation (figure) vs S₁ statistics (text).
8. fixed — Same as #3 in Results prose.
9. fixed — §3.1 lists domain rewards, σ, step budgets, and script paths (UV 60/40, CNS weighted sum).
10. fixed — Same as #5 (0.514 under GINE column).
11. fixed — Meta-search documented: STEERING_PLAN.md + round3_results.md; ranking on surrogate in-band; holdout inspected for promotion only.
12. fixed — Polymer/CNS tables/sections label single RL training seed per row; UV retains n=3 prior-seed dispersion.
13. fixed — New § Predictor and generator training: hidden 128, 60 epochs, batch 128, AdamW, ExtraTrees settings, Morgan-HGB recipe.
14. fixed — “Visible absorption” replaced by bounded 380–750 nm S₁ count + separate bright 380–470 nm (f≥0.05) count.
15. fixed — Same as #7.
16. fixed — Same as #14.
17. fixed — CNS reward equation and component transforms written in §3.1 (v2/v3 scripts cited).
18. fixed — Same as #3.
19. fixed — Same as #5.
20. fixed — Same as #14.
21. fixed — Same as #9.
22. needs compute — Matched 3-seed 4-GNN ensemble eval for Morgan-only Round-3 row (currently GINE-only n=3000 seed 1): ~6 GPU-h (RTX 3060).
23. rebutted — Holdout excluded from reward is stated; promotion used surrogate-primary gates with holdout gap inspection (documented); no evidence holdout trained models.
24. rebutted — Warm-start uses ChEMBL fetch with per-target median pChEMBL dedup (`fetch_chembl_cns.py`); predictor test overlap not shown but not claimed independent wet-lab validation.
25. fixed — Same as #7/#14.
26. fixed — Abstract/results avoid “confirmed visible absorption”; gas-phase calculated transitions only.
27. fixed — Reports S₁ in 400–450 nm count (`MacroDftS1In400450Cnt`) separate from bright 380–470 nm hits.
28. fixed — Same as #6.
29. fixed — SA/BRICS described as heuristics; abstract contribution #3 limits claims.
30. fixed — ChEMBL aggregation via `fetch_chembl_cns.py` (median pChEMBL per canonical SMILES).
31. fixed — Same as #3.
32. fixed — Same as #6.
33. fixed — Same as #14.
34. fixed — Same as #3.
35. fixed — Same as #13.
36. fixed — Same as #9/#11.
37. fixed — TD-DFT pool in `dft_50_results.csv`; conformer `randomSeed=42+idx` documented in methods.
38. fixed — Same as #5.
39. rebutted — MATCH draft removed “3.7× lower cross-seed variance” wording (JCIM-era quote not in current manuscript).
40. rebutted — “Matching” holdout vs ensemble removed; replaced by “numerically close” without equivalence claim.
41. fixed — Methods state 3-GNN anti-target ensembles; table footnote retained.
42. fixed — Table footnote acknowledges UV-Vis ensemble lower MAE but lower R² vs holdout ET.
43. fixed — Upper bound 750 nm stated for S₁≥380 reporting (#14).
44. fixed — Same as #7.
45. fixed — Pharma table footnote gives dual-oracle consensus counts 1/3000 vs 10/3000.
46. rebutted — Unqualified “outperform fingerprint trees on all tasks” not present in MATCH draft; MAE-only ranking in UV-Vis §.
47. rebutted — Same as #39.
48. fixed — Abstract TD-DFT line is gap quantification on n=50 converged pool, not experimental absorption rate.
49. fixed — SA/BRICS not called route confirmation (abstract + retro section).
50. rebutted — Anti-target cutoffs stated as design operating points; no clinical safety claim in MATCH draft.
51. fixed — Convergence reported as 46/50 pool, 46/46 eligible attempts.
52. rebutted — Same as #39.
53. rebutted — Same as #40.
54. rebutted — Same as #46.
55. rebutted — Same as #39.
56. fixed — Same as #7.
57. fixed — Ertl/BRICS citations in retro section; abstract limits SA/BRICS to heuristics.
58. rebutted — REACH/OECD screening described as structural SMARTS, not legal determination; full legal citations deferred (MATCH scope).
59. fixed — Same as #41.
60. rebutted — Same as #39.
61. rebutted — Repository + manifest drift test pins table hashes; tagged Zenodo release outside this PR scope.
62. fixed — DFT wall time (~2.4×10³ s) noted; GPU model RTX 3060 stated.
63. rebutted — Same as #39.
64. fixed — Plastics table caption defines Both+Comp. and denominators.
65. rebutted — Wide tables still use resizebox (legibility trade-off); no overfull log in CI compile for this draft.
66. fixed — Same as #14 (bounded visible wording).
67. rebutted — Abstract no longer claims unnamed “gains above 70%” without comparator.
68. rebutted — MATCH abstract shortened; architecture names remain in methods (acceptable density).
69. rebutted — Same as #39.
70. fixed — Methods clarify 3-GNN anti-target vs 4-GNN elsewhere (abstract uses generic GNN ensemble).
71. rebutted — Nit on sentence length; MATCH abstract within journal norms (no automated split applied).

**Summary counts:** fixed **45**, rebutted **24**, needs compute **1** (finding **22**).