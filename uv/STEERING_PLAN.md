# Stronger steering and a steering meta-search (track #25, UV-Vis)

Status: code and unit tests in this branch (`exp/steering-metasearch`), CPU smoke run done,
no GPU run yet. Nothing here changes the method or the numbers of the current preprint
(paper PR #5): the preprint reports prefix-token steering with n = 3 seeds, and that stays
the reported method until a stronger one has its own three-seed result.

## 1. How the current steering works

Code: `uv/gen_model.py`, `uv/train_generator.py`, `uv/prepare_corpus.py`, `uv/evaluate.py`.

1. `prepare_corpus.py` runs the fixed seed-0 UV surrogate (Morgan 2048 bits into a
   HistGradientBoostingRegressor, held-out MAE 24.0 nm, R2 0.873 on its own split) over every
   ZINC250k molecule and stores `abs_pred` and the structural-screen flag `passes_screen`.
2. `gen_model.build_sequence` turns each corpus row into
   `<bos> <BAND_IN|BAND_OUT> <CLEAN|FLAG> SELFIES... <eos>`. The band token is
   `abs_pred in [400, 450]`. The plain arm gets no control tokens.
3. `train_generator.py` trains a 2-layer GRU (emb 128, hidden 256) with ordinary
   next-token cross-entropy. The two control tokens are simply the first two targets of
   every sequence; the loss has no term that rewards using them.
4. `evaluate.py` samples the steered arm after the prefix `<BAND_IN> <CLEAN>` at
   temperature 1 and the plain arm with no prefix, then scores unique valid molecules with
   the same surrogate. That is the whole mechanism: one conditioning prefix, no strength
   parameter, the surrogate used only to label the corpus and to score samples.

## 2. Why the effect is modest

Measured (seed 1, 20,000 samples per arm, from the engine run): in-band fraction 0.190
plain vs 0.249 steered; rule pass 0.964 vs 0.9995; validity 1.0 both. Measured on the
full seed-0 corpus (249,455 molecules): 14.7 % of the corpus is in the 400 to 450 nm band by
the surrogate, 96.1 % passes the screen, and the mean SELFIES length is 37.5 tokens (first
20,000 rows). The reasons below are what the code and those numbers imply. (a) and (b) are
arithmetic on measured values; (c) to (e) are mechanisms we have NOT isolated by an
experiment, so they are hypotheses for the meta-search to test.

a. **The label carries little of the training signal.** A binary label with p = 0.147 has
   entropy 0.418 nats (0.60 bits). That is the most the band token can add to the
   likelihood of a sequence. A sequence costs roughly 0.62 nats per token (the logged
   training loss of the seed runs) times about 38 tokens, i.e. about 24 nats. The band
   token is at most about 2 % of what the loss can gain, so plain next-token training gives
   the model little reason to build a strong band-to-structure mapping.
b. **No dial.** Sampling is from p(x | IN, CLEAN) exactly. Whatever the model learned is
   the strength; nothing lets us trade diversity for in-band yield.
c. **The label is a global, non-smooth function of the whole molecule** (a fingerprint
   regressor), but the model sees it as one token at the start of a long sequence. A GRU
   has to carry that bit through ~38 steps and learn which substructures it implies.
d. **The corpus is not a chromophore corpus.** ZINC250k is drug-like. The in-band
   molecules in it are the ones the surrogate extrapolates for, not dyes, so p(x | IN) may
   sit on surrogate artefacts rather than on real absorbers.
e. **The surrogate is only used as a labeller.** Nothing optimises the generator against
   it, so the generator never sees the surrogate's gradient-free feedback on its own samples.

The consequence for method choice: stay compatible with the existing checkpoints and add
(i) a strength knob that needs no training, and (ii) objectives that touch the property
directly instead of through one label token.

## 3. Methods implemented (`uv/steering.py`)

All three run from the existing plain and steered checkpoints, the same vocabulary and the
same fixed surrogate. Selection among them is the job of the meta-search (Section 4); we do
not claim any of them works until it has a three-seed result.

| method | what it does | knobs | why it fits |
|---|---|---|---|
| `guidance` | Sampling-time logit guidance. Next-token log-probs of several (model, prefix) experts are combined linearly: `(1+w) logp(.|IN,CLEAN) - w logp(.|contrast)`. `contrast = bandout` uses the steered model with `<BAND_OUT>` (band token only differs); `contrast = plain` uses the plain model. | w, contrast, temperature | No training at all, uses the two checkpoints we already have, gives the missing dial. w = 0 reproduces prefix steering (unit-tested). Special tokens are masked while sampling so extrapolation cannot emit control tokens. |
| `reinvent` | REINVENT-style fine-tuning of an agent initialised from the plain or the steered model, with the same checkpoint frozen as prior. Loss `(logP_prior + sigma * S - logP_agent)^2`, `S = 2*reward - 1`. The prior term is the KL/prior penalty. Optional soft band reward, optional top-k (augmented hill-climb), optional identical-molecule bucket. | sigma, lr, batch, soft_width, topk_frac, bucket, init | The standard sequence-model RL baseline for property steering; it optimises the in-band indicator directly (point (ii) above). It also reports a sample estimate of KL(agent, prior) in nats per trial so drift is visible. |
| `selftrain` | Surrogate-filtered self-training (expert iteration): sample, keep unique valid molecules the surrogate puts in the band and that pass the screen as a data filter, fine-tune on them mixed with corpus replay, repeat. | steps_per_round, gen_per_round, replay_frac, lr, init | Needs no RL machinery, replay keeps it near the prior, cheap per step. The screen is a hard filter on training data only. |

Not implemented, with reasons: classifier guidance at sampling time needs a property
predictor on partial SELFIES prefixes (the surrogate scores finished molecules only), so a
new partial-sequence model would have to be trained first; the graph generators of the
design document are discussed in Section 4.

### Citations (each checked to exist on 2026-11-17)

- Olivecrona, Blaschke, Engkvist, Chen. *Molecular de-novo design through deep
  reinforcement learning.* J. Cheminform. 9:48 (2017), arXiv:1704.07555,
  doi:10.1186/s13321-017-0235-x. (REINVENT, the augmented-likelihood loss.)
- Blaschke, Arús-Pous, Chen, Margreitter, Tyrchan, Engkvist, Papadopoulos, Patronov.
  *REINVENT 2.0: An AI Tool for De Novo Drug Design.* J. Chem. Inf. Model. 60 (2020),
  doi:10.1021/acs.jcim.0c00915. (Diversity filter, practical use.)
- Thomas, O'Boyle, Bender, de Graaf. *Augmented Hill-Climb increases
  reinforcement learning efficiency for language-based de novo molecule generation.*
  J. Cheminform. 14:68 (2022), doi:10.1186/s13321-022-00646-z. (The `topk_frac` option.)
- Ho, Salimans. *Classifier-Free Diffusion Guidance.* arXiv:2207.12598 (2022). (The
  conditional-minus-unconditional extrapolation.)
- Sanchez, Fan, Spangher, Levi, Ammanamanchi, Biderman. *Stay on topic with
  Classifier-Free Guidance.* arXiv:2306.17806 (2023). (The same guidance on autoregressive
  language models.)
- Liu, Han, Wang, Tsvetkov, Choi, Smith. *Tuning Language Models by Proxy.* arXiv:2401.08565
  (2024). (Logit-space combination of a small tuned and untuned model, used for
  `contrast = plain`.)
- Gulcehre et al. *Reinforced Self-Training (ReST) for Language Modeling.*
  arXiv:2308.08998 (2023). (Sample, filter by reward, fine-tune, iterate.)
- Gao, Fu, Sun, Coley. *Sample Efficiency Matters: A Benchmark for Practical Molecular
  Optimization.* arXiv:2206.12411 (2022). (Why a REINVENT-family baseline is the right one.)
- Gao, Schulman, Hilton. *Scaling Laws for Reward Model Overoptimization.*
  arXiv:2210.10760 (2022). (Optimising a proxy reward past a point lowers the true one;
  the reason for the holdout predictor and the diversity constraint.)
- Jamieson, Talwalkar. *Non-stochastic Best Arm Identification and Hyperparameter
  Optimization.* arXiv:1502.07943 (2015) (successive halving); Li, Jamieson, DeSalvo,
  Rostamizadeh, Talwalkar. *Hyperband.* arXiv:1603.06560 (2016).
- Polykovskiy et al. *Molecular Sets (MOSES).* arXiv:1811.12823 (2018). (Validity,
  uniqueness, novelty, internal diversity conventions.)

## 4. The meta-search (`uv/metasearch.py`)

It builds on Section 4 of `OBJECTIVES_AND_MODELS.md` (open PR #1,
`research/objectives-and-models`): the same protocol shape as the DenoiseOpt bake-off
(identical protocol for every candidate, stage-gated, losers reported, pilot-to-full rank
correlation reported), applied to the steering method instead of the generator family.

**Search object.** method x hyper-parameters, with the family axis kept as a registry
(`FAMILIES`). Only the GRU SELFIES LM is pre-trained here, so it is the only family. The
graph and diffusion families in the design document (DiGress, GDSS, GraphAF, ...) are not
cheap: they need training from scratch (2 to 8 A100 hours per pilot in that document's own
estimate) and would not share the checkpoints, the prefix mechanism or the replay set. They
are deliberately out of this PR; a new family would be one more registry entry plus a
`Trial` subclass.

**Anchors, always evaluated, never eliminated:** `plain` and `prefix` (the existing steered
arm). Every trial is compared with them under identical sample sizes and seeds.

**Stages (successive halving, eta = 3).**

| stage | what | default budget |
|---|---|---|
| rung 0 | every candidate, short budget | 100 updates (trained methods), 1,000 samples |
| rung 1 | top 1/3, continued from checkpoint | 300 updates cumulative, 2,000 samples |
| rung 2 | top 1/3 again | 900 updates cumulative, 3,000 samples |
| confirm | top 3, fresh sampling seed, larger sample, anchors repeated | 10,000 samples |

Candidates: 16 guidance + 24 REINVENT + 12 self-training configs, drawn by a seeded random
sample of the grids in `sample_space` (distinct configs; no screen or compliance knob exists
in the space). Promoted trials continue from their checkpoint, nothing is retrained.
Guidance trials have no training; their rung budget is the number of evaluation samples.

**Objective and constraints.** Rank by in-band fraction (unique valid molecules whose
surrogate prediction is in 400 to 450 nm, the paper's metric). A trial is feasible only if
validity >= 0.95, uniqueness >= 0.90, novelty >= 0.90, rule pass rate >= 0.95 and internal
diversity (1 - mean pairwise Tanimoto, MOSES IntDiv1) >= 0.9 x the plain anchor's at the
same rung. Feasible trials always rank above infeasible ones. The diversity and uniqueness
floors are the guard against mode collapse and surrogate reward hacking: a policy that
emits one in-band molecule scores 1.0 on in-band and fails both. Thresholds are CLI flags
and were chosen before any GPU run, not tuned on results. Per trial we also log the sample
KL estimate to the prior (trained methods) and yield per sample.

**Held-out predictor (`uv/holdout_predictor.py`).** A second UV predictor that differs in
features (Morgan counts r3 + MACCS + conjugation descriptors), learner (ExtraTrees), seed
and split. It scores every trial at every rung, but selection through the rungs uses the
surrogate only; the confirm stage reports surrogate and holdout in-band side by side, and a
winner must also have a gap (surrogate minus holdout) of at most 0.10. Measured on the box:
test MAE 18.5 nm, R2 0.928 on its own 10 % split (643 structures; the surrogate's own split
gave MAE 24.0 nm, R2 0.873 on a different split, so those two are not like for like).
Honest limit: it is fitted on the same chromophore database, so it shares the database's
label noise and domain. It catches surrogate-specific exploits, not a wrong chemistry.
No TD-DFT or experimental check exists in this pipeline.

**Resumability and progress.** Every trained trial writes an atomic checkpoint (`state.pt`:
weights, optimiser, step count, pool, history) after each rung; every (stage, trial) result
is written to `rung_results.json` as it completes. Re-running the same command skips
finished work and continues promoted trials from their checkpoints. A changed schedule in
the same `--root` is refused. Heartbeat lines print steps/s, in-band batch fraction, KL and
elapsed time; every trial result prints one line with the constraint metrics.

**Outputs.** `results.json` and `results.md` in the run directory: confirm table (with
holdout columns, constraint status, winner), per-rung tables for all trials including
the losers, every trial's hyper-parameters, and the Spearman rank correlation between rung-0
and the last rung (does a short pilot predict the longer one).

**Safety scope.** The reward is `band_reward(pred_nm, lo, hi, soft_width)`: a function of
the predicted absorption maximum only (unit-tested, no molecule argument). The structural
screen from `smarts_rules.py` is a blocklist: an evaluation constraint, and in `selftrain`
a hard filter on training data, exactly the role it has in the existing pipeline. It is
never a reward and never a conditioning target. No synthesis, reaction or route code, and
no controlled-substance structures, are in this branch (tests use ordinary aromatic and
aliphatic building blocks).

## 5. Compute

The "about 36 steps/s" of the current generator on engine's RTX 3060 is supervised training
(teacher forcing, batch 512, one forward and backward pass). It does not transfer to this
work. A REINVENT update has to sample a batch (about 120 strictly sequential GRU steps),
decode SELFIES to SMILES, compute fingerprints, run the boosted surrogate, and then do a
forward and backward pass; most of that is CPU work or tiny sequential GPU kernels. No GPU
number for it has been measured (the GPU is busy with the seed runs).

Measured on the build box (CPU only, 8 shared cores, other jobs running, so noisy; two
measurements per quantity because the first run was less loaded):

| quantity | measurement 1 | measurement 2 |
|---|---|---|
| REINVENT update, batch 64 | 0.28 s | 0.79 s |
| REINVENT update, batch 128 | 0.50 s | 0.78 s |
| self-training step, amortised with a 1,000-sample collection round, batch 128 | n/a | 0.27 s |
| evaluation per 1,000 samples, without holdout | 1.2 s | n/a |
| evaluation per 1,000 samples, with holdout | n/a | 3.3 s |

Projection for the default pilot (52 candidates, rungs 100/300/900 updates, 10,000-sample
confirm), `python -m uv.metasearch_cost`, from the second (slower) measurement set:
**about 1.4 h central, 2.5 h upper bound of wall-clock**, where the upper bound makes every
trained survivor the slower method. The GPU is held for the whole run, so this is also the
GPU time to book. Engine's GPU may be faster or slower than the box CPU for the sampling
part; do not assume a speed-up. The engine script starts with the same timing probe on the
real device and writes `cost_probe.json` before the search begins, so the first minutes of
the run replace this estimate with a measured one. The pilot is search on ONE prior (seed 1);
the three-seed confirmation of a winner is a separate, later step and needs its own budget.

Engine script: `uv/engine/run_metasearch_pilot.sh` (refuses to start while the seed jobs are
alive, resumable, writes only under its own directory).
