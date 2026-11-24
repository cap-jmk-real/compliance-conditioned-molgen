# Published-method baselines (open item — not run in this PR)

JCIM reviewers expect comparisons to established de novo generators. The scripts below are **runnable**
wrappers around public implementations. They were **not executed** here because each oracle budget
requires sustained GPU sampling (~10k–50k structures per domain).

## Oracle budget (matches main-paper steering evals)

| Domain | Oracle calls per method | Property oracle |
|--------|-------------------------|-----------------|
| UV-Vis | 10,000 × 3 prior seeds = **30,000** | 4-GNN ensemble + Morgan-HGB (same weights as Track 1) |
| Polymers | **3,000** P-SMILES samples | 4-GNN $T_g$ ensemble |
| CNS | **3,000** SELFIES samples | Multi-target GNN ensemble + holdout ET (scoring only) |

## REINVENT 4 (MIT license)

```bash
# Install once: pip install reinvent-chemistry reinvent-scoring reinvent-models
export T25_ORACLE_BUDGET=30000
python3 baselines/run_reinvent4_uvvis.py \
  --budget 10000 --seeds 0,1,2 \
  --band 400,450 \
  --scorer-config baselines/configs/uvvis_oracle.json \
  --out baselines/outputs/reinvent4_uvvis.json
```

## GraphGA (guacamol / oraloa)

```bash
pip install guacamol
python3 baselines/run_graphga_uvvis.py \
  --budget 10000 --seed 42 \
  --band 400,450 \
  --out baselines/outputs/graphga_uvvis.json
```

## Filter-after-generation (generate-then-screen)

Uses the same ZINC250k prior checkpoint as Track 1, generates 10,000 SELFIES per seed, keeps
REACH-compliant structures whose Morgan-HGB $\lambda_{\max}$ falls in-band (no RL).

```bash
python3 baselines/run_filter_after_generation_uvvis.py \
  --n 10000 --seed 0 \
  --band 400,450 \
  --out baselines/outputs/filter_after_gen_uvvis_seed0.json
```

## Outputs

Write JSON summaries compatible with `paper_3domain/scripts/generate_paper_artifacts.py` (schema TBD after
baseline runs). Until then, the manuscript discusses these as future work in the PR description.