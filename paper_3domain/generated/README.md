# Generated manuscript fragments

Headline numbers in `paper_3domain/main.tex` are sourced from committed result JSON and rendered via
this folder. **Do not hand-edit** `macros.tex` or `tab_*.tex`.

## Regenerate

```bash
python3 paper_3domain/scripts/generate_paper_artifacts.py
pytest tests/test_paper_generated_drift.py -q
```

## Optional evaluation-only updates

| Script | Purpose |
|--------|---------|
| `paper_3domain/scripts/eval_uvvis_gnn_ensemble.py` | Recompute UV-Vis ensemble MAE from checkpoints |
| `paper_3domain/scripts/scaffold_split_eval.py` | Murcko scaffold-group held-out MAE (stress test) |
| `plastics/compute_corpus_compliance.py` | Polymer corpus REACH+BPA failure rate |

Checkpoints (`uv/results/round3_gnn/checkpoints/`) are intentionally **not** committed (large, reproducible from `uv/gnn_predictor.py`).