"""Single entrypoint for the UV-Vis steered-generation experiment.

Modes:
  --calibration   run a short fixed workload and report wall-clock and throughput,
                  so the real A100 rate can be measured before the full run. Writes a
                  JSON report and projects the full-run wall-clock.
  --full          the full pipeline at full scale: the complete UV-Vis dataset and the
                  full ZINC250k corpus, both generator arms plus evaluation, sized to
                  finish in roughly 3 to 6 hours on an A100.
  --smoke         tiny end-to-end run on CPU (few hundred molecules, 1 epoch).
  (default)       the earlier mid-scale pipeline (kept for backward compatibility).

Every heavy step is a separate module so it can be rerun and resumes from its own
checkpoints. This script sequences them, prints progress with timing, and keeps child
output unbuffered so a watching human sees the run is alive.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable


def run(step_name, module_args):
    print(f"\n=== {step_name} ===", flush=True)
    t0 = time.time()
    # -u on the child and PYTHONUNBUFFERED in the environment keep grandchild output
    # unbuffered, so nothing in the call chain swallows it.
    cmd = [PY, "-u", "-m"] + module_args
    print("  " + " ".join(cmd), flush=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    rc = subprocess.call(cmd, cwd=ROOT, env=env)
    dt = time.time() - t0
    print(f"=== {step_name} finished in {dt:.1f}s rc={rc} ===", flush=True)
    if rc != 0:
        raise SystemExit(f"{step_name} failed with rc={rc}")
    return dt


def calibration(args):
    """Fixed small workload that measures the real device rate and projects the full
    run. All instrumentation lives in uv.calibrate."""
    out_json = os.path.join(args.ckpt_root, "calibration.json")
    run("calibration", [
        "uv.calibrate",
        "--ckpt-root", args.ckpt_root,
        "--out-json", out_json,
        "--uvvis-local", args.uvvis_local or "",
        "--zinc-local", args.zinc_local or "",
        "--band-lo", str(args.band_lo), "--band-hi", str(args.band_hi),
        "--full-corpus-n", str(args.corpus_n_full),
        "--full-batch-size", str(args.batch_size),
        "--full-pred-trees", str(args.pred_trees),
        "--full-n-samples", str(args.n_samples),
        "--target-hours", str(args.target_hours),
    ])
    print(f"calibration report written to {out_json}", flush=True)


def _sized_epochs(args, corpus_rows):
    """Choose generator epochs. An explicit --gen-epochs wins. Otherwise size from the
    calibration report so the full run lands near --target-hours, using the same
    formula as the calibration projection."""
    if args.gen_epochs and args.gen_epochs > 0:
        print(f"using explicit --gen-epochs {args.gen_epochs}", flush=True)
        return args.gen_epochs
    calib_path = os.path.join(args.ckpt_root, "calibration.json")
    if not os.path.exists(calib_path):
        print(f"no calibration report at {calib_path}; falling back to "
              f"{args.default_gen_epochs} epochs. Run --calibration first for a sized "
              f"budget.", flush=True)
        return args.default_gen_epochs
    with open(calib_path) as fh:
        rep = json.load(fh)
    rate = rep["throughput"]["train_steps_per_sec"]
    gen_rate = rep["throughput"]["gen_samples_per_sec"]
    stage = rep["stage_seconds"]
    calib = rep["workload"]
    steps_per_epoch = max(math.ceil(corpus_rows / args.batch_size), 1)
    # Project the non-training stages at full scale from the calibration scale.
    t_data = stage["data"] * (args.uvvis_rows_full / max(calib["uvvis_rows"], 1))
    t_pred = (stage["predictor"] * (args.pred_trees / max(calib["pred_trees"], 1))
              * (args.uvvis_rows_full / max(calib["uvvis_rows"], 1)))
    t_corpus = stage["corpus"] * (corpus_rows / max(calib["corpus_kept"], 1))
    t_eval = (args.n_samples * 2) / max(gen_rate, 1e-9)
    non_training = t_data + t_pred + t_corpus + t_eval
    budget = max(args.target_hours * 3600.0 - non_training, 0.0)
    sec_per_epoch = steps_per_epoch / max(rate, 1e-9)
    epochs = max(int((budget / 2.0) / max(sec_per_epoch, 1e-9)), 1)
    proj_hours = (non_training + 2 * epochs * sec_per_epoch) / 3600.0
    print(f"sized from calibration: {epochs} epochs per arm "
          f"(rate {rate:.2f} steps/s, {steps_per_epoch} steps/epoch, "
          f"projected total {proj_hours:.2f} h, target {args.target_hours} h)", flush=True)
    return epochs


def full(args):
    uv_local = ["--local", args.uvvis_local] if args.uvvis_local else []
    zinc_local = ["--local", args.zinc_local] if args.zinc_local else []

    run("data prep (full)", ["uv.prepare_uvvis", "--out", "uv/data/uvvis_dataset.csv"] + uv_local)
    run("predictor (full)", [
        "uv.train_predictor", "--data", "uv/data/uvvis_dataset.csv",
        "--out-model", "uv/artifacts/uvvis_predictor.joblib",
        "--metrics", "uv/artifacts/predictor_metrics.json",
        "--run-dir", os.path.join(args.ckpt_root, "uv-predictor"),
        "--total-trees", str(args.pred_trees)])
    run("corpus (full)", [
        "uv.prepare_corpus", "--out", "uv/data/corpus.csv",
        "--predictor", "uv/artifacts/uvvis_predictor.joblib",
        "--n", str(args.corpus_n_full)] + zinc_local)

    import pandas as pd
    corpus_rows = len(pd.read_csv("uv/data/corpus.csv"))
    print(f"full corpus has {corpus_rows} molecules", flush=True)
    epochs = _sized_epochs(args, corpus_rows)

    for arm in ("plain", "steered"):
        run(f"generator {arm} (full)", [
            "uv.train_generator", "--corpus", "uv/data/corpus.csv", "--arm", arm,
            "--vocab", "uv/artifacts/vocab.json",
            "--out", f"uv/artifacts/generator_{arm}.pt",
            "--run-dir", os.path.join(args.ckpt_root, f"uv-generator-{arm}"),
            "--epochs", str(epochs), "--batch-size", str(args.batch_size),
            "--ckpt-every", str(args.ckpt_every),
            "--band-lo", str(args.band_lo), "--band-hi", str(args.band_hi)])

    run("evaluate (full)", [
        "uv.evaluate", "--corpus", "uv/data/corpus.csv",
        "--vocab", "uv/artifacts/vocab.json",
        "--predictor", "uv/artifacts/uvvis_predictor.joblib",
        "--predictor-metrics", "uv/artifacts/predictor_metrics.json",
        "--n-samples", str(args.n_samples),
        "--n-bootstrap", str(args.n_bootstrap),
        "--band-lo", str(args.band_lo), "--band-hi", str(args.band_hi),
        "--out-json", "uv/artifacts/eval_results.json",
        "--out-md", "uv/artifacts/eval_results.md"])
    print("full run complete. Results in uv/artifacts/eval_results.json and .md", flush=True)


def default_pipeline(args):
    uv_local = ["--local", args.uvvis_local] if args.uvvis_local else []
    zinc_local = ["--local", args.zinc_local] if args.zinc_local else []
    run("data prep", ["uv.prepare_uvvis", "--out", "uv/data/uvvis_dataset.csv"] + uv_local)
    run("predictor", [
        "uv.train_predictor", "--data", "uv/data/uvvis_dataset.csv",
        "--run-dir", os.path.join(args.ckpt_root, "uv-predictor"),
        "--total-trees", str(args.pred_trees)])
    run("corpus", [
        "uv.prepare_corpus", "--out", "uv/data/corpus.csv",
        "--n", str(args.corpus_n)] + zinc_local)
    for arm in ("plain", "steered"):
        run(f"generator {arm}", [
            "uv.train_generator", "--corpus", "uv/data/corpus.csv", "--arm", arm,
            "--run-dir", os.path.join(args.ckpt_root, f"uv-generator-{arm}"),
            "--epochs", str(args.default_gen_epochs), "--batch-size", str(args.batch_size),
            "--band-lo", str(args.band_lo), "--band-hi", str(args.band_hi)])
    run("evaluate", [
        "uv.evaluate", "--corpus", "uv/data/corpus.csv",
        "--n-samples", str(args.n_samples),
        "--band-lo", str(args.band_lo), "--band-hi", str(args.band_hi)])


def smoke(args):
    uv_local = ["--local", args.uvvis_local] if args.uvvis_local else []
    zinc_local = ["--local", args.zinc_local] if args.zinc_local else []
    run("data prep (smoke)", [
        "uv.prepare_uvvis", "--out", "uv/data/uvvis_smoke.csv", "--max-rows", "800"] + uv_local)
    run("predictor (smoke)", [
        "uv.train_predictor", "--data", "uv/data/uvvis_smoke.csv",
        "--out-model", "uv/artifacts/predictor_smoke.joblib",
        "--metrics", "uv/artifacts/predictor_smoke_metrics.json",
        "--run-dir", os.path.join(args.ckpt_root, "smoke-predictor"),
        "--total-trees", "100", "--stage-trees", "100"])
    run("corpus (smoke)", [
        "uv.prepare_corpus", "--out", "uv/data/corpus_smoke.csv",
        "--predictor", "uv/artifacts/predictor_smoke.joblib", "--n", "400"] + zinc_local)
    for arm in ("plain", "steered"):
        run(f"generator {arm} (smoke)", [
            "uv.train_generator", "--corpus", "uv/data/corpus_smoke.csv", "--arm", arm,
            "--vocab", "uv/artifacts/vocab_smoke.json",
            "--run-dir", os.path.join(args.ckpt_root, f"smoke-generator-{arm}"),
            "--epochs", "1", "--batch-size", "64"])
    run("evaluate (smoke)", [
        "uv.evaluate", "--corpus", "uv/data/corpus_smoke.csv",
        "--vocab", "uv/artifacts/vocab_smoke.json",
        "--predictor", "uv/artifacts/predictor_smoke.joblib",
        "--n-samples", "200",
        "--out-json", "uv/artifacts/eval_smoke.json",
        "--out-md", "uv/artifacts/eval_smoke.md"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt-root", default="/content/drive/MyDrive/t25-checkpoints")
    ap.add_argument("--uvvis-local", default=None)
    ap.add_argument("--zinc-local", default=None)
    ap.add_argument("--pred-trees", type=int, default=600)
    ap.add_argument("--corpus-n", type=int, default=20000,
                    help="corpus size for the default mid-scale pipeline")
    ap.add_argument("--corpus-n-full", type=int, default=249455,
                    help="corpus size for --full (full ZINC250k)")
    ap.add_argument("--uvvis-rows-full", type=int, default=6425,
                    help="expected unique UV-Vis structures at full scale, for sizing")
    ap.add_argument("--gen-epochs", type=int, default=0,
                    help="generator epochs; 0 means auto-size --full from calibration")
    ap.add_argument("--default-gen-epochs", type=int, default=50,
                    help="fallback epochs when no calibration report is present")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--band-lo", type=float, default=400.0)
    ap.add_argument("--band-hi", type=float, default=450.0)
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--target-hours", type=float, default=4.5)
    args = ap.parse_args()

    if args.calibration:
        calibration(args)
    elif args.full:
        full(args)
    elif args.smoke:
        smoke(args)
    else:
        default_pipeline(args)


if __name__ == "__main__":
    main()
