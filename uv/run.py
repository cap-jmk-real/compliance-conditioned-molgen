"""Single entrypoint for the UV-Vis steered-generation experiment.

Modes:
  --calibration   run a short fixed workload and report wall-clock and throughput,
                  so the real A100 rate can be measured before the full run.
  --smoke         tiny end-to-end run on CPU (few hundred molecules, 1 epoch).
  (default)       full pipeline: data prep, predictor, corpus, both generator arms,
                  evaluation.

Every heavy step is a separate module so it can be rerun and resumes from its own
checkpoints. This script just sequences them and prints progress with timing.
"""
from __future__ import annotations

import argparse
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
    cmd = [PY, "-m"] + module_args
    print("  " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    dt = time.time() - t0
    print(f"=== {step_name} finished in {dt:.1f}s rc={rc} ===", flush=True)
    if rc != 0:
        raise SystemExit(f"{step_name} failed with rc={rc}")
    return dt


def calibration(args):
    """Fixed small workload: prep a small dataset, fit a small predictor, train a
    fixed number of generator steps, and report throughput."""
    print("CALIBRATION mode: short fixed workload to measure the real rate", flush=True)
    t0 = time.time()
    run("data (calibration subset)", [
        "uv.prepare_uvvis", "--out", "uv/data/uvvis_calib.csv",
        "--local", args.uvvis_local or "", "--max-rows", "1500"])
    run("predictor (calibration)", [
        "uv.train_predictor", "--data", "uv/data/uvvis_calib.csv",
        "--out-model", "uv/artifacts/predictor_calib.joblib",
        "--metrics", "uv/artifacts/predictor_calib_metrics.json",
        "--run-dir", os.path.join(args.ckpt_root, "calib-predictor"),
        "--total-trees", "200", "--stage-trees", "100"])
    run("corpus (calibration)", [
        "uv.prepare_corpus", "--out", "uv/data/corpus_calib.csv",
        "--predictor", "uv/artifacts/predictor_calib.joblib",
        "--local", args.zinc_local or "", "--n", "2000"])
    gen_t = run("generator calibration (fixed 3 epochs)", [
        "uv.train_generator", "--corpus", "uv/data/corpus_calib.csv", "--arm", "plain",
        "--vocab", "uv/artifacts/vocab_calib.json",
        "--out", "uv/artifacts/generator_calib.pt",
        "--run-dir", os.path.join(args.ckpt_root, "calib-generator"),
        "--epochs", "3", "--batch-size", "256"])
    total = time.time() - t0
    print(f"\nCALIBRATION total wall-clock: {total:.1f}s", flush=True)
    print(f"generator step throughput measured during the generator stage above "
          f"(see steps/s in its heartbeat). Use colab_burn_rate.py to turn the Colab "
          f"unit readings around this run into units per hour.", flush=True)


def full(args):
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
            "--epochs", str(args.gen_epochs), "--batch-size", str(args.batch_size),
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
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt-root", default="/content/drive/MyDrive/t25-checkpoints")
    ap.add_argument("--uvvis-local", default=None)
    ap.add_argument("--zinc-local", default=None)
    ap.add_argument("--pred-trees", type=int, default=600)
    ap.add_argument("--corpus-n", type=int, default=20000)
    ap.add_argument("--gen-epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--band-lo", type=float, default=450.0)
    ap.add_argument("--band-hi", type=float, default=550.0)
    ap.add_argument("--n-samples", type=int, default=5000)
    args = ap.parse_args()

    if args.calibration:
        calibration(args)
    elif args.smoke:
        smoke(args)
    else:
        full(args)


if __name__ == "__main__":
    main()
