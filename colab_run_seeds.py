"""Run the additional training seeds for the #25 UV-Vis steered-generation experiment.

Seed 0 is the single run already reported in the paper. This entrypoint adds
independent training seeds (default 1 and 2) so training-level variance can be
quantified, which is the reviewer's main request. For each requested seed it trains
both arms (plain and steered) with EXACTLY the seed-0 hyperparameters and step budget,
then evaluates that seed with the same 20,000-sample protocol as the paper.

What is held fixed vs what varies
---------------------------------
The surrogate property model and the molecular corpus are held FIXED at seed 0. Both
are produced deterministically (scikit-learn split seeds 42/7, numpy seed 0, the full
ZINC250k corpus in release order with no subsampling at --full scale) and shared across
all seeds. So they are identical to the seed-0 run. The ONLY things that vary with
--seed are the generator's weight initialisation and the data-shuffle order. The spread
across seeds is therefore training variance measured under one surrogate and one corpus,
which is the comparison the paper needs. This choice is documented in uv/NOTES.md.

Budget note (read before launching)
------------------------------------
Each arm trains 301,584 steps (618 epochs) per the seed-0 budget. On the Colab
A100-SXM4-40GB the seed-0 run took about 52.5 min of training per arm plus ~6.5 min of
evaluation per seed. Two new seeds are therefore about 2 x (2 x 52.5 + 6.5) = ~3.7 h of
GPU time, plus a one-time ~30 min to build the shared surrogate and corpus if they are
not already on Drive (reused on resume). Keep an eye on the Colab compute-unit meter.

Resumability and streaming
--------------------------
Checkpoints, the shared surrogate/corpus, and the per-seed eval JSONs all go to mounted
Google Drive under --ckpt-root (default /content/drive/MyDrive/t25-colab/ckpt-seeds).
train_generator resumes from the newest checkpoint, so a disconnect loses at most one
--ckpt-every interval. Child output is streamed line by line so a watching human sees
step, epoch, loss, elapsed and ETA as the run goes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

REPO_DIR = os.environ.get("T25_REPO_DIR", "/content/compliance-conditioned-molgen")

# --- EXACT seed-0 hyperparameters and step budget (see paper Methods, "Training and
# --- evaluation", and uv/run.py --full defaults). Do not change these: seeds 1 and 2
# --- must match seed 0 so the spread is training variance, not a budget difference.
GEN_EPOCHS = 618          # 618 epochs == 301,584 steps/arm at batch 512, full corpus
BATCH_SIZE = 512
LR = 1e-3
EMB = 128
HIDDEN = 256
LAYERS = 2
BAND_LO = 400.0
BAND_HI = 450.0
PRED_TREES = 600
CORPUS_N_FULL = 249455    # full ZINC250k; no subsampling, so corpus is seed-independent
N_SAMPLES = 20000
N_BOOTSTRAP = 2000
CKPT_EVERY = 1000

# Seeds to add. Seed 0 is the paper run and is NOT retrained here. Override with
# T25_SEEDS="1" to run a single seed per Colab session if you want to stay well under
# the compute-unit budget.
SEEDS = [int(x) for x in os.environ.get("T25_SEEDS", "1,2").split(",") if x.strip()]

CKPT_ROOT = os.environ.get("T25_CKPT_ROOT",
                           "/content/drive/MyDrive/t25-colab/ckpt-seeds")
SHARED = os.path.join(CKPT_ROOT, "shared")

# Optional local dataset paths (skip the download). Empty => download verified URLs.
UVVIS_LOCAL = os.environ.get("T25_UVVIS_LOCAL", "")
ZINC_LOCAL = os.environ.get("T25_ZINC_LOCAL", "")


def stream(cmd, env=None):
    print("==> " + " ".join(str(c) for c in cmd), flush=True)
    env = env or dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env, cwd=REPO_DIR)
    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)
    proc.stdout.close()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"command failed with rc={rc}: {' '.join(str(c) for c in cmd)}")
    return rc


def py(*module_args):
    return [sys.executable, "-u", "-m", *module_args]


def ensure_shared():
    """Build the fixed surrogate + corpus + vocab once, shared by every seed.

    Deterministic: identical to the seed-0 run. Skipped stage by stage if its output is
    already on Drive, so a resumed session does not repeat the ~30 min corpus build.
    """
    os.makedirs(SHARED, exist_ok=True)
    predictor = os.path.join(SHARED, "uvvis_predictor.joblib")
    pred_metrics = os.path.join(SHARED, "predictor_metrics.json")
    corpus = os.path.join(SHARED, "corpus.csv")
    vocab = os.path.join(SHARED, "vocab.json")
    uvvis_csv = os.path.join(SHARED, "uvvis_dataset.csv")

    if not os.path.exists(predictor):
        print("\n==> [shared] building the fixed surrogate predictor (seed 0)", flush=True)
        data_cmd = py("uv.prepare_uvvis", "--out", uvvis_csv)
        if UVVIS_LOCAL:
            data_cmd += ["--local", UVVIS_LOCAL]
        stream(data_cmd)
        stream(py("uv.train_predictor", "--data", uvvis_csv,
                  "--out-model", predictor, "--metrics", pred_metrics,
                  "--run-dir", os.path.join(CKPT_ROOT, "uv-predictor"),
                  "--total-trees", str(PRED_TREES)))
    else:
        print(f"==> [shared] reusing surrogate {predictor}", flush=True)

    if not os.path.exists(corpus):
        print("\n==> [shared] building the fixed corpus (full ZINC250k, seed 0)", flush=True)
        corpus_cmd = py("uv.prepare_corpus", "--out", corpus,
                        "--predictor", predictor, "--n", str(CORPUS_N_FULL))
        if ZINC_LOCAL:
            corpus_cmd += ["--local", ZINC_LOCAL]
        stream(corpus_cmd)
    else:
        print(f"==> [shared] reusing corpus {corpus}", flush=True)

    return predictor, pred_metrics, corpus, vocab


def train_seed(seed, predictor, pred_metrics, corpus, vocab):
    gen_paths = {}
    for arm in ("plain", "steered"):
        out = os.path.join(CKPT_ROOT, f"generator_{arm}_seed{seed}.pt")
        run_dir = os.path.join(CKPT_ROOT, f"uv-generator-{arm}-seed{seed}")
        gen_paths[arm] = out
        print(f"\n==> [seed {seed}] training {arm} arm "
              f"({GEN_EPOCHS} epochs, batch {BATCH_SIZE}, lr {LR})", flush=True)
        stream(py("uv.train_generator", "--corpus", corpus, "--arm", arm,
                  "--vocab", vocab, "--out", out, "--run-dir", run_dir,
                  "--epochs", str(GEN_EPOCHS), "--batch-size", str(BATCH_SIZE),
                  "--lr", str(LR), "--emb", str(EMB), "--hidden", str(HIDDEN),
                  "--layers", str(LAYERS), "--band-lo", str(BAND_LO),
                  "--band-hi", str(BAND_HI), "--ckpt-every", str(CKPT_EVERY),
                  "--seed", str(seed)))
    return gen_paths


def eval_seed(seed, gen_paths, predictor, pred_metrics, corpus, vocab):
    out_json = os.path.join(CKPT_ROOT, f"eval_seed{seed}.json")
    out_md = os.path.join(CKPT_ROOT, f"eval_seed{seed}.md")
    print(f"\n==> [seed {seed}] evaluating both arms on {N_SAMPLES} samples", flush=True)
    stream(py("uv.evaluate", "--corpus", corpus, "--vocab", vocab,
              "--plain", gen_paths["plain"], "--steered", gen_paths["steered"],
              "--predictor", predictor, "--predictor-metrics", pred_metrics,
              "--n-samples", str(N_SAMPLES), "--n-bootstrap", str(N_BOOTSTRAP),
              "--band-lo", str(BAND_LO), "--band-hi", str(BAND_HI),
              "--seed", str(seed), "--out-json", out_json, "--out-md", out_md))
    # Also split the combined result into one JSON per arm (no new numbers, just a
    # split) so there is one JSON per seed AND per arm on Drive, as requested.
    with open(out_json) as fh:
        meta = json.load(fh)
    for arm, summary in meta.get("arms", {}).items():
        rec = {"seed": seed, "arm": arm, "band_lo": meta.get("band_lo"),
               "band_hi": meta.get("band_hi"), "n_samples": meta.get("n_samples"),
               "metrics": summary,
               "predictor_metrics": meta.get("predictor_metrics"),
               "bootstrap_in_band_diff": meta.get("bootstrap_in_band_diff")}
        arm_path = os.path.join(CKPT_ROOT, f"eval_seed{seed}_{arm}.json")
        with open(arm_path, "w") as fh:
            json.dump(rec, fh, indent=2)
        print(f"==> wrote {arm_path}", flush=True)
    return out_json


def main():
    print(f"==> repo: {REPO_DIR}", flush=True)
    print(f"==> ckpt root (Drive): {CKPT_ROOT}", flush=True)
    print(f"==> seeds to add: {SEEDS} (seed 0 is the paper run, not retrained)", flush=True)
    os.makedirs(CKPT_ROOT, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         text=True, cwd=REPO_DIR).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                         text=True, cwd=REPO_DIR).strip()
        print(f"==> code: {branch} @{commit}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"==> could not read git head: {exc}", flush=True)

    predictor, pred_metrics, corpus, vocab = ensure_shared()

    eval_jsons = []
    for seed in SEEDS:
        gen_paths = train_seed(seed, predictor, pred_metrics, corpus, vocab)
        eval_jsons.append(eval_seed(seed, gen_paths, predictor, pred_metrics, corpus, vocab))

    print("\n==> all requested seeds finished", flush=True)
    print(f"==> per-seed eval JSONs: {eval_jsons}", flush=True)
    print("==> aggregate with: python -m uv.aggregate_seeds "
          "uv/results/eval_results.json " + " ".join(eval_jsons), flush=True)


if __name__ == "__main__":
    main()
