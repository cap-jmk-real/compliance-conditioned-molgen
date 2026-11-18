"""Second-dataset replication of the plain vs steered in-band effect (#25).

Protocol (identical to the main result unless stated):
  surrogate   Morgan ECFP2048 + HistGradientBoosting, 600 trees, same split seeds, but
              trained on the second dataset (uv.dataset2.prepare_beard2019).
  corpus      the same 249,455 ZINC250k molecules in the same order, relabelled with the
              new surrogate (uv.dataset2.relabel_corpus).
  generators  same GRU SELFIES LM and the same hyperparameters as colab_run_seeds.py
              (618 epochs, batch 512, lr 1e-3, band 400 to 450 nm), seeds 0, 1, 2.
  evaluation  20,000 samples per arm, paired t-test across the 3 seeds with
              uv/aggregate_seeds.py, gates as in uv.dataset2.evaluate_ds2.

One deliberate reuse: the plain arm never reads the surrogate labels, so its training data
are identical to the main run. The already trained plain checkpoints of seeds 0, 1, 2 are
therefore reused, and only the steered arm is retrained on the new labels. Retraining the
plain arm would reproduce the same model up to GPU non-determinism and cost 2 x 2.3 h
per seed on the RTX 3060. The plain checkpoints are passed with --plain-ckpt.

Subcommands:
  prepare   build the dataset-2 CSV, train the surrogate, relabel the corpus (CPU)
  train     train the steered arm for one seed (GPU)
  evaluate  evaluate one seed (plain reused, steered new)
  report    aggregate the seeds (paired t-test) and write RESULTS_DATASET2.md

Everything is ordinary dye chemistry. The structural screen is a metric only; it is never
a conditioning target and never a reward. No synthesis, no routes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import colab_run_seeds as base  # noqa: E402  (hyperparameters live in one place)


def stream(cmd):
    print("==> " + " ".join(str(c) for c in cmd), flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, env=env, cwd=REPO)
    for line in iter(proc.stdout.readline, ""):
        print(line, end="", flush=True)
    proc.stdout.close()
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"command failed with rc={rc}: {' '.join(map(str, cmd))}")


def py(*a):
    return [sys.executable, "-u", "-m", *a]


def paths(root):
    sh = os.path.join(root, "shared")
    return {
        "shared": sh,
        "data": os.path.join(sh, "uvvis_dataset2.csv"),
        "data_stats": os.path.join(sh, "dataset2_stats.json"),
        "predictor": os.path.join(sh, "uvvis_predictor2.joblib"),
        "pred_metrics": os.path.join(sh, "predictor2_metrics.json"),
        "corpus": os.path.join(sh, "corpus_ds2.csv"),
        "steered": lambda s: os.path.join(root, f"generator_steered_ds2_seed{s}.pt"),
        "run_dir": lambda s: os.path.join(root, f"uv-generator-steered-ds2-seed{s}"),
        "eval_json": lambda s: os.path.join(root, f"eval_seed{s}.json"),
        "eval_md": lambda s: os.path.join(root, f"eval_seed{s}.md"),
    }


def cmd_prepare(a):
    p = paths(a.root)
    os.makedirs(p["shared"], exist_ok=True)
    if not os.path.exists(p["data"]):
        c = py("uv.dataset2.prepare_beard2019", "--out", p["data"],
               "--stats-json", p["data_stats"], "--band-lo", str(base.BAND_LO),
               "--band-hi", str(base.BAND_HI))
        if a.local_csv:
            c += ["--local-csv", a.local_csv]
        if a.local_zip:
            c += ["--local-zip", a.local_zip]
        if a.main_dataset:
            c += ["--compare", a.main_dataset]
        stream(c)
    if not os.path.exists(p["predictor"]):
        stream(py("uv.train_predictor", "--data", p["data"], "--out-model", p["predictor"],
                  "--metrics", p["pred_metrics"],
                  "--run-dir", os.path.join(a.root, "uv-predictor2"),
                  "--total-trees", str(base.PRED_TREES)))
    if not os.path.exists(p["corpus"]):
        stream(py("uv.dataset2.relabel_corpus", "--corpus", a.main_corpus,
                  "--predictor", p["predictor"], "--out", p["corpus"]))
    # the vocabulary is shared with the main run (identical tokens, same corpus)
    vocab = os.path.join(p["shared"], "vocab.json")
    if not os.path.exists(vocab):
        stream(["cp", "-n", a.main_vocab, vocab])


def cmd_train(a):
    p = paths(a.root)
    out, run_dir = p["steered"](a.seed), p["run_dir"](a.seed)
    stream(py("uv.train_generator", "--corpus", p["corpus"], "--arm", "steered",
              "--vocab", os.path.join(p["shared"], "vocab.json"), "--out", out,
              "--run-dir", run_dir, "--epochs", str(base.GEN_EPOCHS),
              "--batch-size", str(base.BATCH_SIZE), "--lr", str(base.LR),
              "--emb", str(base.EMB), "--hidden", str(base.HIDDEN),
              "--layers", str(base.LAYERS), "--band-lo", str(base.BAND_LO),
              "--band-hi", str(base.BAND_HI), "--ckpt-every", str(base.CKPT_EVERY),
              "--seed", str(a.seed)))


def cmd_evaluate(a):
    p = paths(a.root)
    plain = dict(kv.split("=", 1) for kv in a.plain_ckpt)[str(a.seed)]
    c = py("uv.dataset2.evaluate_ds2", "--corpus", p["corpus"],
           "--vocab", os.path.join(p["shared"], "vocab.json"), "--plain", plain,
           "--steered", p["steered"](a.seed), "--predictor", p["predictor"],
           "--predictor-metrics", p["pred_metrics"], "--n-samples", str(base.N_SAMPLES),
           "--n-bootstrap", str(base.N_BOOTSTRAP), "--band-lo", str(base.BAND_LO),
           "--band-hi", str(base.BAND_HI), "--seed", str(a.seed),
           "--out-json", p["eval_json"](a.seed), "--out-md", p["eval_md"](a.seed))
    if a.main_predictor:
        c += ["--cross-predictor", a.main_predictor]
    stream(c)


def cmd_report(a):
    p = paths(a.root)
    jsons = [p["eval_json"](s) for s in a.seeds]
    out_json = os.path.join(a.root, "aggregate_ds2.json")
    agg = subprocess.run(py("uv.aggregate_seeds", *jsons, "--out-json", out_json),
                         cwd=REPO, check=True, capture_output=True, text=True).stdout
    print(agg, flush=True)
    # gates and cross-surrogate per seed, as a compact table
    rows = []
    for s, j in zip(a.seeds, jsons):
        with open(j) as fh:
            m = json.load(fh)
        st, pl, g = m["arms"]["steered"], m["arms"]["plain"], m["gates"]["steered"]
        rows.append((s, pl, st, g, m.get("bootstrap_in_band_diff"),
                     m.get("bootstrap_in_band_diff_cross_surrogate")))
    lines = ["", "## Gates per seed (steered arm; limits: validity >= 0.95, uniqueness >= 0.90, "
             "novelty >= 0.90, rule pass >= 0.95, int_div >= 0.9 x plain)", "",
             "| seed | validity | uniqueness | novelty | rule pass | int_div (steered / plain) "
             "| all gates |", "|---|---|---|---|---|---|---|"]
    for s, pl, st, g, _, _ in rows:
        lines.append(f"| {s} | {st['validity']} | {st['uniqueness']} | {st['novelty']} | "
                     f"{st['rule_pass_rate']} | {st['int_div']} / {pl['int_div']} | "
                     f"{'pass' if g['passes_all'] else 'FAIL'} |")
    lines += ["", "## Secondary check: in-band fraction under the main dataset's surrogate", "",
              "| seed | plain | steered | diff | bootstrap 95% CI |", "|---|---|---|---|---|"]
    for s, pl, st, _, _, cb in rows:
        a_, b_ = pl.get("cross_surrogate_in_band_fraction"), st.get("cross_surrogate_in_band_fraction")
        if a_ is None or cb is None:
            continue
        lines.append(f"| {s} | {a_} | {b_} | {b_ - a_:+.4f} | [{cb['ci_low']}, {cb['ci_high']}] |")
    with open(os.path.join(a.root, "results_ds2.md"), "w") as fh:
        fh.write("# Second dataset (Beard 2019), plain vs steered, 3 seeds\n\n" + agg
                 + "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("T25_DS2_ROOT", "ckpt-ds2"),
                    help="output root for checkpoints and results")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("prepare")
    sp.add_argument("--main-corpus", required=True)
    sp.add_argument("--main-vocab", required=True)
    sp.add_argument("--main-dataset", default=None, help="main dataset CSV for the overlap report")
    sp.add_argument("--local-csv", default=None)
    sp.add_argument("--local-zip", default=None)
    st = sub.add_parser("train")
    st.add_argument("--seed", type=int, required=True)
    se = sub.add_parser("evaluate")
    se.add_argument("--seed", type=int, required=True)
    se.add_argument("--plain-ckpt", nargs="+", required=True, metavar="SEED=PATH")
    se.add_argument("--main-predictor", default=None)
    sr = sub.add_parser("report")
    sr.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    {"prepare": cmd_prepare, "train": cmd_train, "evaluate": cmd_evaluate,
     "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
