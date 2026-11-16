"""Measured calibration run for the UV-Vis steered-generation pipeline.

This runs a small fixed workload on whatever device is available and reports real
numbers so the full A100 run can be sized before any compute units are spent. It
prints continuously and writes a JSON report.

What it measures:
  - the detected device and GPU name, plus torch and CUDA versions
  - the fixed workload it ran (dataset rows, epochs, batch size, model parameters)
  - wall-clock for each stage (data, predictor, corpus, generator training, sampling)
  - generator training steps per second and generation samples per second
  - a projection of the full-run wall-clock from the measured rates

Nothing here is invented. Every number comes from the workload that actually ran.
The projection is labelled as a projection and uses the measured rates. On a box
without a GPU the rates are CPU rates, so the projected full-run time is a CPU
projection, not an A100 one.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import selfies as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uv import prepare_uvvis, train_predictor, prepare_corpus, gen_model  # noqa: E402
from uv.common import canonical_smiles, morgan_fingerprint, passes_screen  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402


def log(msg):
    print(msg, flush=True)


def size_full_training(rate_steps_per_sec, gen_samples_per_sec, stage_seconds,
                       calib_scale, full_scale, keep_ratio, target_hours):
    """Project full-run wall-clock and recommend generator epochs for a target budget.

    All inputs are measured. The returned epoch count divides the training budget
    evenly between the two arms. Returns a dict that both the calibration report and
    the --full sizing use, so there is one formula.
    """
    full_corpus_kept = max(int(round(keep_ratio * full_scale["corpus_n"])), 1)
    steps_per_epoch = max(math.ceil(full_corpus_kept / full_scale["batch_size"]), 1)

    # Non-training stages, projected linearly from the calibration scale.
    t_data = stage_seconds["data"] * (full_scale["uvvis_rows"] / max(calib_scale["uvvis_rows"], 1))
    t_pred = (stage_seconds["predictor"]
              * (full_scale["pred_trees"] / max(calib_scale["pred_trees"], 1))
              * (full_scale["uvvis_rows"] / max(calib_scale["uvvis_rows"], 1)))
    t_corpus = stage_seconds["corpus"] * (full_scale["corpus_n"] / max(calib_scale["corpus_n"], 1))
    t_eval = (full_scale["n_samples"] * 2) / max(gen_samples_per_sec, 1e-9)
    non_training = t_data + t_pred + t_corpus + t_eval

    target_seconds = target_hours * 3600.0
    training_budget = max(target_seconds - non_training, 0.0)
    per_arm_budget = training_budget / 2.0
    # Seconds per epoch for one arm at the measured rate.
    sec_per_epoch = steps_per_epoch / max(rate_steps_per_sec, 1e-9)
    epochs = int(math.floor(per_arm_budget / max(sec_per_epoch, 1e-9)))
    epochs = max(epochs, 1)

    train_seconds = 2 * epochs * sec_per_epoch
    projected_total = non_training + train_seconds
    return {
        "full_corpus_kept_est": full_corpus_kept,
        "steps_per_epoch_full": steps_per_epoch,
        "sec_per_epoch_per_arm": round(sec_per_epoch, 3),
        "projected_non_training_seconds": round(non_training, 1),
        "projected_data_seconds": round(t_data, 1),
        "projected_predictor_seconds": round(t_pred, 1),
        "projected_corpus_seconds": round(t_corpus, 1),
        "projected_eval_seconds": round(t_eval, 1),
        "recommended_gen_epochs": epochs,
        "projected_training_seconds": round(train_seconds, 1),
        "projected_total_seconds": round(projected_total, 1),
        "projected_total_hours": round(projected_total / 3600.0, 3),
        "target_hours": target_hours,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="/content/drive/MyDrive/t25-checkpoints")
    ap.add_argument("--uvvis-local", default=None)
    ap.add_argument("--zinc-local", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--band-lo", type=float, default=400.0)
    ap.add_argument("--band-hi", type=float, default=450.0)
    # Fixed calibration workload.
    ap.add_argument("--calib-uvvis-rows", type=int, default=1500)
    ap.add_argument("--calib-pred-trees", type=int, default=200)
    ap.add_argument("--calib-corpus-n", type=int, default=2000)
    ap.add_argument("--calib-epochs", type=int, default=3)
    ap.add_argument("--calib-batch-size", type=int, default=256)
    ap.add_argument("--calib-gen-samples", type=int, default=1000)
    # Full-run configuration the projection sizes against (TASK 2 defaults).
    ap.add_argument("--full-uvvis-rows", type=int, default=6425)
    ap.add_argument("--full-pred-trees", type=int, default=600)
    ap.add_argument("--full-corpus-n", type=int, default=249455)
    ap.add_argument("--full-batch-size", type=int, default=512)
    ap.add_argument("--full-n-samples", type=int, default=20000)
    ap.add_argument("--target-hours", type=float, default=4.5)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else None
    log("CALIBRATION mode: fixed workload, measuring the real rate")
    log(f"device={device} gpu={gpu_name}")
    log(f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"cuda_available={torch.cuda.is_available()}")
    log(f"python={platform.python_version()} platform={platform.platform()}")

    stage_seconds = {}

    # ---- Stage 1: data ----
    log("\n=== stage: data (calibration subset) ===")
    t0 = time.time()
    raw = prepare_uvvis.load_raw(args.uvvis_local)
    clean = prepare_uvvis.build(raw)
    if len(clean) > args.calib_uvvis_rows:
        clean = clean.sample(n=args.calib_uvvis_rows, random_state=0).reset_index(drop=True)
    stage_seconds["data"] = time.time() - t0
    log(f"data stage: {len(clean)} structures in {stage_seconds['data']:.1f}s")

    # ---- Stage 2: predictor ----
    log("\n=== stage: predictor (calibration) ===")
    t0 = time.time()
    X, keep = train_predictor.featurize(clean["smiles"].tolist(), 2048)
    y = clean["abs_max_nm"].to_numpy()[keep]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1, random_state=42)
    predictor = HistGradientBoostingRegressor(
        max_iter=args.calib_pred_trees, learning_rate=0.1,
        l2_regularization=1.0, random_state=0)
    predictor.fit(X_tr, y_tr)
    val_mae = float(mean_absolute_error(y_te, predictor.predict(X_te)))
    stage_seconds["predictor"] = time.time() - t0
    log(f"predictor stage: test MAE {val_mae:.2f} nm in {stage_seconds['predictor']:.1f}s")

    # ---- Stage 3: corpus ----
    log("\n=== stage: corpus (calibration) ===")
    t0 = time.time()
    smiles = prepare_corpus.load_smiles(args.zinc_local, args.calib_corpus_n)
    rows = []
    processed = 0
    for smi in smiles:
        processed += 1
        canon = canonical_smiles(smi)
        if canon is None:
            continue
        try:
            enc = sf.encoder(canon)
        except Exception:
            continue
        fp = morgan_fingerprint(canon, n_bits=2048)
        if fp is None:
            continue
        abs_pred = float(predictor.predict(fp.reshape(1, -1))[0])
        rows.append((canon, enc, abs_pred, bool(passes_screen(canon))))
    corpus = pd.DataFrame(rows, columns=["smiles", "selfies", "abs_pred", "passes_screen"])
    stage_seconds["corpus"] = time.time() - t0
    keep_ratio = len(corpus) / max(processed, 1)
    log(f"corpus stage: kept {len(corpus)}/{processed} "
        f"(keep ratio {keep_ratio:.3f}) in {stage_seconds['corpus']:.1f}s")

    # ---- Stage 4: generator training (measured steps/s) ----
    log("\n=== stage: generator training (fixed epochs) ===")
    all_tokens = []
    for s in corpus["selfies"]:
        all_tokens += gen_model.tokenize(s)
    vocab = gen_model.Vocab(all_tokens)
    seqs = []
    for row in corpus.itertuples(index=False):
        ids = gen_model.build_sequence(
            vocab, row.selfies, steered=False, abs_pred=row.abs_pred,
            passes=row.passes_screen, band_lo=args.band_lo, band_hi=args.band_hi)
        if len(ids) >= 3:
            seqs.append(torch.tensor(ids, dtype=torch.long))
    log(f"built {len(seqs)} training sequences, vocab {len(vocab)}")

    model = gen_model.SelfiesLM(len(vocab), args.emb, args.hidden, args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model parameters: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    def batches(seqs, bs, pad_id, seed):
        idx = np.arange(len(seqs))
        np.random.default_rng(seed).shuffle(idx)
        for s in range(0, len(idx), bs):
            chunk = [seqs[i] for i in idx[s:s + bs]]
            yield pad_sequence(chunk, batch_first=True, padding_value=pad_id)

    model.train()
    t0 = time.time()
    steps = 0
    for epoch in range(args.calib_epochs):
        for batch in batches(seqs, args.calib_batch_size, vocab.pad_id, seed=epoch):
            batch = batch.to(device)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            logits, _ = model(inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
            if steps % 10 == 0:
                el = time.time() - t0
                log(f"  train step {steps} loss {float(loss.detach()):.4f} "
                    f"elapsed {el:.1f}s {steps / max(el, 1e-9):.2f} steps/s")
    if device == "cuda":
        torch.cuda.synchronize()
    stage_seconds["generator_train"] = time.time() - t0
    steps_per_sec = steps / max(stage_seconds["generator_train"], 1e-9)
    log(f"generator training: {steps} steps in {stage_seconds['generator_train']:.1f}s "
        f"= {steps_per_sec:.2f} steps/s")

    # ---- Stage 5: generation (measured samples/s) ----
    log("\n=== stage: generation (sampling) ===")
    t0 = time.time()
    samples = gen_model.sample(model, vocab, device, args.calib_gen_samples,
                               steered=False, seed=0)
    if device == "cuda":
        torch.cuda.synchronize()
    stage_seconds["generation"] = time.time() - t0
    samples_per_sec = len(samples) / max(stage_seconds["generation"], 1e-9)
    log(f"generation: {len(samples)} samples in {stage_seconds['generation']:.1f}s "
        f"= {samples_per_sec:.2f} samples/s")

    # ---- Projection for the full configuration (TASK 2) ----
    calib_scale = {
        "uvvis_rows": len(clean), "pred_trees": args.calib_pred_trees,
        "corpus_n": processed, "batch_size": args.calib_batch_size,
    }
    full_scale = {
        "uvvis_rows": args.full_uvvis_rows, "pred_trees": args.full_pred_trees,
        "corpus_n": args.full_corpus_n, "batch_size": args.full_batch_size,
        "n_samples": args.full_n_samples,
    }
    projection = size_full_training(
        steps_per_sec, samples_per_sec, stage_seconds,
        calib_scale, full_scale, keep_ratio, args.target_hours)

    report = {
        "device": device,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "python_version": platform.python_version(),
        "workload": {
            "uvvis_rows": len(clean),
            "pred_trees": args.calib_pred_trees,
            "corpus_processed": processed,
            "corpus_kept": len(corpus),
            "keep_ratio": round(keep_ratio, 4),
            "epochs": args.calib_epochs,
            "batch_size": args.calib_batch_size,
            "n_training_sequences": len(seqs),
            "vocab_size": len(vocab),
            "model_parameters": int(n_params),
            "gen_samples": len(samples),
        },
        "stage_seconds": {k: round(v, 3) for k, v in stage_seconds.items()},
        "throughput": {
            "train_steps_per_sec": round(steps_per_sec, 4),
            "gen_samples_per_sec": round(samples_per_sec, 4),
            "train_steps_measured": steps,
        },
        "full_config": full_scale,
        "projection": projection,
    }

    out_json = args.out_json or os.path.join(args.ckpt_root, "calibration.json")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2)
    log("\n=== calibration summary ===")
    log(json.dumps(report, indent=2))
    log(f"\nwrote {out_json}")
    log(f"projected full-run wall-clock: {projection['projected_total_hours']} h "
        f"at {projection['recommended_gen_epochs']} epochs per arm "
        f"(target {args.target_hours} h)")
    if device != "cuda":
        log("NOTE: no CUDA device present. These are CPU rates and the projection is a "
            "CPU projection. Rerun on the A100 to size the real run.")


if __name__ == "__main__":
    main()
