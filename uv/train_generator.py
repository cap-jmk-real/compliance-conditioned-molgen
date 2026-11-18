"""Train one arm of the SELFIES generative model: plain or steered.

Both arms share the architecture (gen_model.SelfiesLM) and the training budget
(epochs, batch size, learning rate, max sequence length). The only difference is the
control tokens: the steered arm prepends a band token and a structural-screen token
to every training sequence, so at sampling time the model can be asked for molecules
in the target absorption band that also pass the screen.

Checkpoints go to a configurable run directory (default under a mounted Google Drive
path), are written every --ckpt-every steps, and the script resumes from the latest.
Progress is printed each heartbeat: step, epoch, loss, elapsed, ETA.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from colab_checkpoint import CheckpointStore, Heartbeat  # noqa: E402
from uv import gen_model  # noqa: E402


def set_seed(seed):
    """Seed every RNG that affects a training run so a given --seed is reproducible.

    Covers Python ``random``, NumPy, torch (CPU) and CUDA. This seeds model weight
    initialisation and any torch sampling; the data-shuffle RNG is seeded separately
    per epoch in ``batches`` with an offset derived from this seed. We deliberately do
    not force cuDNN into fully deterministic mode: the goal is independent, seed-varied
    training runs to quantify training variance, not bitwise determinism, and the
    seed-0 run that the paper reports did not pin cuDNN either.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sequences(df, vocab, steered, band_lo, band_hi):
    seqs = []
    for row in df.itertuples(index=False):
        ids = gen_model.build_sequence(
            vocab, row.selfies, steered,
            abs_pred=getattr(row, "abs_pred", None),
            passes=getattr(row, "passes_screen", None),
            band_lo=band_lo, band_hi=band_hi,
        )
        if len(ids) >= 3:
            seqs.append(torch.tensor(ids, dtype=torch.long))
    return seqs


def batches(seqs, batch_size, pad_id, shuffle=True, seed=0):
    idx = np.arange(len(seqs))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for s in range(0, len(idx), batch_size):
        chunk = [seqs[i] for i in idx[s:s + batch_size]]
        yield pad_sequence(chunk, batch_first=True, padding_value=pad_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="uv/data/corpus.csv")
    ap.add_argument("--arm", choices=["plain", "steered"], required=True)
    ap.add_argument("--vocab", default="uv/artifacts/vocab.json")
    ap.add_argument("--out", default=None, help="output model path (defaults by arm)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--band-lo", type=float, default=450.0)
    ap.add_argument("--band-hi", type=float, default=550.0)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--heartbeat-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0,
                    help="run seed; seeds python/numpy/torch/cuda, the data-shuffle "
                         "order, and (when --run-dir is not given) the checkpoint "
                         "directory name. seed 0 reproduces the original run.")
    args = ap.parse_args()

    set_seed(args.seed)

    out = args.out or f"uv/artifacts/generator_{args.arm}.pt"
    seed_suffix = "" if args.seed == 0 else f"-seed{args.seed}"
    run_dir = (args.run_dir
               or f"/content/drive/MyDrive/t25-checkpoints/uv-generator-{args.arm}{seed_suffix}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"arm={args.arm} device={device} seed={args.seed}", flush=True)

    df = pd.read_csv(args.corpus)
    print(f"corpus {len(df)} molecules", flush=True)

    # The vocabulary is built once from all corpus tokens and shared by both arms.
    if os.path.exists(args.vocab):
        vocab = gen_model.Vocab.load(args.vocab)
        print(f"loaded vocab size {len(vocab)}", flush=True)
    else:
        all_tokens = []
        for s in df["selfies"]:
            all_tokens += gen_model.tokenize(s)
        vocab = gen_model.Vocab(all_tokens)
        os.makedirs(os.path.dirname(args.vocab), exist_ok=True)
        vocab.save(args.vocab)
        print(f"built vocab size {len(vocab)} -> {args.vocab}", flush=True)

    steered = args.arm == "steered"
    seqs = make_sequences(df, vocab, steered, args.band_lo, args.band_hi)
    print(f"built {len(seqs)} training sequences", flush=True)

    model = gen_model.SelfiesLM(len(vocab), args.emb, args.hidden, args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    steps_per_epoch = (len(seqs) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    ck = CheckpointStore(run_dir, suffix=".pt",
                         save_fn=lambda o, p: torch.save(o, p),
                         load_fn=lambda p: torch.load(p, map_location=device, weights_only=False))
    state = ck.latest()
    start_step = 0
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_step = state["step"]
        print(f"resumed at step {start_step}", flush=True)

    hb = Heartbeat(total_steps=total_steps, every=args.heartbeat_every)
    step = start_step
    model.train()
    for epoch in range(args.epochs):
        for batch in batches(seqs, args.batch_size, vocab.pad_id,
                              seed=args.seed * 100_000 + epoch):
            if step < start_step:
                step += 1
                continue
            batch = batch.to(device)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            logits, _ = model(inp)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            hb.tick(step, epoch=epoch, loss=round(float(loss.detach()), 4))
            if step % args.ckpt_every == 0:
                ck.save({"model": model.state_dict(), "opt": opt.state_dict(),
                         "step": step, "arm": args.arm}, step=step,
                        metric=float(loss.detach()), lower_is_better=True)
    ck.save({"model": model.state_dict(), "opt": opt.state_dict(),
             "step": step, "arm": args.arm}, step=step,
            metric=float(loss.detach()), lower_is_better=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"model": model.state_dict(), "arm": args.arm, "vocab": args.vocab,
                "emb": args.emb, "hidden": args.hidden, "layers": args.layers,
                "band_lo": args.band_lo, "band_hi": args.band_hi}, out)
    print(f"final loss {float(loss.detach()):.4f} after {step} steps", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
