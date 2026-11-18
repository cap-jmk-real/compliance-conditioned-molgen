"""Measure per-step costs on the current device and project the meta-search wall-clock.

`python -m uv.metasearch_cost <same data args as uv.metasearch>` times, on whatever device
it is given (CPU on the box, the GPU on engine): a REINVENT update at batch 64 and 128, a
self-training round, and the sampling + scoring cost per 1000 evaluation molecules. It
then projects the wall-clock of the schedule in --rungs. The projection is arithmetic on
those measurements. It does not include the one-off holdout-predictor fit.

The GPU is held for the whole wall-clock of a run, so projected wall-clock hours are also
the GPU-hours to book, even though most of each step is CPU work (RDKit decoding,
fingerprints, the boosted surrogate) and short GPU kernels.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uv import steering  # noqa: E402
from uv.metasearch import Context, Trial, build_context, make_parser, parse_rungs, sample_space  # noqa: E402


def survivors_per_rung(n_candidates: int, n_rungs: int, eta: float, n_confirm: int) -> List[int]:
    """Number of candidate trials evaluated at each rung (anchors excluded)."""
    out, n = [], n_candidates
    for r in range(n_rungs):
        out.append(n)
        n = max(1, math.ceil(n / eta)) if r < n_rungs - 2 else min(n_confirm, max(1, math.ceil(n / eta)))
    return out


def project(timings: Dict[str, float], counts: Dict[str, int], rungs: List[Tuple[int, int]],
            eta: float, n_confirm: int, confirm_n_eval: int, n_anchors: int = 2) -> dict:
    """Projected seconds. `timings` keys: reinvent_step_s, selftrain_step_s (amortised
    incl. the sampling round), eval_s_per_1000. Central case: survivors keep the
    candidate mix. Upper case: every trained survivor is the slower method."""
    n_cand = sum(counts.values())
    n_train = counts.get("reinvent", 0) + counts.get("selftrain", 0)
    per_rung = survivors_per_rung(n_cand, len(rungs), eta, n_confirm)
    mix_r = counts.get("reinvent", 0) / max(n_train, 1)
    slow = max(timings["reinvent_step_s"], timings["selftrain_step_s"])
    mix_cost = mix_r * timings["reinvent_step_s"] + (1 - mix_r) * timings["selftrain_step_s"]
    central = upper = 0.0
    prev_steps = 0
    for (steps, n_eval), n_alive in zip(rungs, per_rung):
        frac_trained = n_train / max(n_cand, 1)
        trained_alive = n_alive * frac_trained
        d = steps - prev_steps
        central += trained_alive * d * mix_cost
        upper += n_alive * d * slow
        ev = (n_alive + n_anchors) * n_eval / 1000.0 * timings["eval_s_per_1000"]
        central += ev
        upper += ev
        prev_steps = steps
    final = (min(n_confirm, per_rung[-1]) + n_anchors) * confirm_n_eval / 1000.0 * timings["eval_s_per_1000"]
    central += final
    upper += final
    return {"candidates_per_rung": per_rung, "central_hours": central / 3600.0,
            "upper_hours": upper / 3600.0}


def measure(ctx: Context, reps: int = 6) -> Dict[str, float]:
    which = "plain"
    prior = ctx.frozen(which)
    t: Dict[str, float] = {}
    for batch in (64, 128):
        agent = ctx.fresh(which)
        agent.gru.dropout = 0.0
        opt = torch.optim.Adam(agent.parameters(), lr=1e-4)
        cfg = steering.ReinventConfig(batch=batch)
        steering.reinvent_step(agent, prior, ctx.vocab, ctx.device, [], ctx.scorer, ctx.band,
                               cfg, opt, seed=0)           # warm-up
        t0 = time.time()
        for i in range(reps):
            steering.reinvent_step(agent, prior, ctx.vocab, ctx.device, [], ctx.scorer, ctx.band,
                                   cfg, opt, seed=1 + i)
        t[f"reinvent_step_s_b{batch}"] = (time.time() - t0) / reps
    t["reinvent_step_s"] = float(np.mean([t["reinvent_step_s_b64"], t["reinvent_step_s_b128"]]))

    model = ctx.fresh("steered")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    prefix = ctx.prefix("steered")
    replay = ctx.replay("steered")
    cfg = steering.SelfTrainConfig(steps_per_round=25, gen_per_round=1000, batch=128)
    pool: List[str] = []
    steering.selftrain_round(model, opt, ctx.vocab, ctx.device, prefix, ctx.scorer, ctx.band,
                             dataclasses.replace(cfg, steps_per_round=2), replay, pool, seed=0)
    t0 = time.time()
    steering.selftrain_round(model, opt, ctx.vocab, ctx.device, prefix, ctx.scorer, ctx.band,
                             cfg, replay, pool, seed=1)
    t["selftrain_round_s"] = time.time() - t0
    t["selftrain_step_s"] = t["selftrain_round_s"] / cfg.steps_per_round

    trial = Trial({"method": "prefix", "family": "gru_selfies"}, ctx, "/tmp/_cost_unused", 0)
    trial.evaluate(200, 0)                                    # warm-up
    t0 = time.time()
    trial.evaluate(1000, 1)
    t["eval_s_per_1000"] = time.time() - t0
    return t


def main(argv=None):
    ap = make_parser()
    ap.add_argument("--out", default=None, help="write the timing/projection JSON here")
    args = ap.parse_args(argv)
    os.makedirs(args.root, exist_ok=True)
    ctx = build_context(args)
    print(f"device={ctx.device}; timing a few steps (this is a short probe)", flush=True)
    t = measure(ctx)
    counts = {"guidance": args.n_guidance, "reinvent": args.n_reinvent, "selftrain": args.n_selftrain}
    proj = project(t, counts, parse_rungs(args.rungs), args.eta, args.n_confirm, args.confirm_n_eval)
    report = {"device": ctx.device, "timings_s": {k: round(v, 4) for k, v in t.items()},
              "counts": counts, "rungs": args.rungs, "projection": proj}
    print(json.dumps(report, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
