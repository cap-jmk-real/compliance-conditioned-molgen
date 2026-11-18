"""Round 2 of the steering meta-search: diversity-aware REINVENT with checkpoint selection.

Why a second round. The pilot (uv/metasearch.py, prior seed 1) found no configuration that
passed every pre-registered gate and the holdout check. The best in-band fraction came from
a REINVENT run that was started from the plain prior (reinvent-99e794: surrogate 0.648,
holdout 0.737 at the confirm stage) but its uniqueness was 0.773 against the 0.90 gate, and
uniqueness fell as the optimiser ran longer. Runs started from the steered prior with
a duplicate bucket kept uniqueness but exploited the surrogate (holdout 0.156 and 0.074).

What this round changes. The gates are NOT relaxed (validity 0.95, uniqueness 0.90, novelty
0.90, rule pass 0.95, internal diversity 0.9 x plain, surrogate-minus-holdout gap 0.10). The
search space around the 99e794 region gains:

  scaffold_bucket  Bemis-Murcko scaffold memory (reward 0 once a scaffold is full), in the
                   spirit of Blaschke et al. 2020 and Thomas et al. 2022
  bucket           identical-molecule memory (already in round 1; wider range here)
  entropy_beta     per-token entropy bonus in the loss
  ad_threshold     applicability-domain damping of the reward from similarity to the
                   surrogate's own training molecules (Renz et al. 2019)
  sigma, lr, batch, soft_width, topk_frac   as in round 1; smaller sigma means a stronger
                   anchor to the prior (a larger KL penalty weight)
  eval_temp        sampling temperature at generation time (training samples at 1.0)
  post_w           logit-space blend of the trained agent with its own prior,
                   (1+w) logp_agent - w logp_prior; w < 0 pulls back toward the prior,
                   w > 0 extrapolates (guidance on top of a REINVENT checkpoint)

and checkpoint selection (early stopping): weights are saved every `--snap-every` steps; every
`--eval-every` steps a snapshot is evaluated at the SAME sample size as the final gate
(default 10,000), because uniqueness depends on the sample size (the pilot's 0.857 at 3,000
became 0.773 at 10,000). A trial is represented by its best snapshot that passes the
selection limits (the gate floors plus a small margin), by in-band fraction under the
surrogate. Successive halving runs over cumulative step budgets.

Hygiene. The holdout predictor is computed for every snapshot and reported, but it is never
used for ranking or selection. It is used once, as a pass/fail gate, on fresh samples of
the top candidates (`confirm`). The reward uses the fixed seed-0 surrogate only; the
structural screen is never part of a reward (see uv/steering.py).

Modes
  search   successive halving, refine, confirm, report (resumable)
  seeds    retrain one chosen configuration from the prior of each seed in --seed-priors,
           stopping at the chosen step, evaluate 10,000 fresh samples each, report mean +- sd
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uv import gen_model, metasearch, steering  # noqa: E402
from uv.metasearch import (Context, Limits, Trial, anchors, atomic_json,  # noqa: E402
                           atomic_save, parse_rungs, rank_key, trial_id, violations)
from uv.steer_eval import compute_metrics  # noqa: E402

# Parameters a round-2 spec always carries explicitly (so a spec is self-describing and its
# id does not depend on library defaults).
BASE_SPEC = {
    "method": "reinvent", "family": "gru_selfies", "init": "plain",
    "sigma": 30.0, "lr": 1e-4, "batch": 128, "soft_width": 25.0, "topk_frac": 1.0,
    "bucket": 0, "scaffold_bucket": 0, "entropy_beta": 0.0, "ad_threshold": 0.0,
    "post_w": 0.0, "eval_temp": 1.0,
}

GRID = {
    "init": ["plain", "plain", "steered"],
    "sigma": [10.0, 20.0, 30.0, 50.0],
    "lr": [5e-5, 1e-4, 2e-4],
    "batch": [128, 256],
    "soft_width": [15.0, 25.0, 40.0],
    "topk_frac": [1.0, 0.5],
    "bucket": [0, 1, 2, 3],
    "scaffold_bucket": [0, 10, 25, 50],
    "entropy_beta": [0.0, 5.0, 20.0],
    "ad_threshold": [0.0, 0.35, 0.5],
    "post_w": [0.0, -0.25, -0.5, 0.25],
    "eval_temp": [1.0, 1.1],
}


def spec_id(spec: dict) -> str:
    return trial_id(spec)


def sample_space_r2(n_local: int, n_global: int, seed: int) -> List[dict]:
    """The reference spec (pilot reinvent-99e794 with the new knobs at their defaults),
    `n_local` neighbours that change 1 to 3 knobs of it, and `n_global` random draws.
    Deterministic in `seed`; specs are distinct."""
    rng = np.random.default_rng(seed)
    keys = sorted(GRID)
    out, seen = [dict(BASE_SPEC)], {spec_id(BASE_SPEC)}

    def add(spec):
        sid = spec_id(spec)
        if sid not in seen:
            seen.add(sid)
            out.append(spec)
            return True
        return False

    tries = 0
    got = 0
    while got < n_local and tries < 50 * max(n_local, 1):
        tries += 1
        spec = dict(BASE_SPEC)
        for k in rng.choice(keys, size=int(rng.integers(1, 4)), replace=False):
            choices = [v for v in GRID[k] if v != BASE_SPEC[k]]
            spec[k] = choices[int(rng.integers(len(choices)))]
        got += add(spec)
    tries = 0
    got = 0
    while got < n_global and tries < 50 * max(n_global, 1):
        tries += 1
        spec = dict(BASE_SPEC)
        for k in keys:
            spec[k] = GRID[k][int(rng.integers(len(GRID[k])))]
        got += add(spec)
    return out


def reinvent_cfg(spec: dict, max_len: int) -> steering.ReinventConfig:
    return steering.ReinventConfig(
        sigma=spec["sigma"], lr=spec["lr"], batch=spec["batch"], soft_width=spec["soft_width"],
        topk_frac=spec["topk_frac"], bucket=spec["bucket"],
        scaffold_bucket=spec.get("scaffold_bucket", 0), entropy_beta=spec.get("entropy_beta", 0.0),
        ad_threshold=spec.get("ad_threshold", 0.0), max_len=max_len)


# ---------------------------------------------------------------------------
# Trial with snapshots
# ---------------------------------------------------------------------------

class Trial2(Trial):
    """REINVENT trial that writes a weights snapshot every `snap_every` steps."""

    def __init__(self, spec, ctx, root, base_seed, snap_every=50):
        super().__init__(spec, ctx, root, base_seed)
        self.snap_every = snap_every
        self.scaffolds: Dict[str, int] = {}
        self._eval_model = None
        self._eval_step = None

    def snap_path(self, step: int) -> str:
        return os.path.join(self.dir, "snaps", f"step-{step:05d}.pt")

    def load(self):
        had = self.model is not None
        super().load()
        if had or not self.trained:
            return
        if os.path.exists(self._ckpt):
            st = torch.load(self._ckpt, map_location=self.ctx.device, weights_only=False)
            self.scaffolds = st.get("scaffolds", {})

    def save(self):
        atomic_save({"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                     "steps": self.steps, "rounds": self.rounds, "pool": [], "seen": self.seen,
                     "scaffolds": self.scaffolds, "log": self.log[-200:],
                     "train_seconds": self.train_seconds, "spec": self.spec}, self._ckpt)

    def advance_to(self, target: int, print_every: int = 50):
        if self.steps >= target:
            return
        ctx, spec = self.ctx, self.spec
        cfg = reinvent_cfg(spec, ctx.max_len)
        prior = ctx.frozen(spec["init"])
        prefix = ctx.prefix(spec["init"])
        ad = getattr(ctx, "ad", None)
        t0, start = time.time(), self.steps
        while self.steps < target:
            stats = steering.reinvent_step(
                self.model, prior, ctx.vocab, ctx.device, prefix, ctx.scorer, ctx.band, cfg,
                self.opt, seed=self.seed * 1000003 + self.steps, seen=self.seen,
                scaffolds=self.scaffolds, ad=ad)
            self.steps += 1
            self.log.append({"step": self.steps, **stats})
            if self.steps % self.snap_every == 0 or self.steps == target:
                if not os.path.exists(self.snap_path(self.steps)):
                    atomic_save({k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                                self.snap_path(self.steps))
                self.train_seconds += time.time() - t0
                t0 = time.time()
                self.save()
            if self.steps % print_every == 0:
                rate = (self.steps - start) / max(self.train_seconds, 1e-9)
                print(f"    [{self.id}] step {self.steps}/{target} | in-band batch "
                      f"{stats['in_band_batch']:.3f} | unique {stats['unique_batch']:.3f} | "
                      f"kl {stats['kl_nats']:.2f}", flush=True)

    # -- snapshot sampling ---------------------------------------------------
    def _load_snapshot(self, step: int):
        if self._eval_step == step:
            return self._eval_model
        if self._eval_model is None:
            self._eval_model = self.ctx.fresh(self.spec["init"])
        sd = torch.load(self.snap_path(step), map_location=self.ctx.device, weights_only=False)
        self._eval_model.load_state_dict(sd)
        self._eval_model.eval()
        self._eval_step = step
        return self._eval_model

    def sample_snapshot(self, step: int, n: int, seed: int) -> List[str]:
        ctx, spec = self.ctx, self.spec
        model = self._load_snapshot(step) if step > 0 else ctx.frozen(spec["init"])
        w = float(spec.get("post_w", 0.0))
        temp = float(spec.get("eval_temp", 1.0))
        experts = steering.blend_experts(model, ctx.frozen(spec["init"]),
                                         ctx.prefix(spec["init"]), w)
        out: List[str] = []
        for i, start in enumerate(range(0, n, 512)):
            b = min(512, n - start)
            tok, ln = steering.sample_tokens(experts, ctx.vocab, ctx.device, b, ctx.max_len,
                                             temp, seed * 7919 + i, mask_special=(w != 0.0))
            out += steering.tokens_to_selfies(ctx.vocab, tok, ln)
        return out

    def evaluate_snapshot(self, step: int, n: int, seed: int) -> dict:
        t0 = time.time()
        selfies_list = self.sample_snapshot(step, n, seed)
        m = compute_metrics(selfies_list, self.ctx.train_set, self.ctx.scorer, self.ctx.band[0],
                            self.ctx.band[1],
                            holdout=self.ctx.holdout.predict if self.ctx.holdout else None,
                            seed=seed)
        m["eval_seconds"] = round(time.time() - t0, 2)
        tail = [r for r in self.log if step - 20 < r["step"] <= step and "kl_nats" in r]
        if tail:
            m["kl_nats_recent"] = round(float(np.mean([r["kl_nats"] for r in tail])), 3)
        m.update({"trial": self.id, "step": step, "n_eval": n})
        return m


# ---------------------------------------------------------------------------
# Selection logic (pure functions, unit-tested)
# ---------------------------------------------------------------------------

def selection_limits(lim: Limits, margin: float) -> Limits:
    """Gate floors plus a small safety margin, used ONLY to choose which snapshot and
    trial to carry forward. The gates themselves (confirm stage) use `lim` unchanged."""
    up = lambda x: min(1.0, x + margin)  # noqa: E731
    return Limits(validity=up(lim.validity), uniqueness=up(lim.uniqueness),
                  novelty=up(lim.novelty), rule_pass=up(lim.rule_pass),
                  div_ratio=lim.div_ratio + margin, holdout_gap=lim.holdout_gap)


def best_snapshot(snaps: Dict[int, dict], lim: Limits, plain_div: Optional[float]
                  ) -> Optional[Tuple[int, dict, dict]]:
    """(step, metrics, violations) of the best snapshot: feasible first, then highest
    in-band fraction (surrogate); infeasible ones by smallest total shortfall."""
    best, best_key = None, None
    for step, m in snaps.items():
        v = violations(m, lim, plain_div)
        key = rank_key(m, v)
        if best_key is None or key < best_key:
            best, best_key = (step, m, v), key
    return best


def gate_report(m: dict, lim: Limits, plain_div: Optional[float]) -> dict:
    """Full gate evaluation of one metrics dict with the pre-registered limits."""
    viol = violations(m, lim, plain_div)
    gap = None
    if "in_band_holdout" in m:
        gap = round(m["in_band_fraction"] - m["in_band_holdout"], 5)
    holdout_ok = gap is not None and gap <= lim.holdout_gap
    return {"violations": viol, "feasible": not viol, "holdout_gap": gap,
            "holdout_ok": holdout_ok, "passes_all": (not viol) and holdout_ok}


def frontier(rows: Sequence[dict], lim: Limits, plain_div: Optional[float],
             levels=(0.70, 0.75, 0.80, 0.85, 0.90, 0.95)) -> List[dict]:
    """Best in-band fraction at each uniqueness level, among snapshots that meet every
    OTHER gate (validity, novelty, rule pass, internal diversity) and the holdout gap."""
    relaxed = dataclasses.replace(lim, uniqueness=0.0)
    ok = []
    for m in rows:
        if violations(m, relaxed, plain_div):
            continue
        g = gate_report(m, relaxed, plain_div)
        if not g["holdout_ok"]:
            continue
        ok.append(m)
    out = []
    for u in levels:
        cand = [m for m in ok if m["uniqueness"] >= u]
        if not cand:
            out.append({"uniqueness_at_least": u, "best": None})
            continue
        m = max(cand, key=lambda z: z["in_band_fraction"])
        out.append({"uniqueness_at_least": u, "best": {
            "trial": m["trial"], "step": m["step"], "in_band": m["in_band_fraction"],
            "in_band_holdout": m.get("in_band_holdout"), "uniqueness": m["uniqueness"],
            "int_div": m["int_div"], "n_eval": m.get("n_eval")}})
    return out


# ---------------------------------------------------------------------------
# Driver: search
# ---------------------------------------------------------------------------

def mean_sd(x: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(list(x), dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else float("nan")


class Search2:
    def __init__(self, ctx: Context, root: str, args):
        self.ctx, self.root, self.args = ctx, root, args
        self.rungs = [int(x) for x in args.rungs.split(",")]
        self.path = os.path.join(root, "snap_results.json")
        self.res: Dict[str, dict] = {}
        if os.path.exists(self.path):
            with open(self.path) as fh:
                self.res = json.load(fh)
            print(f"resuming: {len(self.res)} evaluations on disk", flush=True)

    @staticmethod
    def key(tid, step, stage="snap"):
        return f"{stage}|{tid}|{step}"

    def eval_snap(self, tr: Trial2, step: int, stage="snap", seed=None, n=None):
        k = self.key(tr.id, step, stage)
        if k in self.res:
            return self.res[k]
        a = self.args
        if seed is None:
            seed = a.seed * 1000 + step // max(a.snap_every, 1)
        m = tr.evaluate_snapshot(step, n or a.snap_n_eval, seed)
        m["stage"] = stage
        self.res[k] = m
        atomic_json(self.res, self.path)
        return m

    def snaps_of(self, tid) -> Dict[int, dict]:
        return {m["step"]: m for k, m in self.res.items() if k.startswith(f"snap|{tid}|")}

    def run(self):
        a, ctx = self.args, self.ctx
        lim = ctx.limits
        sel = selection_limits(lim, a.sel_margin)
        specs = sample_space_r2(a.n_local, a.n_global, a.seed)
        trials = {spec_id(s): Trial2(s, ctx, self.root, a.seed, a.snap_every) for s in specs}
        anchor_specs = anchors()
        anchor_ids = [trial_id(s) for s in anchor_specs]
        atomic_json({"args": vars(a), "specs": {t: tr.spec for t, tr in trials.items()},
                     "band": list(ctx.band), "limits": asdict(lim),
                     "selection_limits": asdict(sel)}, os.path.join(self.root, "config.json"))
        ref_id = spec_id(BASE_SPEC)
        print(f"round 2: {len(trials)} REINVENT trials (reference {ref_id}), rungs {self.rungs} "
              f"steps, snapshot every {a.snap_every}, evaluated every {a.eval_every} at "
              f"n={a.snap_n_eval}, eta {a.eta}", flush=True)
        t_start = time.time()

        # anchors: plain and prefix at the same n and seed as the snapshots
        anchor_res = {}
        for s in anchor_specs:
            tid = trial_id(s)
            tr = metasearch.Trial(s, ctx, self.root, a.seed)
            k = f"anchor|{tid}"
            if k not in self.res:
                m = tr.evaluate(a.snap_n_eval, a.seed * 1000 + 999)
                m.update({"trial": tid, "stage": "anchor", "n_eval": a.snap_n_eval, "step": 0})
                self.res[k] = m
                atomic_json(self.res, self.path)
            anchor_res[tid] = self.res[k]
        plain_div = anchor_res[anchor_ids[0]]["int_div"]
        print(f"anchors at n={a.snap_n_eval}: plain in-band {anchor_res[anchor_ids[0]]['in_band_fraction']:.3f} "
              f"int_div {plain_div:.3f}; prefix in-band {anchor_res[anchor_ids[1]]['in_band_fraction']:.3f}",
              flush=True)

        alive = list(trials)
        history = []
        for r, steps in enumerate(self.rungs):
            print(f"\n=== rung{r}: {steps} steps, {len(alive)} trials ===", flush=True)
            for i, tid in enumerate(alive, 1):
                tr = trials[tid]
                tr.load()
                tr.advance_to(steps)
                grid = [s for s in range(a.eval_every, steps + 1, a.eval_every)]
                for s in grid:
                    if s <= tr.steps:
                        self.eval_snap(tr, s)
                bs = best_snapshot(self.snaps_of(tid), sel, plain_div)
                if bs:
                    st, m, v = bs
                    print(f"  [rung{r} {i}/{len(alive)}] {tid:<18} best step {st} in-band "
                          f"{m['in_band_fraction']:.3f} uniq {m['uniqueness']:.3f} div "
                          f"{m['int_div']:.3f} {'ok' if not v else 'viol ' + ','.join(v)} "
                          f"(elapsed {(time.time() - t_start) / 60:.1f} min)", flush=True)
                tr.model = None   # free GPU memory between trials
                tr.opt = None
            scored = []
            for tid in alive:
                bs = best_snapshot(self.snaps_of(tid), sel, plain_div)
                scored.append((rank_key(bs[1], bs[2]), tid))
            scored.sort()
            n_keep = len(alive) if r == len(self.rungs) - 1 else max(1, math.ceil(len(alive) / a.eta))
            kept = [t for _, t in scored[:n_keep]]
            history.append({"rung": r, "steps": steps, "evaluated": alive, "kept": kept})
            atomic_json(history, os.path.join(self.root, "history.json"))
            print(f"  gate: kept {len(kept)}/{len(alive)}: {', '.join(kept)}", flush=True)
            alive = kept

        # refine: evaluate the in-between snapshots of the best trials
        ranked = self.rank_trials(trials, sel, plain_div)
        for tid in ranked[:a.n_refine]:
            tr = trials[tid]
            tr.load()
            for s in range(a.snap_every, tr.steps + 1, a.snap_every):
                if os.path.exists(tr.snap_path(s)):
                    self.eval_snap(tr, s)
            tr.model = None
            tr.opt = None
        ranked = self.rank_trials(trials, sel, plain_div)

        # confirm: fresh samples, the actual gates, the holdout gate
        print(f"\n=== confirm: top {a.n_confirm} trials' best snapshot, n={a.confirm_n_eval}, fresh seed ===",
              flush=True)
        confirmed = []
        for tid in ranked[:a.n_confirm]:
            tr = trials[tid]
            tr.load()
            step, _, _ = best_snapshot(self.snaps_of(tid), sel, plain_div)
            m = self.eval_snap(tr, step, stage="confirm", seed=a.seed * 1000 + 777 + step,
                               n=a.confirm_n_eval)
            confirmed.append((tid, step))
            print(f"  [confirm] {tid:<18} step {step} in-band {m['in_band_fraction']:.3f} "
                  f"holdout {m.get('in_band_holdout', float('nan')):.3f} uniq {m['uniqueness']:.3f}",
                  flush=True)
            tr.model = None
            tr.opt = None
        # anchors at the confirm sample size and seed family
        for s in anchor_specs:
            tid = trial_id(s)
            tr = metasearch.Trial(s, ctx, self.root, a.seed)
            k = self.key(tid, 0, "confirm")
            if k not in self.res:
                m = tr.evaluate(a.confirm_n_eval, a.seed * 1000 + 777)
                m.update({"trial": tid, "stage": "confirm", "n_eval": a.confirm_n_eval, "step": 0})
                self.res[k] = m
                atomic_json(self.res, self.path)
        write_report(self, trials, anchor_ids, history, confirmed, plain_div, sel)
        print(f"\ndone in {(time.time() - t_start) / 60:.1f} min. results: "
              f"{os.path.join(self.root, 'results_round2.md')}", flush=True)

    def rank_trials(self, trials, sel, plain_div) -> List[str]:
        scored = []
        for tid in trials:
            snaps = self.snaps_of(tid)
            if not snaps:
                continue
            _, m, v = best_snapshot(snaps, sel, plain_div)
            scored.append((rank_key(m, v), tid))
        scored.sort()
        return [t for _, t in scored]


def _f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def write_report(search: Search2, trials, anchor_ids, history, confirmed, plain_div_snap, sel):
    ctx, lim, rr = search.ctx, search.ctx.limits, search.res
    plain_c = rr[search.key(anchor_ids[0], 0, "confirm")]
    rows = []
    for tid, step in confirmed:
        m = rr[search.key(tid, step, "confirm")]
        g = gate_report(m, lim, plain_c["int_div"])
        rows.append({"trial": tid, "step": step, "spec": trials[tid].spec, **m, **g})
    for tid in anchor_ids:
        m = rr[search.key(tid, 0, "confirm")]
        g = gate_report(m, lim, plain_c["int_div"])
        spec = next(x for x in anchors() if trial_id(x) == tid)
        rows.append({"trial": tid, "step": 0, "spec": spec, **m, **g})
    winners = [x for x in rows if x["trial"] not in anchor_ids and x["passes_all"]]
    winners.sort(key=lambda x: -x["in_band_fraction"])
    snap_rows = [m for k, m in rr.items() if k.startswith("snap|")]
    fr = frontier(snap_rows, lim, plain_div_snap)
    doc = {"band_nm": list(ctx.band), "limits": asdict(lim), "selection_limits": asdict(sel),
           "rungs": search.rungs, "plain_int_div_confirm": plain_c["int_div"],
           "plain_int_div_snapshots": plain_div_snap, "confirm": rows,
           "winner": None if not winners else {"trial": winners[0]["trial"],
                                               "step": winners[0]["step"],
                                               "spec": winners[0]["spec"]},
           "frontier_snapshots": fr, "history": history,
           "holdout_predictor": getattr(ctx.holdout, "metrics", None)}
    atomic_json(doc, os.path.join(search.root, "results_round2.json"))
    md = ["# Steering meta-search, round 2", "",
          f"Band {ctx.band[0]:.0f} to {ctx.band[1]:.0f} nm, surrogate = fixed seed-0 UV predictor. "
          f"Gates (not relaxed): validity >= {lim.validity}, uniqueness >= {lim.uniqueness}, "
          f"novelty >= {lim.novelty}, rule pass >= {lim.rule_pass}, internal diversity >= "
          f"{lim.div_ratio} x plain, holdout gap <= {lim.holdout_gap}. "
          f"Selection (not gate) margin {search.args.sel_margin}.", ""]
    md += ["## Confirmation (fresh samples)", "",
           "| trial | step | in-band (surr) | in-band (holdout) | gap | valid | uniq | novel | rule | "
           "int_div | kl | passes all | note |", "|" + "---|" * 13]
    for x in rows:
        md.append(f"| {x['trial']} | {x['step']} | {_f(x['in_band_fraction'])} | "
                  f"{_f(x.get('in_band_holdout'))} | {_f(x['holdout_gap'])} | {_f(x['validity'])} | "
                  f"{_f(x['uniqueness'])} | {_f(x['novelty'])} | {_f(x['rule_pass_rate'])} | "
                  f"{_f(x['int_div'])} | {_f(x.get('kl_nats_recent'), 2)} | "
                  f"{'YES' if x['passes_all'] else 'no'} | "
                  f"{('fails ' + ','.join(x['violations'])) if x['violations'] else ''}"
                  f"{'' if x['holdout_ok'] else ' holdout gap'} |")
    md += ["", f"Winner: {doc['winner'] or 'none'}", "",
           "## Frontier over all evaluated snapshots (n per snapshot as set in config)", "",
           "Best surrogate in-band fraction at each uniqueness level, among snapshots meeting all "
           "other gates and the holdout gap.", "",
           "| uniqueness >= | trial | step | in-band (surr) | in-band (holdout) | uniq | int_div |",
           "|---|---|---|---|---|---|---|"]
    for f in fr:
        b = f["best"]
        md.append(f"| {f['uniqueness_at_least']} | " + (
            "none | | | | | |" if b is None else
            f"{b['trial']} | {b['step']} | {_f(b['in_band'])} | {_f(b['in_band_holdout'])} | "
            f"{_f(b['uniqueness'])} | {_f(b['int_div'])} |"))
    md += ["", "## Trial hyper-parameters (confirmed trials)", "", "| trial | hparams |", "|---|---|"]
    for tid, _ in confirmed:
        sp = trials[tid].spec
        md.append(f"| {tid} | " + " ".join(f"{k}={sp[k]}" for k in sorted(sp)
                                           if k not in ("method", "family")) + " |")
    with open(os.path.join(search.root, "results_round2.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")


# ---------------------------------------------------------------------------
# Driver: multi-seed confirmation of one configuration
# ---------------------------------------------------------------------------

def run_seeds(args):
    import joblib
    from uv import holdout_predictor
    spec = json.loads(args.spec_json) if args.spec_json.strip().startswith("{") else \
        json.load(open(args.spec_json))
    if "spec" in spec:
        spec = spec["spec"]
    steps = int(args.steps)
    priors = [x.split(":") for x in args.seed_priors.split(",")]   # seed:plain.pt:steered.pt
    os.makedirs(args.root, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    vocab = gen_model.Vocab.load(args.vocab)
    df = pd.read_csv(args.corpus)
    scorer = steering.UVScorer.from_bundle(joblib.load(args.predictor))
    holdout = holdout_predictor.load_or_train(
        os.path.join(args.root, "holdout_predictor.joblib"), args.uvvis_data,
        seed=args.holdout_seed, n_estimators=args.holdout_trees)
    lim = Limits(validity=args.min_validity, uniqueness=args.min_uniqueness,
                 novelty=args.min_novelty, rule_pass=args.min_rule_pass,
                 div_ratio=args.div_ratio, holdout_gap=args.max_holdout_gap)
    ad = None
    if spec.get("ad_threshold", 0.0) > 0:
        ad = steering.ApplicabilityDomain(pd.read_csv(args.uvvis_data)["smiles"].tolist())
    per_seed = []
    for entry in priors:
        sd, plain_p, steered_p = int(entry[0]), entry[1], entry[2]
        ctx = Context(vocab, plain_p, steered_p, df, scorer, holdout, (args.band_lo, args.band_hi),
                      device, lim, replay_n=args.replay_n)
        ctx.ad = ad
        root = os.path.join(args.root, f"prior-seed{sd}")
        tr = Trial2(dict(spec), ctx, root, base_seed=1000 + sd, snap_every=args.snap_every)
        tr.load()
        tr.advance_to(steps)
        seed_eval = args.seed * 1000 + 31 * sd + 5
        t0 = time.time()
        m = tr.evaluate_snapshot(steps, args.confirm_n_eval, seed_eval)
        plain = metasearch.Trial({"method": "plain", "family": "gru_selfies"}, ctx, root, 0)
        pm = plain.evaluate(args.confirm_n_eval, seed_eval)
        g = gate_report(m, lim, pm["int_div"])
        per_seed.append({"prior_seed": sd, "steps": steps, "metrics": m, "plain": pm, "gates": g})
        print(f"[seed {sd}] in-band {m['in_band_fraction']:.3f} holdout "
              f"{m.get('in_band_holdout', float('nan')):.3f} uniq {m['uniqueness']:.3f} div "
              f"{m['int_div']:.3f} (plain {pm['in_band_fraction']:.3f}/{pm['int_div']:.3f}) "
              f"passes_all={g['passes_all']} ({time.time() - t0:.0f}s)", flush=True)
        atomic_json({"spec": spec, "steps": steps, "per_seed": per_seed},
                    os.path.join(args.root, "seeds_partial.json"))
        tr.model = None
    keys = ["in_band_fraction", "in_band_holdout", "validity", "uniqueness", "novelty",
            "rule_pass_rate", "int_div", "yield_per_sample"]
    summary = {k: mean_sd([p["metrics"][k] for p in per_seed]) for k in keys}
    summary["plain_in_band"] = mean_sd([p["plain"]["in_band_fraction"] for p in per_seed])
    summary["plain_int_div"] = mean_sd([p["plain"]["int_div"] for p in per_seed])
    gaps = [p["gates"]["holdout_gap"] for p in per_seed]
    summary["holdout_gap"] = mean_sd(gaps)
    doc = {"spec": spec, "steps": steps, "n_eval": args.confirm_n_eval, "limits": asdict(lim),
           "per_seed": per_seed, "summary_mean_sd": summary,
           "all_seeds_pass_all": all(p["gates"]["passes_all"] for p in per_seed)}
    atomic_json(doc, os.path.join(args.root, "seeds_result.json"))
    md = ["# Multi-seed confirmation", "",
          f"Config `{trial_id(spec)}` at {steps} steps, {args.confirm_n_eval} fresh samples per prior seed. "
          f"Fixed seed-0 surrogate. Gates unchanged.", "",
          "| prior seed | in-band (surr) | in-band (holdout) | gap | valid | uniq | novel | rule | int_div "
          "| plain in-band | plain int_div | passes all |", "|" + "---|" * 12]
    for p in per_seed:
        m, g = p["metrics"], p["gates"]
        md.append(f"| {p['prior_seed']} | {_f(m['in_band_fraction'])} | {_f(m.get('in_band_holdout'))} | "
                  f"{_f(g['holdout_gap'])} | {_f(m['validity'])} | {_f(m['uniqueness'])} | "
                  f"{_f(m['novelty'])} | {_f(m['rule_pass_rate'])} | {_f(m['int_div'])} | "
                  f"{_f(p['plain']['in_band_fraction'])} | {_f(p['plain']['int_div'])} | "
                  f"{'YES' if g['passes_all'] else 'no: ' + ','.join(g['violations'])}"
                  f"{'' if g['holdout_ok'] else ' holdout gap'} |")
    md.append("| mean +- sd | " + " | ".join(
        f"{_f(summary[k][0])} +- {_f(summary[k][1])}" for k in
        ("in_band_fraction", "in_band_holdout", "holdout_gap", "validity", "uniqueness", "novelty",
         "rule_pass_rate", "int_div", "plain_in_band", "plain_int_div")) +
        f" | {'all' if doc['all_seeds_pass_all'] else 'not all'} |")
    with open(os.path.join(args.root, "seeds_result.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["search", "seeds"], default="search")
    ap.add_argument("--root", required=True)
    ap.add_argument("--plain")
    ap.add_argument("--steered")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--uvvis-data", required=True)
    ap.add_argument("--band-lo", type=float, default=400.0)
    ap.add_argument("--band-hi", type=float, default=450.0)
    ap.add_argument("--n-local", type=int, default=40)
    ap.add_argument("--n-global", type=int, default=55)
    ap.add_argument("--rungs", default="300,600,1200", help="cumulative steps per rung")
    ap.add_argument("--eta", type=float, default=3.0)
    ap.add_argument("--snap-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--snap-n-eval", type=int, default=10000)
    ap.add_argument("--sel-margin", type=float, default=0.01)
    ap.add_argument("--n-refine", type=int, default=6)
    ap.add_argument("--n-confirm", type=int, default=8)
    ap.add_argument("--confirm-n-eval", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--replay-n", type=int, default=20000)
    ap.add_argument("--holdout-seed", type=int, default=1337)
    ap.add_argument("--holdout-trees", type=int, default=300)
    ap.add_argument("--min-validity", type=float, default=0.95)
    ap.add_argument("--min-uniqueness", type=float, default=0.90)
    ap.add_argument("--min-novelty", type=float, default=0.90)
    ap.add_argument("--min-rule-pass", type=float, default=0.95)
    ap.add_argument("--div-ratio", type=float, default=0.90)
    ap.add_argument("--max-holdout-gap", type=float, default=0.10)
    ap.add_argument("--cpu", action="store_true")
    # seeds mode
    ap.add_argument("--spec-json", default="", help="spec json (file or inline) for --mode seeds")
    ap.add_argument("--steps", type=int, default=0, help="chosen step count for --mode seeds")
    ap.add_argument("--seed-priors", default="",
                    help="comma list of seed:plain.pt:steered.pt for --mode seeds")
    return ap


def build_context2(args) -> Context:
    ctx = metasearch.build_context(args)
    ref = pd.read_csv(args.uvvis_data)["smiles"].tolist()
    ctx.ad = steering.ApplicabilityDomain(ref)
    return ctx


def main(argv=None):
    args = make_parser().parse_args(argv)
    os.makedirs(args.root, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.mode == "seeds":
        return run_seeds(args)
    ctx = build_context2(args)
    print(f"device={ctx.device} band={ctx.band} corpus={len(ctx.corpus)}", flush=True)
    Search2(ctx, args.root, args).run()


if __name__ == "__main__":
    main()
