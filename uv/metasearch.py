"""Meta-search over steering method x hyper-parameters, by successive halving.

Search object: how to steer the UV-Vis SELFIES generator toward a band, from the
existing pre-trained checkpoints (plain prior, steered prior) under the fixed seed-0
UV surrogate. Method pool (see uv/steering.py and uv/STEERING_PLAN.md):

  guidance   sampling-time logit guidance, strength w and contrast kind     (no training)
  reinvent   REINVENT-style fine-tuning with a prior penalty                (trained)
  selftrain  surrogate-filtered self-training with corpus replay            (trained)

plus two untrained anchors that are always evaluated: `plain` and `prefix` (the
existing steered arm). This is the shape of the DenoiseOpt bake-off described in
OBJECTIVES_AND_MODELS.md Section 4 (same protocol for every candidate, stage-gated,
losers reported, pilot-to-full rank correlation reported), applied to the steering
method instead of the generator family. Only one generator family (the GRU SELFIES LM)
is pre-trained, so the family axis is a registry with a single entry for now.

Protocol
  rungs   --rungs "STEPS:N_EVAL,..." e.g. "60:500,180:1000,540:2000". STEPS is the
          cumulative number of optimiser updates a trained trial has received at the end
          of that rung. N_EVAL is the number of molecules sampled to score it. Promoted
          trials CONTINUE from their checkpoint; nothing is retrained.
  gate    after each rung keep the top ceil(n / eta) trials, ranked feasible-first, then
          by in-band fraction under the surrogate. Anchors are never eliminated.
  confirm the survivors are re-sampled with fresh seeds and a larger sample (winner's
          curse guard) and also scored by the independent holdout predictor.
  constraints (all must hold for a trial to be feasible)
          validity, uniqueness, novelty, rule pass rate above absolute floors, and
          internal diversity at least `div_ratio` times the plain anchor's at the same
          rung (catches mode collapse and surrogate reward hacking).

Everything is resumable: per-trial checkpoints (atomic writes) and a results file per
(trial, rung). Re-running the same command continues where it stopped.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import itertools
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uv import gen_model, steering  # noqa: E402
from uv.steer_eval import compute_metrics  # noqa: E402

FAMILIES = {"gru_selfies": "2-layer GRU over SELFIES (gen_model.SelfiesLM)"}
METHODS = ("guidance", "reinvent", "selftrain")


# ---------------------------------------------------------------------------
# Context: everything that is fixed across trials
# ---------------------------------------------------------------------------

@dataclass
class Limits:
    validity: float = 0.95
    uniqueness: float = 0.90
    novelty: float = 0.90
    rule_pass: float = 0.95
    div_ratio: float = 0.90
    holdout_gap: float = 0.10


class Context:
    def __init__(self, vocab, plain_path, steered_path, corpus_df, scorer, holdout,
                 band, device, limits: Limits, max_len=120, replay_n=20000):
        self.vocab = vocab
        self.device = device
        self.scorer = scorer
        self.holdout = holdout
        self.band = band
        self.limits = limits
        self.max_len = max_len
        self.corpus = corpus_df
        self.train_set = set(corpus_df["smiles"].tolist())
        self.paths = {"plain": plain_path, "steered": steered_path}
        self._frozen: Dict[str, torch.nn.Module] = {}
        self._replay: Dict[str, List[torch.Tensor]] = {}
        self.replay_n = replay_n

    def frozen(self, which: str):
        if which not in self._frozen:
            m, _ = steering.load_generator(self.paths[which], self.vocab, self.device)
            self._frozen[which] = steering.freeze(m)
        return self._frozen[which]

    def fresh(self, which: str):
        m, _ = steering.load_generator(self.paths[which], self.vocab, self.device)
        return m

    def prefix(self, which: str) -> List[int]:
        if which == "steered":
            return steering.prefix_ids(self.vocab, [gen_model.BAND_IN, gen_model.CLEAN])
        return []

    def replay(self, which: str) -> List[torch.Tensor]:
        """Corpus sequences with the same prefix/labelling the init model was trained on."""
        if which not in self._replay:
            df = self.corpus
            if len(df) > self.replay_n:
                df = df.sample(n=self.replay_n, random_state=0)
            seqs = []
            for row in df.itertuples(index=False):
                ids = gen_model.build_sequence(
                    self.vocab, row.selfies, which == "steered",
                    abs_pred=getattr(row, "abs_pred", None),
                    passes=getattr(row, "passes_screen", None),
                    band_lo=self.band[0], band_hi=self.band[1])
                if len(ids) >= 3:
                    seqs.append(torch.tensor(ids, dtype=torch.long))
            self._replay[which] = seqs
        return self._replay[which]


# ---------------------------------------------------------------------------
# Trial specs and sampling of the search space
# ---------------------------------------------------------------------------

def trial_id(spec: dict) -> str:
    blob = json.dumps(spec, sort_keys=True)
    h = hashlib.sha1(blob.encode()).hexdigest()[:6]
    return f"{spec['method']}-{h}"


def anchors() -> List[dict]:
    return [{"method": "plain", "family": "gru_selfies"},
            {"method": "prefix", "family": "gru_selfies"}]


def sample_space(methods: Sequence[str], n_per_method: Dict[str, int], seed: int) -> List[dict]:
    """Deterministic random sample of distinct configurations per method."""
    rng = np.random.default_rng(seed)
    grids = {
        "guidance": {"kind": ["bandout", "plain"], "w": [0.25, 0.5, 1.0, 2.0, 3.0, 5.0],
                     "temperature": [0.9, 1.0]},
        "reinvent": {"init": ["plain", "steered"], "sigma": [10.0, 30.0, 100.0],
                     "lr": [1e-4, 3e-4], "batch": [64, 128], "soft_width": [0.0, 25.0],
                     "topk_frac": [1.0, 0.5], "bucket": [0, 3]},
        "selftrain": {"init": ["plain", "steered"], "steps_per_round": [25, 50],
                      "gen_per_round": [1000, 2000], "replay_frac": [0.25, 0.5, 0.75],
                      "lr": [1e-4, 3e-4], "batch": [128]},
    }
    out = []
    for m in methods:
        grid = grids[m]
        keys = sorted(grid)
        combos = list(itertools.product(*[grid[k] for k in keys]))
        order = rng.permutation(len(combos))
        for i in order[:min(n_per_method.get(m, 0), len(combos))]:
            spec = {"method": m, "family": "gru_selfies"}
            spec.update({k: v for k, v in zip(keys, combos[i])})
            out.append(spec)
    return out


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------

def atomic_save(obj, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json(obj, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
    os.replace(tmp, path)


class Trial:
    """One configuration. `advance_to(steps)` trains; `sample(n, seed)` returns SELFIES."""

    def __init__(self, spec: dict, ctx: Context, root: str, base_seed: int):
        self.spec = spec
        self.ctx = ctx
        self.id = trial_id(spec)
        self.dir = os.path.join(root, "trials", self.id)
        self.seed = int(hashlib.sha1(f"{base_seed}-{self.id}".encode()).hexdigest()[:8], 16)
        self.steps = 0
        self.rounds = 0
        self.model = None
        self.opt = None
        self.pool: List[str] = []
        self.seen: Dict[str, int] = {}
        self.log: List[dict] = []
        self.train_seconds = 0.0
        self.trained = spec["method"] in ("reinvent", "selftrain")
        self._ckpt = os.path.join(self.dir, "state.pt")

    # -- persistence -------------------------------------------------------
    def _init_model(self):
        which = self.spec["init"]
        self.model = self.ctx.fresh(which)
        if self.spec["method"] == "reinvent":
            # The REINVENT likelihood ratio against the frozen prior should not carry
            # dropout noise; cuDNN also needs train mode for the backward pass.
            self.model.gru.dropout = 0.0
        lr = self.spec["lr"]
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

    def load(self):
        if not self.trained or self.model is not None:
            return
        if os.path.exists(self._ckpt):
            st = torch.load(self._ckpt, map_location=self.ctx.device, weights_only=False)
            self._init_model()
            self.model.load_state_dict(st["model"])
            self.opt.load_state_dict(st["opt"])
            self.steps = st["steps"]
            self.rounds = st.get("rounds", 0)
            self.pool = st.get("pool", [])
            self.seen = st.get("seen", {})
            self.log = st.get("log", [])
            self.train_seconds = st.get("train_seconds", 0.0)
        elif self.model is None:
            self._init_model()

    def save(self):
        if not self.trained:
            return
        atomic_save({"model": self.model.state_dict(), "opt": self.opt.state_dict(),
                     "steps": self.steps, "rounds": self.rounds, "pool": self.pool, "seen": self.seen,
                     "log": self.log[-200:], "train_seconds": self.train_seconds,
                     "spec": self.spec}, self._ckpt)

    # -- training ----------------------------------------------------------
    def advance_to(self, target: int, print_every: int = 10):
        if not self.trained or self.steps >= target:
            return
        ctx, spec = self.ctx, self.spec
        which = spec["init"]
        prefix = ctx.prefix(which)
        t0 = time.time()
        start_steps = self.steps
        if spec["method"] == "reinvent":
            cfg = steering.ReinventConfig(
                sigma=spec["sigma"], lr=spec["lr"], batch=spec["batch"],
                soft_width=spec["soft_width"], topk_frac=spec["topk_frac"],
                bucket=spec["bucket"], max_len=ctx.max_len)
            prior = ctx.frozen(which)
            while self.steps < target:
                stats = steering.reinvent_step(
                    self.model, prior, ctx.vocab, ctx.device, prefix, ctx.scorer,
                    ctx.band, cfg, self.opt, seed=self.seed * 1000003 + self.steps,
                    seen=self.seen)
                self.steps += 1
                self.log.append({"step": self.steps, **stats})
                if self.steps % print_every == 0 or self.steps == target:
                    dt = time.time() - t0
                    rate = (self.steps - start_steps) / max(dt, 1e-9)
                    print(f"    [{self.id}] step {self.steps}/{target} | "
                          f"{rate:.2f} steps/s | in-band batch {stats['in_band_batch']:.3f} | "
                          f"unique {stats['unique_batch']:.3f} | kl {stats['kl_nats']:.2f} | "
                          f"loss {stats['loss']:.2f}", flush=True)
        else:
            cfg = steering.SelfTrainConfig(
                steps_per_round=spec["steps_per_round"], gen_per_round=spec["gen_per_round"],
                replay_frac=spec["replay_frac"], lr=spec["lr"], batch=spec["batch"],
                max_len=ctx.max_len)
            replay = ctx.replay(which)
            while self.steps < target:
                # a round never overshoots the rung budget: the last round of a rung is
                # shortened, so equal budgets mean equal numbers of optimiser updates.
                n_steps = min(cfg.steps_per_round, target - self.steps)
                round_cfg = dataclasses.replace(cfg, steps_per_round=n_steps)
                stats = steering.selftrain_round(
                    self.model, self.opt, ctx.vocab, ctx.device, prefix, ctx.scorer,
                    ctx.band, round_cfg, replay, self.pool,
                    seed=self.seed * 1000003 + self.rounds)
                self.rounds += 1
                self.steps += n_steps
                self.log.append({"step": self.steps, **stats})
                dt = time.time() - t0
                rate = (self.steps - start_steps) / max(dt, 1e-9)
                print(f"    [{self.id}] round {self.rounds} step {self.steps}/{target} | "
                      f"{rate:.2f} steps/s | kept {stats['kept']}/{stats['unique_valid']} | "
                      f"pool {stats['pool']} | loss {stats['loss']:.3f}", flush=True)
        self.train_seconds += time.time() - t0
        self.save()

    # -- sampling ----------------------------------------------------------
    def sample(self, n: int, seed: int) -> List[str]:
        ctx, spec, m = self.ctx, self.spec, self.spec["method"]
        batch = 512
        out: List[str] = []
        for i, start in enumerate(range(0, n, batch)):
            b = min(batch, n - start)
            s = seed * 7919 + i
            if m == "plain":
                experts = [steering.Expert(ctx.frozen("plain"), [], 1.0)]
                temp, mask = 1.0, False
            elif m == "prefix":
                experts = [steering.Expert(ctx.frozen("steered"), ctx.prefix("steered"), 1.0)]
                temp, mask = 1.0, False
            elif m == "guidance":
                experts = steering.guidance_experts(
                    spec["kind"], spec["w"], ctx.frozen("steered"), ctx.frozen("plain"), ctx.vocab)
                temp, mask = spec["temperature"], True
            else:
                self.model.eval()
                experts = [steering.Expert(self.model, ctx.prefix(spec["init"]), 1.0)]
                temp, mask = 1.0, False
            tok, ln = steering.sample_tokens(experts, ctx.vocab, ctx.device, b, ctx.max_len,
                                             temp, s, mask_special=mask)
            out += steering.tokens_to_selfies(ctx.vocab, tok, ln)
        return out

    def evaluate(self, n: int, seed: int) -> dict:
        t0 = time.time()
        selfies_list = self.sample(n, seed)
        t_sample = time.time() - t0
        m = compute_metrics(selfies_list, self.ctx.train_set, self.ctx.scorer,
                            self.ctx.band[0], self.ctx.band[1],
                            holdout=self.ctx.holdout.predict if self.ctx.holdout else None,
                            seed=seed)
        m["eval_seconds"] = round(time.time() - t0, 2)
        m["sample_seconds"] = round(t_sample, 2)
        if self.trained and self.log:
            tail = [r for r in self.log[-20:] if "kl_nats" in r]
            if tail:
                m["kl_nats_recent"] = round(float(np.mean([r["kl_nats"] for r in tail])), 3)
        return m


# ---------------------------------------------------------------------------
# Ranking and gating
# ---------------------------------------------------------------------------

def violations(m: dict, lim: Limits, plain_div: Optional[float]) -> Dict[str, float]:
    """Shortfall per constraint (empty dict means feasible)."""
    v = {}
    for key, floor in (("validity", lim.validity), ("uniqueness", lim.uniqueness),
                       ("novelty", lim.novelty), ("rule_pass_rate", lim.rule_pass)):
        if m.get(key, 0.0) < floor:
            v[key] = round(floor - m.get(key, 0.0), 5)
    if plain_div is not None and not math.isnan(plain_div):
        floor = lim.div_ratio * plain_div
        d = m.get("int_div", float("nan"))
        if math.isnan(d) or d < floor:
            v["int_div"] = round(floor - (0.0 if math.isnan(d) else d), 5)
    return v


def rank_key(m: dict, viol: Dict[str, float]):
    """Sort key (ascending is better): feasible first, then in-band descending."""
    return (0 if not viol else 1, sum(viol.values()), -m.get("in_band_fraction", 0.0))


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 3:
        return None
    from scipy.stats import spearmanr
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    r = spearmanr(a, b)[0]
    return None if r is None or np.isnan(r) else float(r)


def parse_rungs(text: str) -> List[Tuple[int, int]]:
    out = []
    for part in text.split(","):
        steps, n = part.split(":")
        out.append((int(steps), int(n)))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class Search:
    def __init__(self, ctx: Context, root: str, args):
        self.ctx = ctx
        self.root = root
        self.args = args
        self.rungs = parse_rungs(args.rungs)
        self.results_path = os.path.join(root, "rung_results.json")
        self.rung_results: Dict[str, dict] = {}   # "rung|trial" -> metrics
        if os.path.exists(self.results_path):
            with open(self.results_path) as fh:
                self.rung_results = json.load(fh)
            print(f"resuming: {len(self.rung_results)} (rung, trial) results on disk", flush=True)

    def _key(self, stage: str, tid: str) -> str:
        return f"{stage}|{tid}"

    def _run(self, trial: Trial, stage: str, steps: int, n_eval: int, eval_seed: int):
        key = self._key(stage, trial.id)
        if key in self.rung_results:
            return self.rung_results[key]
        trial.load()
        trial.advance_to(steps)
        m = trial.evaluate(n_eval, eval_seed)
        m.update({"trial": trial.id, "stage": stage, "steps": trial.steps, "n_eval": n_eval,
                  "train_seconds": round(trial.train_seconds, 2)})
        self.rung_results[key] = m
        atomic_json(self.rung_results, self.results_path)
        return m

    SCHEDULE_KEYS = ("rungs", "band_lo", "band_hi", "seed", "methods", "n_guidance",
                     "n_reinvent", "n_selftrain", "eta", "n_confirm", "confirm_n_eval")

    def _check_compatible(self, a, specs):
        """Refuse to resume into results that were produced by a different schedule."""
        path = os.path.join(self.root, "config.json")
        if not os.path.exists(path):
            return
        with open(path) as fh:
            old = json.load(fh).get("args", {})
        diff = {k: (old.get(k), getattr(a, k)) for k in self.SCHEDULE_KEYS
                if old.get(k) != getattr(a, k)}
        if diff:
            raise SystemExit(f"--root {self.root} holds a run with a different schedule "
                             f"{diff}; use a new --root or restore the old arguments")

    def run(self):
        a, ctx = self.args, self.ctx
        specs = anchors() + sample_space(
            a.methods.split(","),
            {"guidance": a.n_guidance, "reinvent": a.n_reinvent, "selftrain": a.n_selftrain},
            a.seed)
        trials = {trial_id(s): Trial(s, ctx, self.root, a.seed) for s in specs}
        self._check_compatible(a, specs)
        atomic_json({"args": vars(a), "specs": {t: tr.spec for t, tr in trials.items()},
                     "band": list(ctx.band), "limits": asdict(ctx.limits)},
                    os.path.join(self.root, "config.json"))
        anchor_ids = [trial_id(s) for s in anchors()]
        alive = [t for t in trials if t not in anchor_ids]
        history = []   # per rung: ids evaluated, kept
        t_start = time.time()
        print(f"meta-search: {len(alive)} candidate trials + {len(anchor_ids)} anchors, "
              f"rungs {self.rungs}, eta {a.eta}", flush=True)

        for r, (steps, n_eval) in enumerate(self.rungs):
            stage = f"rung{r}"
            print(f"\n=== {stage}: steps {steps}, n_eval {n_eval}, "
                  f"{len(alive)} trials (+anchors) ===", flush=True)
            evald = anchor_ids + alive
            for i, tid in enumerate(evald, 1):
                tr = trials[tid]
                m = self._run(tr, stage, steps, n_eval, eval_seed=a.seed * 100 + r)
                print(f"  [{stage} {i}/{len(evald)}] {tid:<18} in-band {m['in_band_fraction']:.3f} "
                      f"valid {m['validity']:.3f} uniq {m['uniqueness']:.3f} "
                      f"div {m['int_div']:.3f} rule {m['rule_pass_rate']:.3f} "
                      f"(elapsed {(time.time() - t_start) / 60:.1f} min)", flush=True)
            plain_div = self.rung_results[self._key(stage, anchor_ids[0])]["int_div"]
            scored = []
            for tid in alive:
                m = self.rung_results[self._key(stage, tid)]
                viol = violations(m, ctx.limits, plain_div)
                scored.append((rank_key(m, viol), tid, viol))
            scored.sort()
            n_keep = max(1, math.ceil(len(alive) / a.eta)) if r < len(self.rungs) - 1 else \
                min(a.n_confirm, len(alive))
            kept = [t for _, t, _ in scored[:n_keep]]
            history.append({"rung": r, "evaluated": alive, "kept": kept,
                            "feasible": [t for _, t, v in scored if not v]})
            print(f"  gate: kept {len(kept)}/{len(alive)} "
                  f"({sum(1 for _, _, v in scored if not v)} feasible): {', '.join(kept)}", flush=True)
            alive = kept
            atomic_json(history, os.path.join(self.root, "history.json"))

        # confirm stage: fresh samples, bigger n, anchors included
        print(f"\n=== confirm: {len(alive)} survivors, n_eval {a.confirm_n_eval}, fresh seed ===",
              flush=True)
        steps_final = self.rungs[-1][0]
        for tid in anchor_ids + alive:
            m = self._run(trials[tid], "confirm", steps_final, a.confirm_n_eval,
                          eval_seed=a.seed * 100 + 99)
            print(f"  [confirm] {tid:<18} in-band {m['in_band_fraction']:.3f} "
                  f"holdout {m.get('in_band_holdout', float('nan')):.3f} "
                  f"div {m['int_div']:.3f}", flush=True)
        write_reports(self, trials, anchor_ids, history, alive)
        print(f"\ndone in {(time.time() - t_start) / 60:.1f} min. results: "
              f"{os.path.join(self.root, 'results.md')}", flush=True)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _hp_str(spec: dict) -> str:
    skip = {"method", "family"}
    return " ".join(f"{k}={spec[k]}" for k in sorted(spec) if k not in skip)


def write_reports(search: Search, trials: Dict[str, Trial], anchor_ids, history, finalists):
    ctx, rr = search.ctx, search.rung_results
    lim = ctx.limits
    n_rungs = len(search.rungs)
    plain_c = rr[search._key("confirm", anchor_ids[0])]
    rows = []
    for tid in anchor_ids + finalists:
        m = rr[search._key("confirm", tid)]
        viol = violations(m, lim, plain_c["int_div"])
        gap = None
        if "in_band_holdout" in m:
            gap = round(m["in_band_fraction"] - m["in_band_holdout"], 5)
        rows.append({"trial": tid, "method": trials[tid].spec["method"],
                     "hparams": _hp_str(trials[tid].spec), "feasible": not viol,
                     "violations": viol, "holdout_gap": gap,
                     "holdout_ok": gap is not None and gap <= lim.holdout_gap,
                     **m})
    # rank correlation of rung0 vs last rung among trials that reached the last rung
    corr = None
    last = f"rung{n_rungs - 1}"
    both = [t for t in history[-1]["evaluated"] if search._key("rung0", t) in rr
            and search._key(last, t) in rr]
    if n_rungs > 1 and len(both) >= 3:
        corr = spearman([rr[search._key("rung0", t)]["in_band_fraction"] for t in both],
                        [rr[search._key(last, t)]["in_band_fraction"] for t in both])
    per_rung = {}
    for r in range(n_rungs):
        stage = f"rung{r}"
        per_rung[stage] = [rr[k] for k in rr if k.startswith(stage + "|")]
    winners = [x for x in rows if x["method"] not in ("plain", "prefix")
               and x["feasible"] and x["holdout_ok"]]
    winners.sort(key=lambda x: -x["in_band_fraction"])
    doc = {
        "band_nm": list(ctx.band),
        "limits": asdict(lim),
        "rungs": search.rungs,
        "holdout_predictor": getattr(ctx.holdout, "metrics", None),
        "confirm": rows,
        "winner": winners[0]["trial"] if winners else None,
        "pilot_to_final_rank_correlation_spearman": corr,
        "n_trials_per_rung": {k: len(v) for k, v in per_rung.items()},
        "gate_history": history,
        "all_rung_results": rr,
    }
    atomic_json(doc, os.path.join(search.root, "results.json"))

    def f(x, nd=3):
        return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"

    md = ["# Steering meta-search results", "",
          f"Target band {ctx.band[0]:.0f} to {ctx.band[1]:.0f} nm (surrogate = fixed seed-0 UV "
          f"predictor). Rungs (steps:n_eval): {search.rungs}.",
          f"Constraints: validity >= {lim.validity}, uniqueness >= {lim.uniqueness}, novelty >= "
          f"{lim.novelty}, rule pass >= {lim.rule_pass}, internal diversity >= {lim.div_ratio} x "
          f"plain anchor; holdout gap (surrogate minus holdout in-band) <= {lim.holdout_gap}.", ""]
    if doc["holdout_predictor"]:
        hm = doc["holdout_predictor"]
        md += [f"Holdout predictor: {hm.get('model')}, test MAE {f(hm.get('mae_nm'), 1)} nm, "
               f"R2 {f(hm.get('r2'))} (own split; same database as the surrogate).", ""]
    md += ["## Confirmation stage (fresh samples)", "",
           "| trial | method | hparams | in-band (surr) | in-band (holdout) | gap | valid | "
           "uniq | novel | rule | int_div | kl nats | feasible | holdout ok |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        md.append(f"| {x['trial']} | {x['method']} | {x['hparams']} | {f(x['in_band_fraction'])} | "
                  f"{f(x.get('in_band_holdout'))} | {f(x['holdout_gap'])} | {f(x['validity'])} | "
                  f"{f(x['uniqueness'])} | {f(x['novelty'])} | {f(x['rule_pass_rate'])} | "
                  f"{f(x['int_div'])} | {f(x.get('kl_nats_recent'), 2)} | "
                  f"{'yes' if x['feasible'] else 'no ' + ','.join(x['violations'])} | "
                  f"{'yes' if x['holdout_ok'] else 'no'} |")
    md += ["", f"Winner (best feasible, holdout-ok, by surrogate in-band): "
           f"{doc['winner'] or 'none'}", ""]
    if corr is not None:
        md += [f"Pilot-to-final rank correlation (Spearman, rung0 vs {last}, "
               f"n={len(both)}): {corr:.3f}", ""]
    for stage, items in per_rung.items():
        md += [f"## {stage}", "",
               "| trial | method | steps | n_eval | in-band | valid | uniq | novel | rule | int_div |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for m in sorted(items, key=lambda z: -z["in_band_fraction"]):
            md.append(f"| {m['trial']} | {trials[m['trial']].spec['method']} | {m['steps']} | "
                      f"{m['n_eval']} | {f(m['in_band_fraction'])} | {f(m['validity'])} | "
                      f"{f(m['uniqueness'])} | {f(m['novelty'])} | {f(m['rule_pass_rate'])} | "
                      f"{f(m['int_div'])} |")
        md.append("")
    md += ["## Trial hyper-parameters", "", "| trial | hparams |", "|---|---|"]
    for tid, tr in trials.items():
        md.append(f"| {tid} | {tr.spec['method']} {_hp_str(tr.spec)} |")
    with open(os.path.join(search.root, "results.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_context(args) -> Context:
    import joblib
    from uv import holdout_predictor
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    vocab = gen_model.Vocab.load(args.vocab)
    df = pd.read_csv(args.corpus)
    scorer = steering.UVScorer.from_bundle(joblib.load(args.predictor))
    holdout = None
    if args.uvvis_data:
        holdout = holdout_predictor.load_or_train(
            os.path.join(args.root, "holdout_predictor.joblib"), args.uvvis_data,
            seed=args.holdout_seed, n_estimators=args.holdout_trees)
        print(f"holdout predictor: {holdout.metrics}", flush=True)
    return Context(vocab, args.plain, args.steered, df, scorer, holdout,
                   (args.band_lo, args.band_hi), device, Limits(
                       validity=args.min_validity, uniqueness=args.min_uniqueness,
                       novelty=args.min_novelty, rule_pass=args.min_rule_pass,
                       div_ratio=args.div_ratio, holdout_gap=args.max_holdout_gap),
                   replay_n=args.replay_n)


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="run directory (checkpoints + results)")
    ap.add_argument("--plain", required=True, help="plain generator checkpoint (.pt)")
    ap.add_argument("--steered", required=True, help="steered generator checkpoint (.pt)")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--corpus", required=True, help="corpus.csv with selfies, smiles, abs_pred")
    ap.add_argument("--predictor", required=True, help="fixed seed-0 surrogate joblib")
    ap.add_argument("--uvvis-data", default=None,
                    help="uvvis_dataset.csv; trains/loads the independent holdout predictor")
    ap.add_argument("--band-lo", type=float, default=400.0)
    ap.add_argument("--band-hi", type=float, default=450.0)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--n-guidance", type=int, default=12)
    ap.add_argument("--n-reinvent", type=int, default=12)
    ap.add_argument("--n-selftrain", type=int, default=8)
    ap.add_argument("--rungs", default="60:500,180:1000,540:2000")
    ap.add_argument("--eta", type=float, default=3.0)
    ap.add_argument("--n-confirm", type=int, default=3)
    ap.add_argument("--confirm-n-eval", type=int, default=5000)
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
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is present")
    return ap


def main(argv=None):
    args = make_parser().parse_args(argv)
    os.makedirs(args.root, exist_ok=True)
    torch.manual_seed(args.seed)
    ctx = build_context(args)
    print(f"device={ctx.device} band={ctx.band} corpus={len(ctx.corpus)}", flush=True)
    Search(ctx, args.root, args).run()


if __name__ == "__main__":
    main()
