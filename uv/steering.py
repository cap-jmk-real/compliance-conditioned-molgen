"""Stronger steering options for the UV-Vis SELFIES generator.

The existing pipeline steers by prefix tokens only (gen_model.py): the sampler is
primed with <BAND_IN> <CLEAN> and the language model is left to figure out what that
means. This module adds three ways to push harder, all plugging into the same
SelfiesLM checkpoints, the same vocabulary, and the same fixed seed-0 UV surrogate:

  guidance   sampling-time logit guidance. No training. The next-token log-probs of
             several (model, prefix) experts are combined linearly, which gives
             classifier-free-guidance style extrapolation with a strength knob w.
  reinvent   REINVENT-style policy fine-tuning (Olivecrona et al. 2017). A trainable
             agent starts as a copy of a frozen prior; the loss pulls the agent's
             sequence log-likelihood toward prior log-likelihood plus sigma times the
             score. The prior term is the KL/prior penalty.
  selftrain  surrogate-filtered self-training / expert iteration: sample, keep the
             molecules the surrogate puts in the band, fine-tune on them mixed with
             corpus replay, repeat.

Safety scope. The reward is the UV objective only (a function of the predicted
absorption maximum in nm). The structural screen from smarts_rules.py is never part of
any reward. It is used as a hard FILTER on self-training data and as an evaluation
constraint, the same role it has in the existing pipeline. Nothing here touches
synthesis, reactions or routes.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import selfies as sf

from uv import gen_model
from uv.common import canonical_smiles, morgan_fingerprint, passes_screen

SPECIAL_TOKENS = (gen_model.PAD, gen_model.BOS, gen_model.EOS, *gen_model.CONTROL)


# ---------------------------------------------------------------------------
# Models and decoding helpers
# ---------------------------------------------------------------------------

def load_generator(path: str, vocab, device) -> Tuple[gen_model.SelfiesLM, dict]:
    """Load a generator checkpoint written by train_generator.py."""
    bundle = torch.load(path, map_location=device, weights_only=False)
    model = gen_model.SelfiesLM(len(vocab), bundle["emb"], bundle["hidden"],
                                bundle["layers"]).to(device)
    model.load_state_dict(bundle["model"])
    model.eval()
    return model, bundle


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def prefix_ids(vocab, tokens: Optional[Sequence[str]]) -> List[int]:
    return [vocab.stoi[t] for t in (tokens or [])]


def decode_selfies(vocab, ids: Sequence[int]) -> str:
    """Token ids (generated part only) to a SELFIES string, dropping special tokens."""
    out = []
    for t in ids:
        tok = vocab.itos[int(t)]
        if tok == gen_model.EOS:
            break
        if tok in SPECIAL_TOKENS:
            continue
        out.append(tok)
    return "".join(out)


def selfies_to_smiles(selfies_str: str) -> Optional[str]:
    """Canonical SMILES for a SELFIES string, or None if empty or undecodable."""
    if not selfies_str:
        return None
    try:
        smi = sf.decoder(selfies_str)
    except Exception:
        return None
    return canonical_smiles(smi)


# ---------------------------------------------------------------------------
# UV reward (surrogate only)
# ---------------------------------------------------------------------------

def band_reward(pred_nm: np.ndarray, lo: float, hi: float, soft_width: float = 0.0):
    """Reward in [0, 1] from predicted absorption maxima.

    1 inside [lo, hi]. Outside: 0 when soft_width is 0 (hard indicator), otherwise
    exp(-(d / soft_width)^2) with d the distance in nm to the nearest band edge.
    NaN predictions (unparsable molecule) score 0.
    """
    pred = np.asarray(pred_nm, dtype=float)
    d = np.maximum(np.maximum(lo - pred, pred - hi), 0.0)
    if soft_width and soft_width > 0:
        r = np.exp(-(d / soft_width) ** 2)
    else:
        r = (d == 0).astype(float)
    r = np.where(np.isnan(pred), 0.0, r)
    return r


class UVScorer:
    """Wraps a fingerprint regressor bundle {'model', 'n_bits'} (the seed-0 surrogate)."""

    def __init__(self, model, n_bits: int):
        self.model = model
        self.n_bits = n_bits

    @classmethod
    def from_bundle(cls, bundle):
        return cls(bundle["model"], bundle["n_bits"])

    def predict(self, smiles: Sequence[Optional[str]]) -> np.ndarray:
        out = np.full(len(smiles), np.nan, dtype=float)
        feats, idx = [], []
        for i, s in enumerate(smiles):
            if s is None:
                continue
            fp = morgan_fingerprint(s, n_bits=self.n_bits)
            if fp is not None:
                feats.append(fp)
                idx.append(i)
        if feats:
            out[idx] = self.model.predict(np.vstack(feats))
        return out


# ---------------------------------------------------------------------------
# Sampling, with optional logit guidance
# ---------------------------------------------------------------------------

@dataclass
class Expert:
    """One term of a guided sampler: coef * log_softmax(model(prefix))."""
    model: gen_model.SelfiesLM
    prefix: List[int]
    coef: float = 1.0


def guidance_experts(kind: str, w: float, steered, plain, vocab) -> List[Expert]:
    """Expert list for a guidance recipe.

    kind='bandout'  logp = (1+w) logp(.|IN,CLEAN) - w logp(.|OUT,CLEAN), both from the
                    steered model. Contrasts the band token only, so the screen token
                    is the same in both terms.
    kind='plain'    logp = (1+w) logp_steered(.|IN,CLEAN) - w logp_plain(.|). The
                    extrapolation from the plain model through the steered model, in
                    the spirit of proxy-tuning (Liu et al. 2024).
    w = 0 recovers ordinary prefix steering.
    """
    s_in = prefix_ids(vocab, [gen_model.BAND_IN, gen_model.CLEAN])
    if kind == "bandout":
        s_out = prefix_ids(vocab, [gen_model.BAND_OUT, gen_model.CLEAN])
        return [Expert(steered, s_in, 1.0 + w), Expert(steered, s_out, -w)]
    if kind == "plain":
        return [Expert(steered, s_in, 1.0 + w), Expert(plain, [], -w)]
    raise ValueError(f"unknown guidance kind {kind!r}")


@torch.no_grad()
def sample_tokens(experts: Sequence[Expert], vocab, device, n: int, max_len: int = 120,
                  temperature: float = 1.0, seed: int = 0, mask_special: bool = False
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample n sequences from the (guided) combination of experts.

    Returns (tokens [n, max_len] padded with PAD after EOS, lengths [n]) where lengths
    count the generated tokens including EOS (or max_len if it never ended). The prefix
    is not part of the output. With a single expert of coef 1 this is the model's own
    distribution at the given temperature (mask_special=False).
    """
    g = torch.Generator(device=device).manual_seed(int(seed) & 0x7FFFFFFF)
    bos = vocab.stoi[gen_model.BOS]
    eos = vocab.stoi[gen_model.EOS]
    pad = vocab.pad_id
    hidden = []
    inputs = []
    for ex in experts:
        ex.model.eval()
        hidden.append(None)
        inputs.append(torch.tensor([[bos] + ex.prefix] * n, dtype=torch.long, device=device))
    block = None
    if mask_special:
        block = torch.zeros(len(vocab), dtype=torch.bool, device=device)
        for t in SPECIAL_TOKENS:
            if t != gen_model.EOS:
                block[vocab.stoi[t]] = True
    out = torch.full((n, max_len), pad, dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    lengths = torch.full((n,), max_len, dtype=torch.long, device=device)
    for t in range(max_len):
        logp = 0.0
        for k, ex in enumerate(experts):
            logits, hidden[k] = ex.model(inputs[k], hidden[k])
            lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
            logp = logp + ex.coef * lp
        logits = logp / max(temperature, 1e-6)
        if block is not None:
            logits = logits.masked_fill(block, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        nxt = torch.multinomial(probs, 1, generator=g)
        nxt = torch.where(done.unsqueeze(1), torch.full_like(nxt, pad), nxt)
        out[:, t] = nxt[:, 0]
        newly = (~done) & (nxt[:, 0] == eos)
        lengths = torch.where(newly, torch.full_like(lengths, t + 1), lengths)
        done = done | newly
        inputs = [nxt for _ in experts]
        if bool(done.all()):
            break
    return out, lengths


def tokens_to_selfies(vocab, tokens: torch.Tensor, lengths: torch.Tensor) -> List[str]:
    toks = tokens.cpu().tolist()
    lens = lengths.cpu().tolist()
    return [decode_selfies(vocab, row[:ln]) for row, ln in zip(toks, lens)]


def sequence_logprob(model, vocab, prefix: List[int], tokens: torch.Tensor,
                     lengths: torch.Tensor) -> torch.Tensor:
    """Sum of log p(token) over the generated tokens (EOS included), under `model`
    conditioned on [BOS] + prefix. Differentiable w.r.t. model parameters."""
    n, T = tokens.shape
    bos = vocab.stoi[gen_model.BOS]
    head = torch.tensor([[bos] + prefix] * n, dtype=torch.long, device=tokens.device)
    full = torch.cat([head, tokens], dim=1)
    inp = full[:, :-1]
    logits, _ = model(inp)
    P = len(prefix)
    logits = logits[:, P:, :]                      # aligned with tokens
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = logp.gather(2, tokens.unsqueeze(2)).squeeze(2)
    mask = (torch.arange(T, device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1))
    return (tok_lp * mask).sum(dim=1)


# ---------------------------------------------------------------------------
# REINVENT-style update
# ---------------------------------------------------------------------------

@dataclass
class ReinventConfig:
    sigma: float = 30.0
    lr: float = 1e-4
    batch: int = 64
    soft_width: float = 0.0       # nm; 0 = hard in-band indicator
    topk_frac: float = 1.0        # <1: augmented hill-climb, learn from top fraction only
    bucket: int = 0               # >0: a molecule seen this many times gets reward 0
    max_len: int = 120
    grad_clip: float = 1.0


def reinvent_step(agent, prior, vocab, device, prefix: List[int], scorer: UVScorer,
                  band: Tuple[float, float], cfg: ReinventConfig, opt, seed: int,
                  seen: Optional[Dict[str, int]] = None) -> dict:
    """One REINVENT update. Returns a dict of batch statistics (floats)."""
    agent.eval()   # sampling without dropout noise; the loss pass below is in train mode
    tokens, lengths = sample_tokens([Expert(agent, prefix, 1.0)], vocab, device,
                                    cfg.batch, cfg.max_len, 1.0, seed)
    selfies_list = tokens_to_selfies(vocab, tokens, lengths)
    smiles = [selfies_to_smiles(s) for s in selfies_list]
    pred = scorer.predict(smiles)
    reward = band_reward(pred, band[0], band[1], cfg.soft_width)
    # Within-batch duplicates earn nothing beyond the first copy; across history a
    # molecule seen `bucket` times is no longer rewarded (identical-molecule filter).
    first = set()
    for i, s in enumerate(smiles):
        if s is None:
            reward[i] = 0.0
            continue
        if s in first:
            reward[i] = 0.0
        first.add(s)
        if seen is not None and cfg.bucket > 0:
            if seen.get(s, 0) >= cfg.bucket:
                reward[i] = 0.0
            seen[s] = seen.get(s, 0) + 1
    score = torch.tensor(2.0 * reward - 1.0, dtype=torch.float32, device=device)

    agent.train()
    with torch.no_grad():
        prior_lp = sequence_logprob(prior, vocab, prefix, tokens, lengths)
    agent_lp = sequence_logprob(agent, vocab, prefix, tokens, lengths)
    aug = prior_lp + cfg.sigma * score
    per = (aug - agent_lp) ** 2
    if cfg.topk_frac < 1.0:
        k = max(1, int(math.ceil(cfg.topk_frac * len(per))))
        order = torch.argsort(score + 1e-3 * (aug - agent_lp).detach(), descending=True)[:k]
        per = per[order]
    loss = per.mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_clip)
    opt.step()
    valid = [s for s in smiles if s is not None]
    return {
        "loss": float(loss.detach()),
        "reward_mean": float(np.mean(reward)),
        "in_band_batch": float(np.mean((reward >= 1.0).astype(float))),
        "valid_batch": len(valid) / max(len(smiles), 1),
        "unique_batch": len(set(valid)) / max(len(valid), 1),
        "kl_nats": float((agent_lp.detach() - prior_lp).mean()),
    }


# ---------------------------------------------------------------------------
# Surrogate-filtered self-training
# ---------------------------------------------------------------------------

@dataclass
class SelfTrainConfig:
    steps_per_round: int = 25
    gen_per_round: int = 1000
    replay_frac: float = 0.5
    lr: float = 1e-4
    batch: int = 128
    pool_cap: int = 5000
    screen_filter: bool = True     # hard filter on training data, never a reward
    max_len: int = 120


def build_ids(vocab, selfies_str: str, prefix: List[int]) -> torch.Tensor:
    toks = gen_model.tokenize(selfies_str)
    ids = [vocab.stoi[gen_model.BOS]] + prefix + vocab.encode(toks) + [vocab.stoi[gen_model.EOS]]
    return torch.tensor(ids, dtype=torch.long)


def selftrain_collect(model, vocab, device, prefix: List[int], scorer: UVScorer,
                      band: Tuple[float, float], cfg: SelfTrainConfig, seed: int
                      ) -> Tuple[List[str], dict]:
    """Sample, then keep unique valid in-band molecules (and screen-clean ones)."""
    tokens, lengths = sample_tokens([Expert(model, prefix, 1.0)], vocab, device,
                                    cfg.gen_per_round, cfg.max_len, 1.0, seed)
    selfies_list = tokens_to_selfies(vocab, tokens, lengths)
    smiles = [selfies_to_smiles(s) for s in selfies_list]
    uniq = {}
    for s, smi in zip(selfies_list, smiles):
        if smi is not None and smi not in uniq:
            uniq[smi] = s
    smi_list = list(uniq)
    pred = scorer.predict(smi_list)
    keep = []
    for smi, p in zip(smi_list, pred):
        if np.isnan(p) or not (band[0] <= p <= band[1]):
            continue
        if cfg.screen_filter and not passes_screen(smi):
            continue
        keep.append(uniq[smi])
    stats = {"sampled": len(selfies_list), "unique_valid": len(smi_list),
             "kept": len(keep), "kept_frac": len(keep) / max(len(smi_list), 1)}
    return keep, stats


def pad_batch(seqs: List[torch.Tensor], pad_id: int, device):
    return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True,
                                           padding_value=pad_id).to(device)


def lm_step(model, opt, batch: torch.Tensor, pad_id: int, grad_clip: float = 1.0) -> float:
    model.train()
    inp, tgt = batch[:, :-1], batch[:, 1:]
    logits, _ = model(inp)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                           ignore_index=pad_id)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    return float(loss.detach())


def selftrain_round(model, opt, vocab, device, prefix: List[int], scorer: UVScorer,
                    band: Tuple[float, float], cfg: SelfTrainConfig, replay_seqs: List[torch.Tensor],
                    pool: List[str], seed: int) -> dict:
    """One round: collect, grow the pool, then steps_per_round LM steps."""
    model.eval()
    keep, stats = selftrain_collect(model, vocab, device, prefix, scorer, band, cfg, seed)
    known = set(pool)
    for s in keep:
        if s not in known:
            pool.append(s)
            known.add(s)
    if len(pool) > cfg.pool_cap:
        del pool[:len(pool) - cfg.pool_cap]
    rng = np.random.default_rng(seed + 7)
    losses = []
    if pool:
        pool_seqs_cache = {}
        for _ in range(cfg.steps_per_round):
            n_rep = int(round(cfg.replay_frac * cfg.batch))
            n_new = cfg.batch - n_rep
            seqs = []
            if n_rep and replay_seqs:
                seqs += [replay_seqs[i] for i in rng.integers(0, len(replay_seqs), n_rep)]
            for i in rng.integers(0, len(pool), n_new):
                s = pool[i]
                if s not in pool_seqs_cache:
                    pool_seqs_cache[s] = build_ids(vocab, s, prefix)
                seqs.append(pool_seqs_cache[s])
            losses.append(lm_step(model, opt, pad_batch(seqs, vocab.pad_id, device),
                                  vocab.pad_id))
    stats["pool"] = len(pool)
    stats["loss"] = float(np.mean(losses)) if losses else float("nan")
    return stats
