"""SELFIES language model shared by the generator training and evaluation scripts.

Small GRU autoregressive model over SELFIES tokens. Two arms use the same class and
the same hyper-parameters. The steered arm prepends two control tokens to every
sequence (a band token and a structural-screen token); the plain arm does not.
"""
from __future__ import annotations

import json

import torch
import torch.nn as nn
import selfies as sf

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
BAND_IN, BAND_OUT = "<BAND_IN>", "<BAND_OUT>"
CLEAN, FLAG = "<CLEAN>", "<FLAG>"
CONTROL = [BAND_IN, BAND_OUT, CLEAN, FLAG]


def tokenize(selfies_str):
    return list(sf.split_selfies(selfies_str))


class Vocab:
    def __init__(self, tokens):
        base = [PAD, BOS, EOS] + CONTROL
        syms = sorted(set(tokens) - set(base))
        self.itos = base + syms
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @property
    def pad_id(self):
        return self.stoi[PAD]

    def encode(self, toks):
        return [self.stoi[t] for t in toks if t in self.stoi]

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.itos, fh)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            itos = json.load(fh)
        obj = cls.__new__(cls)
        obj.itos = itos
        obj.stoi = {t: i for i, t in enumerate(itos)}
        return obj


class SelfiesLM(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=256, layers=2, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.gru = nn.GRU(emb, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x, h=None):
        e = self.emb(x)
        y, h = self.gru(e, h)
        return self.out(y), h


def build_sequence(vocab, selfies_str, steered, abs_pred=None, passes=None,
                   band_lo=None, band_hi=None):
    """Token id sequence for one molecule, including <bos> and <eos>.

    Steered sequences carry a band token and a screen token right after <bos>.
    """
    toks = [BOS]
    if steered:
        in_band = (band_lo <= abs_pred <= band_hi)
        toks.append(BAND_IN if in_band else BAND_OUT)
        toks.append(CLEAN if passes else FLAG)
    toks += tokenize(selfies_str)
    toks.append(EOS)
    return vocab.encode(toks)


@torch.no_grad()
def sample(model, vocab, device, n, steered, max_len=120, temperature=1.0,
           prefix_tokens=None, seed=0):
    """Sample n sequences. Returns a list of SELFIES strings (may be empty on failure)."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    results = []
    batch = 256
    done = 0
    eos_id = vocab.stoi[EOS]
    while done < n:
        b = min(batch, n - done)
        start = [vocab.stoi[BOS]]
        if steered:
            start += [vocab.stoi[t] for t in (prefix_tokens or [BAND_IN, CLEAN])]
        seq = torch.tensor([start] * b, dtype=torch.long, device=device)
        h = None
        inp = seq
        finished = [False] * b
        toks_out = [list(start) for _ in range(b)]
        for _ in range(max_len):
            logits, h = model(inp, h)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(logits, dim=-1).cpu()
            nxt = torch.multinomial(probs, 1, generator=g).to(device)
            inp = nxt
            for i in range(b):
                if not finished[i]:
                    tid = int(nxt[i, 0])
                    if tid == eos_id:
                        finished[i] = True
                    else:
                        toks_out[i].append(tid)
            if all(finished):
                break
        for i in range(b):
            ids = toks_out[i]
            syms = [vocab.itos[t] for t in ids
                    if vocab.itos[t] not in (BOS, EOS, PAD, BAND_IN, BAND_OUT, CLEAN, FLAG)]
            results.append("".join(syms))
        done += b
    return results
