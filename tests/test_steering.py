import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steer_fixtures  # noqa: E402

from uv import gen_model, steering  # noqa: E402
from uv.steer_eval import compute_metrics, internal_diversity  # noqa: E402
from uv import holdout_predictor  # noqa: E402


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return steer_fixtures.build(str(tmp_path_factory.mktemp("steer")))


@pytest.fixture(scope="module")
def parts(fx):
    import joblib
    vocab = gen_model.Vocab.load(fx["vocab"])
    plain, _ = steering.load_generator(fx["plain"], vocab, "cpu")
    steered, _ = steering.load_generator(fx["steered"], vocab, "cpu")
    scorer = steering.UVScorer.from_bundle(joblib.load(fx["predictor"]))
    return vocab, plain, steered, scorer


def test_band_reward_hard_and_soft():
    pred = np.array([390.0, 400.0, 425.0, 450.0, 470.0, np.nan])
    hard = steering.band_reward(pred, 400, 450, 0.0)
    assert hard.tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    soft = steering.band_reward(pred, 400, 450, 25.0)
    assert soft[1] == soft[2] == soft[3] == 1.0
    assert 0.0 < soft[0] < 1.0 and 0.0 < soft[4] < 1.0
    assert soft[0] == pytest.approx(np.exp(-(10 / 25) ** 2))
    assert soft[5] == 0.0


def test_reward_depends_on_prediction_only():
    # The reward function has no access to molecules, only to nm values, so the
    # structural screen cannot leak into it.
    import inspect
    params = list(inspect.signature(steering.band_reward).parameters)
    assert params == ["pred_nm", "lo", "hi", "soft_width"]


def test_sample_tokens_shapes_and_determinism(parts):
    vocab, plain, steered, _ = parts
    ex = [steering.Expert(steered, steering.prefix_ids(vocab, [gen_model.BAND_IN, gen_model.CLEAN]))]
    t1, l1 = steering.sample_tokens(ex, vocab, "cpu", 16, 40, 1.0, seed=3)
    t2, l2 = steering.sample_tokens(ex, vocab, "cpu", 16, 40, 1.0, seed=3)
    assert t1.shape == (16, 40) and torch.equal(t1, t2) and torch.equal(l1, l2)
    assert int(l1.max()) <= 40 and int(l1.min()) >= 1
    t3, _ = steering.sample_tokens(ex, vocab, "cpu", 16, 40, 1.0, seed=4)
    assert not torch.equal(t1, t3)


def test_guidance_w0_equals_prefix_sampling(parts):
    vocab, plain, steered, _ = parts
    base = steering.guidance_experts("bandout", 0.0, steered, plain, vocab)
    # w = 0: coefficient 1 on the IN expert, 0 on the OUT expert.
    t_g, l_g = steering.sample_tokens(base, vocab, "cpu", 32, 40, 1.0, seed=5)
    t_p, l_p = steering.sample_tokens(base[:1], vocab, "cpu", 32, 40, 1.0, seed=5)
    assert torch.equal(t_g, t_p) and torch.equal(l_g, l_p)


def test_guidance_masks_special_tokens(parts):
    vocab, plain, steered, _ = parts
    ex = steering.guidance_experts("plain", 3.0, steered, plain, vocab)
    tok, _ = steering.sample_tokens(ex, vocab, "cpu", 64, 40, 1.0, seed=1, mask_special=True)
    forbidden = {vocab.stoi[t] for t in steering.SPECIAL_TOKENS if t != gen_model.EOS and t != gen_model.PAD}
    assert not (set(tok.flatten().tolist()) & forbidden)


def test_sequence_logprob_matches_manual(parts):
    vocab, plain, steered, _ = parts
    prefix = steering.prefix_ids(vocab, [gen_model.BAND_IN, gen_model.CLEAN])
    tok, ln = steering.sample_tokens([steering.Expert(steered, prefix)], vocab, "cpu", 4, 30, 1.0, 2)
    lp = steering.sequence_logprob(steered, vocab, prefix, tok, ln)
    bos = vocab.stoi[gen_model.BOS]
    for i in range(4):
        seq = torch.tensor([[bos] + prefix + tok[i, :ln[i]].tolist()])
        with torch.no_grad():
            logits, _ = steered(seq[:, :-1])
        logp = torch.log_softmax(logits, -1)
        tgt = seq[:, 1:]
        manual = logp[0, torch.arange(tgt.shape[1]), tgt[0]][len(prefix):].sum()
        assert float(lp[i].detach()) == pytest.approx(float(manual), rel=1e-4, abs=1e-4)


def test_reinvent_step_raises_reward_and_keeps_prior_fixed(parts):
    vocab, plain, steered, scorer = parts
    import copy
    agent = copy.deepcopy(plain)
    agent.gru.dropout = 0.0
    prior = steering.freeze(copy.deepcopy(plain))
    before = [p.clone() for p in prior.parameters()]
    cfg = steering.ReinventConfig(sigma=60.0, lr=3e-3, batch=64, bucket=0)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
    first = []
    last = []
    for step in range(25):
        s = steering.reinvent_step(agent, prior, vocab, "cpu", [], scorer, (400.0, 450.0),
                                   cfg, opt, seed=100 + step)
        (first if step < 5 else last if step >= 20 else []).append(s["reward_mean"])
        assert np.isfinite(s["loss"])
    assert all(torch.equal(a, b) for a, b in zip(before, prior.parameters()))
    # the agent moved away from the prior
    assert any(not torch.equal(a, b) for a, b in zip(prior.parameters(), agent.parameters()))
    assert np.mean(last) >= np.mean(first) - 0.05


def test_selftrain_round_runs_and_grows_pool(parts):
    vocab, plain, steered, scorer = parts
    import copy
    model = copy.deepcopy(steered)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    prefix = steering.prefix_ids(vocab, [gen_model.BAND_IN, gen_model.CLEAN])
    replay = [steering.build_ids(vocab, "[C][C][O]", prefix) for _ in range(20)]
    cfg = steering.SelfTrainConfig(steps_per_round=3, gen_per_round=200, batch=16)
    pool = []
    st = steering.selftrain_round(model, opt, vocab, "cpu", prefix, scorer, (200.0, 700.0),
                                  cfg, replay, pool, seed=1)
    assert st["pool"] == len(pool) > 0
    assert np.isfinite(st["loss"])


def test_screen_is_filter_not_reward():
    # SelfTrainConfig exposes the screen only as a boolean data filter.
    assert "screen_filter" in steering.SelfTrainConfig.__dataclass_fields__
    for field in steering.ReinventConfig.__dataclass_fields__:
        assert "screen" not in field and "rule" not in field


def test_compute_metrics_basic(fx, parts):
    vocab, plain, steered, scorer = parts
    import pandas as pd
    corpus = pd.read_csv(fx["corpus"])
    sel = corpus["selfies"].tolist()[:20] + ["", "[C][C]"]
    m = compute_metrics(sel, set(corpus["smiles"]), scorer, 400.0, 450.0, seed=0)
    assert m["n_sampled"] == 22
    assert 0.0 < m["validity"] < 1.0                 # the empty string is invalid
    assert m["novelty"] < 1.0                        # corpus molecules are not novel
    assert 0.0 <= m["in_band_fraction"] <= 1.0
    assert 0.0 < m["int_div"] < 1.0


def test_internal_diversity_extremes():
    assert internal_diversity(["CCO", "CCO"]) == pytest.approx(0.0)
    assert internal_diversity(["CCO", "c1ccccc1", "CCCCCCCCCC"]) > 0.5
    assert np.isnan(internal_diversity(["CCO"]))


def test_holdout_predictor_trains_and_differs_from_surrogate(fx):
    hp = holdout_predictor.train_holdout(fx["uv"], seed=11, n_estimators=30, n_jobs=1)
    assert set(hp.metrics) >= {"mae_nm", "r2", "n_test"}
    p = hp.predict(["c1ccccc1", None, "not a smiles", "c1ccc2ccccc2c1"])
    assert np.isnan(p[1]) and np.isnan(p[2]) and np.isfinite(p[0]) and np.isfinite(p[3])
    # different feature space from the surrogate's Morgan bits
    X, _ = holdout_predictor.featurize(["c1ccccc1"])
    assert X.shape[1] == holdout_predictor.N_BITS + 167 + len(holdout_predictor.DESCRIPTOR_NAMES)


def test_largest_conjugated_system():
    from rdkit import Chem
    assert holdout_predictor.largest_conjugated_system(Chem.MolFromSmiles("CCCC")) == 0
    assert holdout_predictor.largest_conjugated_system(Chem.MolFromSmiles("c1ccccc1")) == 6
    assert holdout_predictor.largest_conjugated_system(Chem.MolFromSmiles("c1ccc2ccccc2c1")) == 10
