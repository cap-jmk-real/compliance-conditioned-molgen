import copy
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steer_fixtures  # noqa: E402

from uv import gen_model, metasearch, metasearch2, steering  # noqa: E402


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return steer_fixtures.build(str(tmp_path_factory.mktemp("ms2")))


@pytest.fixture(scope="module")
def parts(fx):
    import joblib
    vocab = gen_model.Vocab.load(fx["vocab"])
    plain, _ = steering.load_generator(fx["plain"], vocab, "cpu")
    scorer = steering.UVScorer.from_bundle(joblib.load(fx["predictor"]))
    return vocab, plain, scorer


# -- diversity filter ---------------------------------------------------------

def test_murcko_scaffold():
    assert steering.murcko_scaffold("c1ccccc1C") == "c1ccccc1"
    assert steering.murcko_scaffold("CCO") == ""
    assert steering.murcko_scaffold(None) == ""
    assert steering.murcko_scaffold("not a smiles") == ""


def test_diversity_filter_batch_duplicates_and_identical_bucket():
    smi = ["c1ccccc1C", "c1ccccc1C", None, "c1ccccc1O"]
    r = steering.apply_diversity_filter(smi, np.ones(4), {}, {}, bucket=0)
    assert r.tolist() == [1.0, 0.0, 0.0, 1.0]
    seen = {}
    r1 = steering.apply_diversity_filter(["c1ccccc1C"], np.ones(1), seen, {}, bucket=2)
    r2 = steering.apply_diversity_filter(["c1ccccc1C"], np.ones(1), seen, {}, bucket=2)
    r3 = steering.apply_diversity_filter(["c1ccccc1C"], np.ones(1), seen, {}, bucket=2)
    assert (r1[0], r2[0], r3[0]) == (1.0, 1.0, 0.0)


def test_diversity_filter_scaffold_bucket_counts_only_rewarded_and_skips_acyclic():
    scaf = {}
    seen = {}
    # three different molecules, same benzene scaffold, bucket of 2: third earns nothing
    batch = ["c1ccccc1C", "c1ccccc1O", "c1ccccc1N"]
    r = steering.apply_diversity_filter(batch, np.ones(3), seen, scaf, scaffold_bucket=2)
    assert r.tolist() == [1.0, 1.0, 0.0]
    assert scaf == {"c1ccccc1": 2}
    # a zero-reward molecule does not fill a bucket
    scaf2 = {}
    r = steering.apply_diversity_filter(["c1ccccc1C", "c1ccccc1O"], np.array([0.0, 1.0]), {}, scaf2,
                                        scaffold_bucket=1)
    assert r.tolist() == [0.0, 1.0] and scaf2 == {"c1ccccc1": 1}
    # acyclic molecules are not bucketed
    r = steering.apply_diversity_filter(["CCO", "CCN", "CCC"], np.ones(3), {}, {}, scaffold_bucket=1)
    assert r.tolist() == [1.0, 1.0, 1.0]
    # input array is not modified
    inp = np.ones(2)
    steering.apply_diversity_filter(["c1ccccc1C", "c1ccccc1C"], inp, {}, {})
    assert inp.tolist() == [1.0, 1.0]


def test_applicability_domain_factor():
    ad = steering.ApplicabilityDomain(["c1ccccc1C", "c1ccc2ccccc2c1"], threshold=0.5)
    f = ad.factor(["c1ccccc1C", "CCCCCCCCCCCC", None])
    assert f[0] == 1.0 and f[1] < 1.0 and f[2] == 0.0
    assert ad.factor(["CCCCCCCCCCCC"], threshold=0.0).tolist() == [1.0]


# -- entropy, blending, reinvent options ---------------------------------------

def test_sequence_logprob_entropy_is_nonnegative_and_consistent(parts):
    vocab, plain, _ = parts
    ex = [steering.Expert(plain, [], 1.0)]
    tok, ln = steering.sample_tokens(ex, vocab, "cpu", 8, 40, 1.0, seed=1)
    lp = steering.sequence_logprob(plain, vocab, [], tok, ln)
    lp2, ent = steering.sequence_logprob(plain, vocab, [], tok, ln, return_entropy=True)
    assert torch.allclose(lp, lp2)
    assert ent.shape == (8,) and bool((ent >= 0).all())


def test_blend_experts_recipes(parts):
    vocab, plain, _ = parts
    prior = steering.freeze(copy.deepcopy(plain))
    one = steering.blend_experts(plain, prior, [], 0.0)
    assert len(one) == 1 and one[0].coef == 1.0
    two = steering.blend_experts(plain, prior, [], -0.25)
    assert [e.coef for e in two] == [0.75, 0.25]
    tok, ln = steering.sample_tokens(two, vocab, "cpu", 8, 30, 1.0, seed=2, mask_special=True)
    assert tok.shape == (8, 30)
    # with identical agent and prior the blend is the same distribution as the agent
    a, _ = steering.sample_tokens(one, vocab, "cpu", 16, 30, 1.0, seed=5)
    b, _ = steering.sample_tokens(steering.blend_experts(plain, plain, [], 0.5), vocab, "cpu", 16,
                                  30, 1.0, seed=5)
    assert torch.equal(a, b)


def test_reinvent_step_with_new_options_runs_and_records_memory(parts):
    vocab, plain, scorer = parts
    agent = copy.deepcopy(plain)
    agent.gru.dropout = 0.0
    prior = steering.freeze(copy.deepcopy(plain))
    cfg = steering.ReinventConfig(sigma=30.0, lr=1e-3, batch=32, bucket=2, scaffold_bucket=3,
                                  entropy_beta=5.0, ad_threshold=0.3, soft_width=25.0)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr)
    ad = steering.ApplicabilityDomain(["c1ccccc1C", "c1ccc2ccccc2c1", "c1ccncc1"])
    seen, scaf = {}, {}
    for step in range(3):
        s = steering.reinvent_step(agent, prior, vocab, "cpu", [], scorer, (400.0, 450.0), cfg, opt,
                                   seed=step, seen=seen, scaffolds=scaf, ad=ad)
        assert np.isfinite(s["loss"])
    assert seen and all(v >= 1 for v in seen.values())
    assert all(v <= cfg.scaffold_bucket for v in scaf.values())


def test_default_reinvent_config_is_backward_compatible():
    cfg = steering.ReinventConfig()
    assert cfg.scaffold_bucket == 0 and cfg.entropy_beta == 0.0 and cfg.ad_threshold == 0.0


# -- round 2 space and selection logic -----------------------------------------

def test_sample_space_r2_deterministic_distinct_with_reference():
    a = metasearch2.sample_space_r2(10, 10, seed=4)
    b = metasearch2.sample_space_r2(10, 10, seed=4)
    assert a == b and a[0] == metasearch2.BASE_SPEC
    ids = [metasearch2.spec_id(s) for s in a]
    assert len(set(ids)) == len(ids) == 21
    for s in a:
        assert set(s) == set(metasearch2.BASE_SPEC)
        assert not any("screen" in k or "rule" in k for k in s)    # no compliance knob
    # local neighbours differ from the reference in at most 3 knobs
    for s in a[1:11]:
        assert 1 <= sum(s[k] != metasearch2.BASE_SPEC[k] for k in s) <= 3


def test_reference_spec_matches_pilot_region():
    s = metasearch2.BASE_SPEC
    assert (s["init"], s["bucket"], s["lr"], s["sigma"], s["soft_width"], s["batch"],
            s["topk_frac"]) == ("plain", 0, 1e-4, 30.0, 25.0, 128, 1.0)


def _m(inband, uniq=1.0, div=0.88, **kw):
    d = {"validity": 1.0, "uniqueness": uniq, "novelty": 1.0, "rule_pass_rate": 1.0,
         "int_div": div, "in_band_fraction": inband, "in_band_holdout": inband - 0.01,
         "trial": "t"}
    d.update(kw)
    return d


def test_selection_limits_add_margin_but_never_touch_holdout_gap():
    lim = metasearch.Limits()
    sel = metasearch2.selection_limits(lim, 0.02)
    assert sel.uniqueness == pytest.approx(0.92) and sel.validity == 0.97 and sel.novelty == 0.92
    assert sel.div_ratio == pytest.approx(0.92) and sel.holdout_gap == lim.holdout_gap
    assert metasearch2.selection_limits(lim, 0.5).validity == 1.0     # capped


def test_best_snapshot_is_checkpoint_selection():
    lim = metasearch2.selection_limits(metasearch.Limits(), 0.01)
    snaps = {100: _m(0.40), 200: _m(0.55), 300: _m(0.70, uniq=0.80), 400: _m(0.90, uniq=0.30)}
    step, m, v = metasearch2.best_snapshot(snaps, lim, 0.885)
    assert step == 200 and not v
    # none feasible: the least-violating one is returned, with its violations
    bad = {100: _m(0.4, uniq=0.5), 200: _m(0.6, uniq=0.8)}
    step, m, v = metasearch2.best_snapshot(bad, lim, 0.885)
    assert step == 200 and "uniqueness" in v


def test_gate_report_uses_actual_gates_and_holdout_gap():
    lim = metasearch.Limits()
    ok = metasearch2.gate_report(_m(0.6), lim, 0.885)
    assert ok["passes_all"] and ok["holdout_gap"] == pytest.approx(0.01)
    hacked = metasearch2.gate_report(_m(0.6, in_band_holdout=0.1), lim, 0.885)
    assert hacked["feasible"] and not hacked["holdout_ok"] and not hacked["passes_all"]
    low_u = metasearch2.gate_report(_m(0.6, uniq=0.89), lim, 0.885)
    assert not low_u["passes_all"] and "uniqueness" in low_u["violations"]
    # just inside the diversity floor 0.9 x 0.885 = 0.7965
    assert metasearch2.gate_report(_m(0.6, div=0.797), lim, 0.885)["passes_all"]
    assert not metasearch2.gate_report(_m(0.6, div=0.796), lim, 0.885)["passes_all"]


def test_frontier_respects_other_gates_and_holdout():
    lim = metasearch.Limits()
    rows = [_m(0.50, uniq=0.95, step=1), _m(0.70, uniq=0.85, step=2),
            _m(0.90, uniq=0.75, step=3, in_band_holdout=0.1),      # holdout gap too big
            _m(0.80, uniq=0.80, step=4, validity=0.5)]              # fails another gate
    fr = {f["uniqueness_at_least"]: f["best"] for f in metasearch2.frontier(rows, lim, 0.885)}
    assert fr[0.95]["in_band"] == 0.50
    assert fr[0.85]["in_band"] == 0.70
    assert fr[0.70]["in_band"] == 0.70     # the hacked and invalid rows never qualify
    assert fr[0.90]["in_band"] == 0.50


# -- end to end ------------------------------------------------------------------

def argv(fx, root, extra=()):
    return ["--root", str(root), "--plain", fx["plain"], "--steered", fx["steered"],
            "--vocab", fx["vocab"], "--corpus", fx["corpus"], "--predictor", fx["predictor"],
            "--uvvis-data", fx["uv"], "--holdout-trees", "20",
            "--n-local", "3", "--n-global", "3", "--rungs", "4,8", "--eta", "2",
            "--snap-every", "2", "--eval-every", "2", "--snap-n-eval", "120",
            "--confirm-n-eval", "150", "--n-refine", "2", "--n-confirm", "2",
            "--replay-n", "200", "--cpu", *extra]


def test_search_end_to_end_resume_and_seeds_mode(fx, tmp_path, capsys):
    root = tmp_path / "r2"
    metasearch2.main(argv(fx, root))
    out = capsys.readouterr().out
    assert "=== rung0" in out and "=== confirm" in out and "gate:" in out
    res = json.load(open(root / "results_round2.json"))
    assert res["confirm"] and (root / "results_round2.md").exists()
    assert res["limits"]["uniqueness"] == 0.9                      # gates unchanged
    assert res["selection_limits"]["uniqueness"] == pytest.approx(0.91)
    trial_rows = [r for r in res["confirm"] if r["step"] > 0]
    assert trial_rows
    for r in trial_rows:
        assert (root / "trials" / r["trial"] / "snaps" / f"step-{r['step']:05d}.pt").exists()
        assert "passes_all" in r and "holdout_gap" in r
    assert len(res["frontier_snapshots"]) == 6

    before = json.load(open(root / "snap_results.json"))
    metasearch2.main(argv(fx, root))
    assert "resuming:" in capsys.readouterr().out
    assert json.load(open(root / "snap_results.json")) == before   # nothing recomputed

    spec = trial_rows[0]["spec"]
    sroot = tmp_path / "seeds"
    metasearch2.main([
        "--mode", "seeds", "--root", str(sroot), "--vocab", fx["vocab"], "--corpus", fx["corpus"],
        "--predictor", fx["predictor"], "--uvvis-data", fx["uv"], "--holdout-trees", "20",
        "--spec-json", json.dumps(spec), "--steps", "4", "--snap-every", "2",
        "--confirm-n-eval", "150", "--replay-n", "200", "--cpu",
        "--seed-priors", f"0:{fx['plain']}:{fx['steered']},1:{fx['plain']}:{fx['steered']}"])
    sres = json.load(open(sroot / "seeds_result.json"))
    assert len(sres["per_seed"]) == 2 and sres["steps"] == 4
    assert "in_band_fraction" in sres["summary_mean_sd"]
    assert (sroot / "seeds_result.md").exists()
