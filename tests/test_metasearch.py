import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import steer_fixtures  # noqa: E402

from uv import metasearch  # noqa: E402


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return steer_fixtures.build(str(tmp_path_factory.mktemp("ms")))


def argv(fx, root, extra=()):
    return ["--root", str(root), "--plain", fx["plain"], "--steered", fx["steered"],
            "--vocab", fx["vocab"], "--corpus", fx["corpus"], "--predictor", fx["predictor"],
            "--uvvis-data", fx["uv"], "--holdout-trees", "20",
            "--n-guidance", "3", "--n-reinvent", "3", "--n-selftrain", "2",
            "--rungs", "4:60,8:80", "--eta", "2", "--n-confirm", "2",
            "--confirm-n-eval", "100", "--replay-n", "200", "--cpu", *extra]


def test_parse_rungs():
    assert metasearch.parse_rungs("60:500,180:1000") == [(60, 500), (180, 1000)]


def test_sample_space_deterministic_and_distinct():
    a = metasearch.sample_space(["guidance", "reinvent", "selftrain"],
                                {"guidance": 5, "reinvent": 5, "selftrain": 5}, seed=3)
    b = metasearch.sample_space(["guidance", "reinvent", "selftrain"],
                                {"guidance": 5, "reinvent": 5, "selftrain": 5}, seed=3)
    assert a == b and len(a) == 15
    ids = [metasearch.trial_id(s) for s in a]
    assert len(set(ids)) == 15
    for s in a:
        assert s["family"] in metasearch.FAMILIES
        # no structural-screen or compliance knob in the search space
        assert not any("screen" in k or "rule" in k for k in s)


def test_violations_and_rank_key():
    lim = metasearch.Limits()
    good = {"validity": 1.0, "uniqueness": 0.99, "novelty": 1.0, "rule_pass_rate": 0.99,
            "int_div": 0.85, "in_band_fraction": 0.3}
    assert metasearch.violations(good, lim, 0.86) == {}
    collapsed = dict(good, uniqueness=0.2, int_div=0.3, in_band_fraction=0.9)
    v = metasearch.violations(collapsed, lim, 0.86)
    assert "uniqueness" in v and "int_div" in v
    # feasible beats infeasible even with lower in-band fraction
    assert metasearch.rank_key(good, {}) < metasearch.rank_key(collapsed, v)


def test_end_to_end_cpu_smoke_and_resume(fx, tmp_path, capsys):
    root = tmp_path / "run"
    metasearch.main(argv(fx, root))
    out = capsys.readouterr().out
    assert "=== rung0" in out and "=== confirm" in out and "gate:" in out
    res = json.load(open(root / "results.json"))
    assert res["confirm"] and res["rungs"] == [[4, 60], [8, 80]]
    methods = {r["method"] for r in res["confirm"]}
    assert {"plain", "prefix"} <= methods
    for r in res["confirm"]:
        for key in ("in_band_fraction", "validity", "uniqueness", "novelty",
                    "rule_pass_rate", "int_div", "in_band_holdout"):
            assert key in r
    assert (root / "results.md").exists() and (root / "history.json").exists()
    assert res["holdout_predictor"]["mae_nm"] >= 0
    # trained survivors have checkpoints on disk
    trained = [r for r in res["confirm"] if r["method"] in ("reinvent", "selftrain")]
    for r in trained:
        assert (root / "trials" / r["trial"] / "state.pt").exists()
        assert r["steps"] >= 8

    # resume: a second run does no new work (all results are on disk)
    before = json.load(open(root / "rung_results.json"))
    metasearch.main(argv(fx, root))
    out2 = capsys.readouterr().out
    assert "resuming:" in out2
    assert "step " not in out2.replace("steps", "")      # no training heartbeat
    after = json.load(open(root / "rung_results.json"))
    assert before == after


def test_interrupted_run_continues_from_checkpoint(fx, tmp_path, capsys, monkeypatch):
    root = tmp_path / "run2"
    args = argv(fx, root)
    orig = metasearch.Search._run
    state = {"crash": True}

    def crashing(self, trial, stage, steps, n_eval, eval_seed):
        if stage == "rung1" and state["crash"]:
            raise RuntimeError("simulated interruption")
        return orig(self, trial, stage, steps, n_eval, eval_seed)

    monkeypatch.setattr(metasearch.Search, "_run", crashing)
    with pytest.raises(RuntimeError):
        metasearch.main(args)
    capsys.readouterr()
    rung0 = json.load(open(root / "rung_results.json"))
    assert rung0 and all(k.startswith(("rung0|")) for k in rung0)
    import torch
    trained = [d for d in os.listdir(root / "trials") if (root / "trials" / d / "state.pt").exists()]
    assert trained
    for d in trained:
        assert torch.load(root / "trials" / d / "state.pt", weights_only=False)["steps"] >= 4

    state["crash"] = False
    metasearch.main(args)
    capsys.readouterr()
    final = json.load(open(root / "rung_results.json"))
    for k, v in rung0.items():                       # rung-0 results were not recomputed
        assert final[k] == v
    assert any(k.startswith("confirm|") for k in final)
    for d in trained:
        assert torch.load(root / "trials" / d / "state.pt", weights_only=False)["steps"] >= 4


def test_changed_schedule_is_refused(fx, tmp_path, capsys):
    root = tmp_path / "run3"
    metasearch.main(argv(fx, root))
    with pytest.raises(SystemExit):
        metasearch.main([a if a != "4:60,8:80" else "4:60,9:80" for a in argv(fx, root)])


def test_cost_projection_arithmetic():
    from uv import metasearch_cost as mc
    assert mc.survivors_per_rung(24, 3, 3.0, 3) == [24, 8, 3]
    t = {"reinvent_step_s": 1.0, "selftrain_step_s": 0.5, "eval_s_per_1000": 10.0}
    p = mc.project(t, {"guidance": 4, "reinvent": 4, "selftrain": 4},
                   [(10, 100), (30, 200)], eta=2.0, n_confirm=2, confirm_n_eval=500)
    assert p["upper_hours"] >= p["central_hours"] > 0
    # no trained trials and no steps: only evaluation cost remains
    q = mc.project(t, {"guidance": 4}, [(0, 1000)], eta=2.0, n_confirm=1, confirm_n_eval=1000)
    assert q["central_hours"] == pytest.approx((4 + 2) * 10.0 / 3600 + (1 + 2) * 10.0 / 3600)


def test_cost_measure_runs_on_cpu(fx, tmp_path):
    from uv import metasearch_cost as mc
    args = metasearch.make_parser().parse_args(argv(fx, tmp_path / "cost"))
    ctx = metasearch.build_context(args)
    t = mc.measure(ctx, reps=1)
    assert all(v > 0 for v in t.values())
