import json
import math

from uv import aggregate_seeds as ag


def test_t_quantile_matches_tables():
    assert abs(ag.t_quantile_two_sided(0.05, 2) - 4.3027) < 1e-3
    assert abs(ag.t_quantile_two_sided(0.05, 10) - 2.2281) < 1e-3


def test_sd_and_p():
    assert ag.sd([1.0]) == 0.0
    assert abs(ag.sd([1.0, 2.0, 3.0]) - 1.0) < 1e-12
    assert abs(ag.t_two_sided_p(4.3027, 2) - 0.05) < 1e-3


def test_main_writes_summary(tmp_path, monkeypatch):
    paths = []
    for seed, (p, q) in enumerate([(0.18, 0.25), (0.19, 0.25), (0.17, 0.26)]):
        f = tmp_path / f"eval_seed{seed}.json"
        f.write_text(json.dumps({"seed": seed, "arms": {
            "plain": {"in_band_fraction": p, "validity": 1.0},
            "steered": {"in_band_fraction": q, "validity": 1.0}}}))
        paths.append(str(f))
    out = tmp_path / "agg.json"
    monkeypatch.setattr("sys.argv", ["x", *paths, "--out-json", str(out)])
    ag.main()
    s = json.loads(out.read_text())
    eff = s["steering_effect"]
    assert s["n_seeds"] == 3 and eff["n"] == 3
    assert abs(eff["mean_diff"] - (0.07 + 0.06 + 0.09) / 3) < 1e-9
    lo, hi = eff["ci95_mean_diff"]
    assert lo < eff["mean_diff"] < hi and math.isfinite(lo)
