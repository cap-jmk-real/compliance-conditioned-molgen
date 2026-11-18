"""Tests for the second-dataset replication (uv/dataset2). No network, no real data."""
import io
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from uv import evaluate as ev  # noqa: E402
from uv import gen_model  # noqa: E402
from uv.dataset2 import evaluate_ds2 as e2  # noqa: E402
from uv.dataset2 import prepare_beard2019 as prep  # noqa: E402
from uv.dataset2 import relabel_corpus as rel  # noqa: E402

HEADER = "SMI,lambda1 (sTDA, nm),F1 (sTDA),lamda1  (TD-DFT, nm),F1 (TD-DFT),lambda_max (Exp,  nm),extinction (Lmol-1cm-1),solvent\n"


def _csv(rows):
    return (HEADER + "".join(f"{s},NaN,NaN,NaN,NaN,{v},NaN,NaN\n" for s, v in rows)).encode()


def test_csv_header_with_commas_is_parsed_and_checksum_enforced(monkeypatch):
    raw = _csv([("CCO", 210), ("c1ccccc1", 254)])
    with pytest.raises(RuntimeError):
        prep.read_csv_bytes(raw)                     # wrong sha256 is rejected
    monkeypatch.setattr(prep, "CSV_SHA256", prep.sha256_bytes(raw))
    df = prep.read_csv_bytes(raw)
    assert list(df.columns) == prep.COLUMNS and len(df) == 2
    assert df["exp_nm"].tolist() == [210, 254]


def test_build_filters_range_drops_bad_smiles_and_takes_median(monkeypatch):
    raw = _csv([("c1ccccc1", 250), ("C1=CC=CC=C1", 260), ("c1ccccc1", 300),  # same structure x3
                ("CCO", 5000), ("CCN", 100),                                 # out of range
                ("not_a_smiles", 400), ("CC=O", "NaN")])
    monkeypatch.setattr(prep, "CSV_SHA256", prep.sha256_bytes(raw))
    out, st = prep.build(prep.read_csv_bytes(raw))
    assert st["raw_rows"] == 7
    assert st["rows_numeric_exp"] == 6
    assert st["rows_in_range"] == 4
    assert st["rows_unparsable_dropped"] == 1
    assert len(out) == 1 and out["abs_max_nm"].iloc[0] == 260.0   # median of 250, 260, 300


def test_compare_reports_overlap(tmp_path):
    ds2 = pd.DataFrame({"smiles": ["a", "b", "c", "d"], "abs_max_nm": [400.0, 300.0, 200.0, 1.0]})
    ds1 = pd.DataFrame({"smiles": ["a", "b", "c", "z"], "abs_max_nm": [410.0, 310.0, 210.0, 500.0]})
    f = tmp_path / "ds1.csv"
    ds1.to_csv(f, index=False)
    out = prep.compare(ds2, str(f), 400, 450)
    assert out["n_overlap"] == 3 and out["overlap_mae_nm"] == 10.0


class _Const:
    def predict(self, x):
        return np.array([420.0 + 40.0 * float(x[0, :8].sum() % 2)])


def test_relabel_keeps_everything_but_abs_pred():
    df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1"], "selfies": ["[C][C][O]", "[C][=C]"],
                       "abs_pred": [1.0, 2.0], "passes_screen": [True, False]})

    class M:
        def predict(self, X):
            return np.full(len(X), 433.0)
    out = rel.relabel(df, M(), 2048, chunk=1)
    assert out["abs_pred"].tolist() == [433.0, 433.0]
    for c in ("smiles", "selfies", "passes_screen"):
        assert out[c].equals(df[c])


def test_gate_report_thresholds():
    ok = {"validity": 0.99, "uniqueness": 0.95, "novelty": 0.95, "rule_pass_rate": 0.97,
          "int_div": 0.85}
    assert e2.gate_report(ok, plain_int_div=0.9)["passes_all"]
    low_div = dict(ok, int_div=0.80)                 # < 0.9 * 0.9 = 0.81
    g = e2.gate_report(low_div, plain_int_div=0.9)
    assert not g["int_div"]["pass"] and not g["passes_all"]
    assert not e2.gate_report(dict(ok, novelty=0.89), 0.9)["novelty"]["pass"]


def test_internal_diversity_bounds():
    same = e2.internal_diversity(["CCO", "CCO", "CCO"])
    diff = e2.internal_diversity(["CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O"])
    assert same == pytest.approx(0.0) and 0.0 < diff <= 1.0
    assert np.isnan(e2.internal_diversity(["CCO"]))


def test_eval_arm_ext_matches_uv_evaluate_on_shared_metrics(tmp_path):
    """The extended evaluator must reproduce uv.evaluate.eval_arm on the shared keys."""
    torch.manual_seed(0)
    toks = ["[C]", "[=C]", "[O]", "[N]", "[Branch1]", "[Ring1]", "[#C]"]
    vocab = gen_model.Vocab(toks)
    lm = gen_model.SelfiesLM(len(vocab), 8, 16, 1)
    path = tmp_path / "g.pt"
    torch.save({"model": lm.state_dict(), "arm": "steered", "emb": 8, "hidden": 16,
                "layers": 1}, path)
    predictor = _Const()
    args = (str(path), vocab, "cpu", 96, set(), predictor, 2048, 400.0, 450.0, 3)
    ref, ref_flags = ev.eval_arm(*args)
    new, new_flags = e2.eval_arm_ext(*args)
    for k, v in ref.items():
        assert new[k] == v, k
    assert np.array_equal(ref_flags, new_flags)
    assert "int_div" in new
