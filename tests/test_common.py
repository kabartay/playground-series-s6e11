"""Folds, scores, rank transforms, data checks and run files in common.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import logit as scipy_logit

import common
from common import CVScore, fold_ids, folds, logit, logit_rank, rank_fraction


def test_folds_partition_rows_and_keep_the_fraud_rate() -> None:
    y = np.r_[np.ones(100), np.zeros(900)].astype(int)
    fl = folds(y)
    assert len(fl) == 5
    val = np.concatenate([v for _, v in fl])
    assert np.array_equal(np.sort(val), np.arange(len(y)))
    for tr, va in fl:
        assert not np.intersect1d(tr, va).size
        assert y[va].sum() == 20


def test_folds_are_fixed() -> None:
    y = np.random.default_rng(1).integers(0, 2, 500)
    for (a_tr, a_va), (b_tr, b_va) in zip(folds(y), folds(y), strict=True):
        assert np.array_equal(a_tr, b_tr) and np.array_equal(a_va, b_va)


def test_fold_ids_agree_with_folds() -> None:
    y = np.random.default_rng(2).integers(0, 2, 300)
    ids = fold_ids(y)
    for k, (_, va) in enumerate(folds(y)):
        assert (ids[va] == k).all()


def test_cvscore_counts_folds_up_and_formats() -> None:
    a, b = CVScore(0.9, (0.1, 0.5, 0.5, 0.9, 0.2)), CVScore(0.8, (0.2, 0.4, 0.5, 0.8, 0.1))
    assert a.up(b) == 3
    assert str(a).startswith("0.90000  folds [0.1, 0.5")


def test_cvscore_with_row_mask_scores_only_masked_rows() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    pred = y + rng.normal(0, 1, 400)
    rows = rng.random(400) < 0.7
    fl = folds(y)
    s = CVScore.of(y, np.where(rows, pred, np.nan), fl, rows)
    assert np.isfinite(s.cv) and len(s.folds) == 5


def test_logit_clips_and_inverts() -> None:
    p = np.array([0.0, 0.5, 1.0])
    out = logit(p)
    assert out[1] == 0 and np.isfinite(out).all() and np.isclose(out[0], -out[2])


def test_rank_fraction_handles_ties_and_stays_inside_unit_interval() -> None:
    r = rank_fraction(np.array([3.0, 1.0, 1.0, 2.0]))
    assert np.allclose(r, [0.875, 0.25, 0.25, 0.625])
    assert np.array_equal(logit_rank(np.array([3.0, 1.0, 1.0, 2.0])), scipy_logit(r))


def test_logit_rank_ignores_scale() -> None:
    v = np.random.default_rng(4).normal(size=50)
    assert np.allclose(logit_rank(v), logit_rank(np.exp(v)))


def test_bool_codes_refuses_unknown_values() -> None:
    ok = common._bool_codes(pd.Series(["True", "false", "TRUE"]), "flag")
    assert ok.tolist() == [1, 0, 1] and ok.dtype == "int8"
    with pytest.raises(ValueError, match="flag"):
        common._bool_codes(pd.Series(["True", "maybe"]), "flag")


def test_find_data_names_the_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "DATA", tmp_path)
    monkeypatch.setattr(common, "ON_KAGGLE", False)
    (tmp_path / "b.csv").write_text("x\n")
    assert common.find_data("a.csv", "b.csv") == tmp_path / "b.csv"
    with pytest.raises(FileNotFoundError, match=r"c\.csv"):
        common.find_data("c.csv")


def test_save_and_load_run_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "RUNS", tmp_path)
    oof, test = np.arange(5.0), np.arange(3.0)
    d = common.save_run("r", oof, test, {"cv": 0.5, "path": Path("x")})
    assert (d / "meta.json").read_text().count('"cv": 0.5') == 1
    got_oof, got_test = common.load_run("r")
    assert np.array_equal(got_oof, oof) and np.array_equal(got_test, test)
    with pytest.raises(FileNotFoundError, match="missing"):
        common.load_run("missing")
