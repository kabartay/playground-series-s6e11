"""Array sources and member checks of import_public.py on tiny temporary files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import folds
from import_public import STATED, Member, check, fold_status, load_array, load_folds, slug


def member(**kw: object) -> Member:
    base: dict[str, object] = dict(
        tag="t", name="My Model", oof=Path("o"), test=Path("t"), auc=None, folds=None, source="s", note=""
    )
    base.update(kw)
    return Member(**base)  # type: ignore[arg-type]


def test_load_array_reads_every_source_kind(tmp_path: Path) -> None:
    a = np.arange(6.0).reshape(3, 2)
    np.save(tmp_path / "a.npy", a)
    np.savez(tmp_path / "z.npz", k=np.arange(3))
    pd.DataFrame({"p": [0.1, 0.2], "fold": [0, 1]}).to_csv(tmp_path / "c.csv", index=False)
    assert np.array_equal(load_array(tmp_path / "a.npy"), a)
    assert np.array_equal(load_array((tmp_path / "a.npy", 1)), a[:, 1])
    assert np.array_equal(load_array((tmp_path / "z.npz", "k")), np.arange(3))
    assert np.allclose(load_array((tmp_path / "c.csv", "p")), [0.1, 0.2])
    assert load_folds(tmp_path / "c.csv").tolist() == [0, 1]


def test_run_names_are_slugged() -> None:
    assert slug("RealMLP (8 seeds)!") == "realmlp__8_seeds"
    assert member().run == "pub_t_my_model"


def test_check_reports_the_first_failing_condition() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    ours = np.zeros(500, np.int8)
    for k, (_, v) in enumerate(folds(y)):
        ours[v] = k
    oof, test = y + rng.normal(0, 1, 500), np.zeros(10)
    assert check(member(), oof, test, y, ours, 10) is None
    assert "shapes" in check(member(), oof[:-1], test, y, ours, 10)
    assert "non-finite" in check(member(), np.r_[oof[:-1], np.nan], test, y, ours, 10)
    assert "source reports" in check(member(auc=0.5), oof, test, y, ours, 10)
    assert "fold AUCs" in check(member(fold_aucs=[0.5] * 5), oof, test, y, ours, 10)


def test_fold_status_wording() -> None:
    assert fold_status(member(folds=STATED)) == "stated"
    assert fold_status(member(folds=STATED, fold_aucs=[1.0] * 5)) == "stated, fold AUCs match"
    assert fold_status(member(folds=Path("f.npy"))) == "checked"
    assert fold_status(member()) == "unknown"
