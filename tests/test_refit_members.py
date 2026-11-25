"""Refit command lines rebuilt from saved run metadata (D34)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import common
import refit_members
from train_gbdt import check_args, parse_args


def test_refit_argv_round_trips_a_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(refit_members, "RUNS", tmp_path)
    monkeypatch.setattr(common, "RUNS", tmp_path)
    meta = dict(
        model="lgb", fe="art", orig=0.0, params='{"num_leaves": 4}', te="single", te_smooth="5,20,100",
        keys="full", offset="glm", joint="glm", key_counts=True, seeds=3, es="val", cv=0.95,
    )  # fmt: skip
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "meta.json").write_text(json.dumps(meta))
    argv = refit_members.refit_argv("r")
    a = parse_args(argv)
    check_args(a)
    assert a.refit_only and a.es == "inner" and a.key_counts and a.seeds == 3
    assert (a.fe, a.offset, a.joint, a.te_smooth) == ("art", "glm", "glm", "5,20,100")
    assert json.loads(a.params) == {"num_leaves": 4} and " " not in a.params
