"""Job files and kernel bundles of scripts/kaggle_kernel.py (nothing is pushed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import kaggle_kernel


def test_read_jobs_drops_comments_and_blank_lines(tmp_path: Path) -> None:
    f = tmp_path / "j.txt"
    f.write_text("# header\n--model lgb --name a  # first\n\npip install pytabkit\n")
    assert kaggle_kernel.read_jobs(f) == ["--model lgb --name a", "pip install pytabkit"]


def test_read_jobs_refuses_an_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "j.txt"
    f.write_text("# nothing\n")
    with pytest.raises(ValueError, match="no job"):
        kaggle_kernel.read_jobs(f)


def test_build_embeds_every_module_and_sets_the_machine(tmp_path: Path) -> None:
    kaggle_kernel.build("k1", ["--model glm --name g"], tmp_path, internet=False, gpu=True)
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert meta["id"].endswith("/s6e11-k1") and meta["is_private"] == "true"
    assert meta["enable_gpu"] == "true" and meta["machine_shape"] == "NvidiaTeslaT4"
    run = (tmp_path / "run.py").read_text()
    for module in ("common.py", "learners.py", "encoding.py", "train_gbdt.py"):
        assert repr(module) in run
    compile(run, "run.py", "exec")


def test_push_without_jobs_is_a_usage_error() -> None:
    with pytest.raises(SystemExit):
        kaggle_kernel.main(["push", "--kernel", "k1"])
