"""Shared fixtures. The tests need no competition data: they build small synthetic frames."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from common import BOOL, CAT_STR, NUM, TARGET  # noqa: E402


@pytest.fixture
def frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Small (train, test, original) frames with the competition's columns and dtypes."""
    rng = np.random.default_rng(0)

    def frame(n: int, start_id: int) -> pd.DataFrame:
        df = pd.DataFrame({"id": np.arange(start_id, start_id + n)})
        for c in NUM:
            df[c] = np.round(rng.gamma(2.0, 10.0, n), 2)
        for c in CAT_STR:
            df[c] = pd.Categorical(rng.choice(["a", "b", "c"], n), categories=["a", "b", "c"])
        for c in BOOL:
            df[c] = rng.integers(0, 2, n).astype("int8")
        risk = df["merchant_risk_score"].to_numpy() + 20 * df["used_vpn"].to_numpy()
        df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-(risk - 45) / 6))).astype(int)
        return df

    tr, te, og = frame(2000, 0), frame(500, 10_000), frame(600, -600)
    return tr, te.drop(columns=TARGET), og
