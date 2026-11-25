"""Shared foundation for every script in ``src/``: paths, columns, data loading, folds, scores and runs.

Playground Series S6E11 (credit-card fraud, ROC-AUC). Everything that more than one script needs lives
here, so each idea is written once:

* **Paths.** ``DATA`` and ``RUNS`` follow ``S6E11_DATA`` / ``S6E11_RUNS`` and switch to the Kaggle layout
  when ``/kaggle/input`` exists; ``THREADS`` follows ``S6E11_THREADS`` (the Mac is shared: 4 or below).
* **Data.** :func:`load` returns train, test and the original dataset with one dtype convention.
* **Folds.** :func:`folds` is the one split every model and stack is scored on; never change its seed.
* **Scores.** :class:`CVScore` holds an out-of-fold AUC and its per-fold AUCs and compares two of them.
* **Rank transforms.** :func:`logit`, :func:`rank_fraction` and :func:`logit_rank`.
* **Runs.** :func:`save_run` and :func:`load_run` write and read ``runs/<name>/{oof,test}.npy`` and
  ``meta.json``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit as special_logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
ON_KAGGLE = Path("/kaggle/input").exists()
DATA = Path(os.environ.get("S6E11_DATA", ROOT / "data"))
RUNS = Path(os.environ.get("S6E11_RUNS", "/kaggle/working/runs" if ON_KAGGLE else ROOT / "runs"))
THREADS = int(os.environ.get("S6E11_THREADS", "4"))
"""Threads per model. The Mac is shared with a 24/7 pipeline: keep this at 4 or below."""

TARGET = "is_fraud"
N_FOLDS = 5
SEED = 42

CAT_STR = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOL = [
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
NUM = [
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
]
RAW = NUM + CAT_STR + BOOL

Fold = tuple[np.ndarray, np.ndarray]
"""Row positions of one fold: (training rows, validation rows)."""


class ConfigError(ValueError):
    """A command-line option or input file that cannot work; scripts turn it into a one-line exit."""


# --------------------------------------------------------------------------------------------- data


def find_data(name: str, *alts: str) -> Path:
    """Locate a data file in ``DATA``, or anywhere under ``/kaggle/input`` on Kaggle.

    Args:
        name: preferred file name.
        *alts: other names the same file may have (the original dataset has two).

    Raises:
        FileNotFoundError: no candidate exists; the message names every place searched.
    """
    for n in (name, *alts):
        if (DATA / n).exists():
            return DATA / n
    if ON_KAGGLE:
        for n in (name, *alts):
            hits = sorted(Path("/kaggle/input").rglob(n))
            if hits:
                return hits[0]
    where = f"{DATA}" + (" or /kaggle/input" if ON_KAGGLE else "")
    raise FileNotFoundError(f"{' / '.join((name, *alts))} not found in {where} (see docs/DATA.md)")


def _bool_codes(s: pd.Series, column: str) -> pd.Series:
    """Map a True/False column (any case, or 1/0) to int8, refusing anything else."""
    out = s.astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
    if out.isna().any():
        bad = sorted(s[out.isna()].astype(str).unique())[:5]
        raise ValueError(f"column {column}: values other than True/False: {bad}")
    return out.astype("int8")


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, test, original) with booleans as int8 and string categoricals as pandas category.

    Category levels are the union over all three frames, so codes agree everywhere. The original
    dataset's ``transaction_id`` is replaced by negative ids, so ids never collide with competition rows.

    Raises:
        FileNotFoundError: a data file is missing.
        ValueError: a flag column holds something other than True/False.
    """
    tr = pd.read_csv(find_data("train.csv"))
    te = pd.read_csv(find_data("test.csv"))
    og = pd.read_csv(find_data("original.csv", "credit_card_fraud_2026.csv")).drop(columns=["transaction_id"])
    og.insert(0, "id", -1 - np.arange(len(og)))
    for df in (tr, te, og):
        for c in BOOL:
            df[c] = _bool_codes(df[c], c)
    for c in CAT_STR:
        levels = sorted(set(tr[c]) | set(te[c]) | set(og[c]))
        for df in (tr, te, og):
            df[c] = pd.Categorical(df[c], categories=levels)
    tr[TARGET] = tr[TARGET].astype(int)
    og[TARGET] = og[TARGET].astype(int)
    return tr, te, og


# -------------------------------------------------------------------------------------------- folds


def folds(y: np.ndarray, n: int = N_FOLDS, seed: int = SEED) -> list[Fold]:
    """The fixed stratified folds over the competition training rows (original rows never validate)."""
    return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def fold_ids(y: np.ndarray) -> np.ndarray:
    """Fold index (0-4) of every training row under :func:`folds`, as int8."""
    f = np.empty(len(y), dtype=np.int8)
    for k, (_, i_va) in enumerate(folds(y)):
        f[i_va] = k
    return f


# ------------------------------------------------------------------------------------------- scores


@dataclass(frozen=True)
class CVScore:
    """Out-of-fold ROC-AUC over all scored rows and the AUC of each validation fold."""

    cv: float
    folds: tuple[float, ...]

    @classmethod
    def of(
        cls, y: np.ndarray, pred: np.ndarray, fl: Sequence[Fold], rows: np.ndarray | None = None
    ) -> CVScore:
        """Score out-of-fold predictions.

        Args:
            y: labels of all training rows.
            pred: one out-of-fold prediction per training row.
            fl: the folds the predictions were made on.
            rows: optional boolean mask of scored rows (probe runs leave the others NaN).
        """
        if rows is None:
            return cls(
                float(roc_auc_score(y, pred)), tuple(float(roc_auc_score(y[v], pred[v])) for _, v in fl)
            )
        per_fold = tuple(float(roc_auc_score(y[v[rows[v]]], pred[v[rows[v]]])) for _, v in fl)
        return cls(float(roc_auc_score(y[rows], pred[rows])), per_fold)

    def up(self, base: CVScore) -> int:
        """How many folds score higher than in ``base`` (the adoption rule needs 4 of 5)."""
        return sum(a > b for a, b in zip(self.folds, base.folds, strict=True))

    def fold_list(self) -> list[float]:
        """Per-fold AUCs rounded to five decimals, for printing."""
        return np.round(self.folds, 5).tolist()

    def __str__(self) -> str:
        return f"{self.cv:.5f}  folds {self.fold_list()}"


# ---------------------------------------------------------------------------------- rank transforms


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Log-odds of probabilities clipped to ``[eps, 1 - eps]``."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def rank_fraction(v: np.ndarray) -> np.ndarray:
    """Mid-ranks scaled into (0, 1): ``(rank - 0.5) / n``; ties share their mean rank."""
    return (rankdata(v) - 0.5) / len(v)


def logit_rank(v: np.ndarray) -> np.ndarray:
    """Logit of :func:`rank_fraction`, so scores on any scale enter a blend on the same one."""
    return special_logit(rank_fraction(v))


# --------------------------------------------------------------------------------------------- runs


def save_run(name: str, oof: np.ndarray, test: np.ndarray, meta: dict[str, Any]) -> Path:
    """Write OOF and test predictions plus metadata to ``runs/<name>/`` and return that directory."""
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "oof.npy", oof)
    np.save(d / "test.npy", test)
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return d


def load_run(name: str, refit: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Read ``runs/<name>/{oof,test}.npy`` (never unpickles).

    Args:
        name: run name.
        refit: take the test scores from ``test_refit.npy`` (one model fitted on all rows, D34) when the run
            has one; runs without it (public members) keep ``test.npy``.

    Raises:
        FileNotFoundError: the run does not exist; the message names the run.
    """
    d = RUNS / name
    if not (d / "oof.npy").exists() or not (d / "test.npy").exists():
        raise FileNotFoundError(f"run {name!r} not found in {RUNS}")
    test = d / "test_refit.npy" if refit and (d / "test_refit.npy").exists() else d / "test.npy"
    return np.load(d / "oof.npy", allow_pickle=False), np.load(test, allow_pickle=False)
