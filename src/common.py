"""Shared data loading, folds and feature building for Playground S6E11 (credit card fraud, ROC-AUC)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
TARGET = "is_fraud"
N_FOLDS = 5
SEED = 42

CAT_STR = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOL = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
    "billing_shipping_mismatch", "is_ai_generated_scam_attempt",
]
NUM = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
    "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count", "velocity_score",
    "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes",
]
RAW = NUM + CAT_STR + BOOL


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, test, original) with booleans as int8 and string categoricals as pandas category.

    Category levels are the union over all three frames so codes agree everywhere.
    """
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "test.csv")
    og = pd.read_csv(DATA / "original.csv").drop(columns=["transaction_id"])
    og.insert(0, "id", -1 - np.arange(len(og)))
    for df in (tr, te, og):
        for c in BOOL:
            df[c] = df[c].astype(str).str.lower().map({"true": 1, "false": 0}).astype("int8")
    for c in CAT_STR:
        levels = sorted(set(tr[c]) | set(te[c]) | set(og[c]))
        for df in (tr, te, og):
            df[c] = pd.Categorical(df[c], categories=levels)
    tr[TARGET] = tr[TARGET].astype(int)
    og[TARGET] = og[TARGET].astype(int)
    return tr, te, og


def folds(y: np.ndarray, n: int = N_FOLDS, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fixed stratified folds over the competition train rows (original rows never enter validation)."""
    return list(StratifiedKFold(n, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def save_run(name: str, oof: np.ndarray, test: np.ndarray, meta: dict) -> Path:
    """Write OOF and test predictions plus metadata to runs/<name>/."""
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "oof.npy", oof)
    np.save(d / "test.npy", test)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return d
