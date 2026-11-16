"""Feature sets. Each builder takes (train, test, original) and returns frames with identical columns.

Fold-safe by construction: nothing here looks at the competition target. Target statistics come only
from the original dataset, which never appears in validation.
"""

from __future__ import annotations

import pandas as pd

from common import BOOL, CAT_STR, NUM, RAW, TARGET


def fe_raw(tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame) -> list[pd.DataFrame]:
    """Raw columns only."""
    return [df[RAW].copy() for df in (tr, te, og)]


def fe_orig_te(tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame) -> list[pd.DataFrame]:
    """Raw columns plus, for every column, the original dataset's fraud rate and count at that value.

    Unseen values get NaN. Smoothing towards the global rate with weight 20 rows.
    """
    out = fe_raw(tr, te, og)
    prior = og[TARGET].mean()
    for c in RAW:
        key = og[c].astype(str)
        g = og.groupby(key, observed=True)[TARGET].agg(["sum", "count"])
        rate = (g["sum"] + 20 * prior) / (g["count"] + 20)
        for src, dst in zip((tr, te, og), out, strict=True):
            k = src[c].astype(str)
            dst[f"ogte_{c}"] = k.map(rate).astype("float32").values
            dst[f"ogcnt_{c}"] = k.map(g["count"]).astype("float32").values
    return out


def fe_combo(tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame) -> list[pd.DataFrame]:
    """fe_orig_te plus simple ratios and a count of risk flags."""
    out = fe_orig_te(tr, te, og)
    for src, dst in zip((tr, te, og), out, strict=True):
        dst["flags"] = src[BOOL].sum(axis=1).astype("int8")
        dst["amt_over_bal"] = (src.amount_usd / (src.account_balance_usd + 1)).astype("float32")
        dst["txn_per_hour"] = (src.txn_count_last_24h / (src.hours_since_last_txn + 0.1)).astype("float32")
        dst["risk_x_vel"] = (src.merchant_risk_score * src.velocity_score).astype("float32")
    return out


def fe_count(tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame) -> list[pd.DataFrame]:
    """fe_combo plus frequency of each numeric value over train+test+original (unsupervised)."""
    out = fe_combo(tr, te, og)
    allv = pd.concat([tr[NUM], te[NUM], og[NUM]])
    for c in NUM:
        vc = allv[c].value_counts()
        for src, dst in zip((tr, te, og), out, strict=True):
            dst[f"freq_{c}"] = src[c].map(vc).astype("float32").values
    return out


FEATURE_SETS = {"raw": fe_raw, "ogte": fe_orig_te, "combo": fe_combo, "count": fe_count}
CATS = CAT_STR
