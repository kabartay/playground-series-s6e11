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


def fe_ogmodel(tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame) -> list[pd.DataFrame]:
    """fe_count plus the prediction of a LightGBM trained on the original dataset only.

    Train and test get a model fitted on all original rows; original rows get 5-fold out-of-fold
    predictions so none sees its own label. The competition target is never read.
    """
    import lightgbm as lgb
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    out = fe_count(tr, te, og)
    params = dict(
        objective="binary",
        learning_rate=0.03,
        n_estimators=400,
        num_leaves=15,
        min_child_samples=50,
        feature_fraction=0.7,
        verbose=-1,
        num_threads=4,
    )
    y = og[TARGET].to_numpy()
    oof = np.zeros(len(og))
    for i_tr, i_va in StratifiedKFold(5, shuffle=True, random_state=0).split(og, y):
        m = lgb.LGBMClassifier(**params).fit(og[RAW].iloc[i_tr], y[i_tr])
        oof[i_va] = m.predict_proba(og[RAW].iloc[i_va])[:, 1]
    m = lgb.LGBMClassifier(**params).fit(og[RAW], y)
    out[0]["ogmodel"] = m.predict_proba(tr[RAW])[:, 1].astype("float32")
    out[1]["ogmodel"] = m.predict_proba(te[RAW])[:, 1].astype("float32")
    out[2]["ogmodel"] = oof.astype("float32")
    return out


FEATURE_SETS = {
    "raw": fe_raw,
    "ogte": fe_orig_te,
    "combo": fe_combo,
    "count": fe_count,
    "ogmodel": fe_ogmodel,
}
CATS = CAT_STR
