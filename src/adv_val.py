"""Adversarial validation (D30): can a classifier tell train rows from test rows?

A LightGBM model is trained, with 5-fold CV over train + test rows, to predict whether a row comes from
``test.csv``. An AUC near 0.5 means the two files are drawn alike, so out-of-fold CV is a fair stand-in for
the private board. Otherwise the out-of-fold P(test) of each training row gives importance weights
``p / (1 - p)``, and a stack's OOF AUC weighted by them estimates its AUC on test-like rows.

Run from ``src/``::

    ../.venv/bin/python adv_val.py --fe raw --oof ../runs/pub_fa_wide_lm_stack/oof.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import TARGET, THREADS, load
from features import FEATURE_SETS


def adversarial_oof(X: pd.DataFrame, is_test: np.ndarray, seed: int = 0) -> tuple[np.ndarray, pd.Series]:
    """Return out-of-fold P(test) for every row and the mean split gain of each feature.

    Args:
        X: features of train rows followed by test rows (no id, no target).
        is_test: 1 for test rows, 0 for train rows.
        seed: seed of the CV split and the model.
    """
    p = np.zeros(len(X))
    gain = pd.Series(0.0, index=X.columns)
    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        feature_fraction=0.8,
        n_estimators=400,
        n_jobs=THREADS,
        random_state=seed,
        verbose=-1,
    )
    for i_fit, i_app in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, is_test):
        m = lgb.LGBMClassifier(**params).fit(X.iloc[i_fit], is_test[i_fit])
        p[i_app] = m.predict_proba(X.iloc[i_app])[:, 1]
        gain += pd.Series(m.booster_.feature_importance("gain"), index=X.columns) / 5
    return p, gain.sort_values(ascending=False)


def main() -> None:
    """Print the adversarial AUC, the most separating features and, with ``--oof``, weighted OOF AUCs."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fe", default="raw", choices=list(FEATURE_SETS), help="feature set (as train_gbdt.py)")
    ap.add_argument("--oof", type=Path, nargs="*", default=[], help="OOF score files to reweight")
    a = ap.parse_args()

    tr, te, og = load()
    Xtr, Xte, _ = FEATURE_SETS[a.fe](tr, te, og)
    X = pd.concat([Xtr, Xte], ignore_index=True).drop(columns=["id"], errors="ignore")
    is_test = np.r_[np.zeros(len(Xtr), int), np.ones(len(Xte), int)]
    p, gain = adversarial_oof(X, is_test)
    print(f"features {a.fe} ({X.shape[1]} columns): adversarial AUC {roc_auc_score(is_test, p):.5f}")
    print("top features by gain share:")
    print((gain / gain.sum()).head(8).round(3).to_string())

    p_tr = np.clip(p[: len(Xtr)], 1e-6, 1 - 1e-6)
    w = p_tr / (1 - p_tr)
    w /= w.mean()
    ess = w.sum() ** 2 / (w**2).sum()
    print(
        f"train-row weights p/(1-p): min {w.min():.3f} max {w.max():.3f}; effective sample size {ess:,.0f}"
        f" of {len(w):,}"
    )
    y = tr[TARGET].to_numpy()
    q = np.quantile(p_tr, [0.1, 0.9])
    lo, hi = p_tr <= q[0], p_tr >= q[1]
    print(
        f"fraud rate: all {y.mean():.4f}, least test-like decile {y[lo].mean():.4f}, "
        f"most test-like decile {y[hi].mean():.4f}"
    )
    for f in a.oof:
        s = np.load(f, allow_pickle=False)
        a_all, a_w = roc_auc_score(y, s), roc_auc_score(y, s, sample_weight=w)
        a_lo, a_hi = roc_auc_score(y[lo], s[lo]), roc_auc_score(y[hi], s[hi])
        print(
            f"{f.parent.name}/{f.name}: OOF AUC {a_all:.5f}, weighted {a_w:.5f}; "
            f"least / most test-like decile {a_lo:.5f} / {a_hi:.5f}"
        )


if __name__ == "__main__":
    main()
