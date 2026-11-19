"""Boost a few shallow trees on the raw features starting from a stack's score (LightGBM init_score).

The zero-fraud zones (D27) showed that the stack still mis-ranks rows in simple regions of the raw
features. A residual booster can look for such corrections itself. For every outer fold it trains on the
other four, starting from the stack's nested out-of-fold logit, stops early on a 10% split of those four
folds (never on the scored fold) and scores the held-out fold. On the test set it starts from the stack's
test logit and runs the mean number of rounds the five folds chose, trained on all rows.

Usage (from src/):
  ../.venv/bin/python stack.py --no-hill --C 0.1 --oof-out X.npy --out X.csv <runs...>
  ../.venv/bin/python residual.py --oof X.npy --sub X.csv --leaves 4 [--out X_res.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from common import TARGET, folds, load

THREADS = 4


def logit(p: np.ndarray) -> np.ndarray:
    """Log-odds of probabilities, clipped away from 0 and 1."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", type=Path, required=True, help="nested OOF stack logits (stack.py --oof-out)")
    ap.add_argument("--sub", type=Path, help="submission CSV of the same stack (probabilities)")
    ap.add_argument("--out", type=Path, help="write the corrected submission")
    ap.add_argument("--leaves", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--min-leaf", type=int, default=1000)
    ap.add_argument("--l2", type=float, default=10.0)
    a = ap.parse_args()

    tr, te, _ = load()
    y = tr[TARGET].to_numpy()
    X = tr.drop(columns=["id", TARGET])
    Xt = te[X.columns]
    base = np.load(a.oof, allow_pickle=False)
    assert base.shape == (len(tr),) and np.isfinite(base).all()
    params = dict(
        objective="binary",
        learning_rate=a.lr,
        num_leaves=a.leaves,
        min_data_in_leaf=a.min_leaf,
        lambda_l2=a.l2,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        num_threads=THREADS,
        verbose=-1,
        seed=0,
    )

    pred = base.copy()
    rounds, before, after = [], [], []
    for i_tr, i_va in folds(y):
        i_fit, i_es = train_test_split(i_tr, test_size=0.1, stratify=y[i_tr], random_state=0)
        d_fit = lgb.Dataset(X.iloc[i_fit], y[i_fit], init_score=base[i_fit])
        d_es = lgb.Dataset(X.iloc[i_es], y[i_es], init_score=base[i_es], reference=d_fit)
        b = lgb.train(
            params, d_fit, 3000, valid_sets=[d_es], callbacks=[lgb.early_stopping(200, verbose=False)]
        )
        pred[i_va] = base[i_va] + b.predict(X.iloc[i_va], num_iteration=b.best_iteration, raw_score=True)
        rounds.append(b.best_iteration)
        before.append(roc_auc_score(y[i_va], base[i_va]))
        after.append(roc_auc_score(y[i_va], pred[i_va]))
    up = sum(x > z for x, z in zip(after, before, strict=True))
    print(f"  leaves {a.leaves} lr {a.lr} min_leaf {a.min_leaf}: rounds {rounds}")
    print(f"  stack     {roc_auc_score(y, base):.5f}  folds {np.round(before, 5).tolist()}")
    print(f"  + booster {roc_auc_score(y, pred):.5f}  folds {np.round(after, 5).tolist()}  up {up}/5")

    if a.sub is not None and a.out is not None:
        sub = pd.read_csv(a.sub)
        assert (sub["id"].to_numpy() == te["id"].to_numpy()).all()
        t0 = logit(sub[TARGET].to_numpy())
        n = max(1, int(round(np.mean(rounds))))
        b = lgb.train(params, lgb.Dataset(X, y, init_score=base), n)
        sub[TARGET] = 1 / (1 + np.exp(-(t0 + b.predict(Xt, raw_score=True))))
        sub.to_csv(a.out, index=False)
        print(f"wrote {a.out} ({n} rounds)")


if __name__ == "__main__":
    main()
