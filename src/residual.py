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
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import TARGET, THREADS, CVScore, folds, load, logit


def main(argv: list[str] | None = None) -> int:
    """Score the residual booster fold by fold; optionally write the corrected submission."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", type=Path, required=True, help="nested OOF stack logits (stack.py --oof-out)")
    ap.add_argument("--sub", type=Path, help="submission CSV of the same stack (probabilities)")
    ap.add_argument("--out", type=Path, help="write the corrected submission")
    ap.add_argument("--leaves", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--min-leaf", type=int, default=1000)
    ap.add_argument("--l2", type=float, default=10.0)
    a = ap.parse_args(argv)

    tr, te, _ = load()
    y = tr[TARGET].to_numpy()
    X = tr.drop(columns=["id", TARGET])
    Xt = te[X.columns]
    base = np.load(a.oof, allow_pickle=False)
    if base.shape != (len(tr),) or not np.isfinite(base).all():
        print(f"residual.py: error: {a.oof} must hold {len(tr)} finite scores", file=sys.stderr)
        return 2
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

    fl = folds(y)
    pred = base.copy()
    rounds: list[int] = []
    for i_tr, i_va in fl:
        i_fit, i_es = train_test_split(i_tr, test_size=0.1, stratify=y[i_tr], random_state=0)
        d_fit = lgb.Dataset(X.iloc[i_fit], y[i_fit], init_score=base[i_fit])
        d_es = lgb.Dataset(X.iloc[i_es], y[i_es], init_score=base[i_es], reference=d_fit)
        b = lgb.train(
            params, d_fit, 3000, valid_sets=[d_es], callbacks=[lgb.early_stopping(200, verbose=False)]
        )
        pred[i_va] = base[i_va] + b.predict(X.iloc[i_va], num_iteration=b.best_iteration, raw_score=True)
        rounds.append(b.best_iteration)
    before, after = CVScore.of(y, base, fl), CVScore.of(y, pred, fl)
    print(f"  leaves {a.leaves} lr {a.lr} min_leaf {a.min_leaf}: rounds {rounds}")
    print(f"  stack     {before}")
    print(f"  + booster {after}  up {after.up(before)}/5")

    if a.sub is not None and a.out is not None:
        sub = pd.read_csv(a.sub)
        if not np.array_equal(sub["id"].to_numpy(), te["id"].to_numpy()):
            print(f"residual.py: error: {a.sub} ids differ from test.csv", file=sys.stderr)
            return 2
        t0 = logit(sub[TARGET].to_numpy(), eps=1e-12)
        n = max(1, round(float(np.mean(rounds))))
        b = lgb.train(params, lgb.Dataset(X, y, init_score=base), n)
        sub[TARGET] = 1 / (1 + np.exp(-(t0 + b.predict(Xt, raw_score=True))))
        sub.to_csv(a.out, index=False)
        print(f"wrote {a.out} ({n} rounds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
