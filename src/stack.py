"""Stack saved runs: rank-average and a logistic regression over logit OOF predictions, scored by nested CV.

Usage:
  stack.py lgb_ogte_o1 xgb_ogte_o1 cat_ogte_o1 [--out submissions/stack1.csv]

The logistic stacker is evaluated with the same 5 folds as the members (fit on 4 folds of OOF, score
the 5th), so its CV is honest. Runs with NaN OOF (probes) are scored only on their non-NaN rows and
cannot be stacked into a submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import RUNS, TARGET, folds, load


def logit(p: np.ndarray) -> np.ndarray:
    """Clipped logit."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    tr, te, _ = load()
    y = tr[TARGET].to_numpy()
    oof = np.column_stack([np.load(RUNS / r / "oof.npy") for r in a.runs])
    test = np.column_stack([np.load(RUNS / r / "test.npy") for r in a.runs])
    m = ~np.isnan(oof).any(axis=1)
    print(f"rows scored: {m.sum()} of {len(y)}")
    for i, r in enumerate(a.runs):
        print(f"  {r:28s} {roc_auc_score(y[m], oof[m, i]):.5f}")

    rank = np.column_stack([rankdata(oof[m, i]) for i in range(oof.shape[1])]).mean(axis=1)
    print(f"  {'rank average':28s} {roc_auc_score(y[m], rank):.5f}")

    Z, Zt = logit(oof), logit(test)
    pred = np.full(len(y), np.nan)
    fold_auc = []
    for i_tr, i_va in folds(y):
        i_tr, i_va = i_tr[m[i_tr]], i_va[m[i_va]]
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z[i_tr], y[i_tr])
        pred[i_va] = lr.decision_function(Z[i_va])
        fold_auc.append(roc_auc_score(y[i_va], pred[i_va]))
    cv = roc_auc_score(y[m], pred[m])
    print(f"  {'logistic stack (nested)':28s} {cv:.5f}  folds {np.round(fold_auc, 5).tolist()}")
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z[m], y[m])
    print("  weights", dict(zip(a.runs, np.round(lr.coef_[0], 3).tolist(), strict=True)))

    if a.out:
        if not m.all() or np.isnan(test).any():
            raise SystemExit("cannot write a submission from probe runs with missing predictions")
        sub = pd.DataFrame({"id": te["id"], TARGET: 1 / (1 + np.exp(-lr.decision_function(Zt)))})
        a.out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(a.out, index=False)
        a.out.with_suffix(".json").write_text(
            json.dumps({"runs": a.runs, "cv": cv, "folds": fold_auc}, indent=2)
        )
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
