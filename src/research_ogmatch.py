"""Best-matching original row (D29, idea 4): does a row's closest original row carry signal?

The generator reuses the original's values but recombines them into new rows (D2). Pairs of values copied
from one original row were measured in D14; this looks at more columns at once. For each train and test
row, over the six high-cardinality columns (MATCH_COLS) it finds the original rows sharing the most exact
values, and records:

  ogm_max    how many of the six columns the best original row shares (0-6)
  ogm_n      log1p of how many original rows reach that maximum
  ogm_fraud  fraud rate among those original rows (base rate where ogm_max is 0)

The original labels are external to train, so no fold handling is needed for these features. The script
prints fraud rates by ogm_max, the features' own AUC, and a nested test: the logistic stacker of a final
stack with and without the three features (standardised) as extra inputs.

Usage (from src/):
  ../.venv/bin/python research_ogmatch.py ../submissions/stack_r12z.json [--save]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import RUNS, TARGET, folds, load
from stack import to_logit

MATCH_COLS = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
]


def indicator(df: pd.DataFrame, vocab: list[dict[float, int]], offsets: np.ndarray) -> sp.csr_matrix:
    """Sparse 0/1 matrix, one column per (column, original value); values not in the original drop out."""
    rows, cols = [], []
    for j, c in enumerate(MATCH_COLS):
        k = df[c].map(vocab[j]).to_numpy()
        ok = ~np.isnan(k)
        rows.append(np.flatnonzero(ok))
        cols.append(k[ok].astype(np.int64) + offsets[j])
    r, c = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(df), int(offsets[-1])))


def match_features(df: pd.DataFrame, og: pd.DataFrame, chunk: int = 50_000) -> pd.DataFrame:
    """ogm_max, ogm_n and ogm_fraud for every row of df against the original rows."""
    vocab = [{v: i for i, v in enumerate(np.unique(og[c].to_numpy()))} for c in MATCH_COLS]
    offsets = np.r_[0, np.cumsum([len(v) for v in vocab])]
    B = indicator(og, vocab, offsets).T.tocsr()
    lab = og[TARGET].to_numpy().astype(np.float32)
    base = float(lab.mean())
    A = indicator(df, vocab, offsets)
    mx, n, fr = (np.zeros(len(df), np.float32) for _ in range(3))
    for s in range(0, len(df), chunk):
        M = (A[s : s + chunk] @ B).tocsr()
        for i in range(M.shape[0]):
            a, b = M.indptr[i], M.indptr[i + 1]
            if a == b:
                fr[s + i] = base
                continue
            v = M.data[a:b]
            top = v == v.max()
            ids = M.indices[a:b][top]
            mx[s + i], n[s + i], fr[s + i] = v.max(), np.log1p(top.sum()), lab[ids].mean()
    return pd.DataFrame({"ogm_max": mx, "ogm_n": n, "ogm_fraud": fr}, index=df.index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", type=Path, help="submission JSON with runs and C")
    ap.add_argument("--save", action="store_true", help="write runs/_ogmatch_{train,test}.npy for reuse")
    a = ap.parse_args()
    tr, te, og = load()
    y = tr[TARGET].to_numpy()
    f_tr = match_features(tr, og)
    print(f"base fraud rate {y.mean():.4%}")
    for k, g in f_tr.groupby("ogm_max"):
        print(f"  ogm_max {int(k)}: {len(g):7d} rows, fraud {y[g.index].mean():.4%}")
    for c in f_tr.columns:
        print(f"  AUC of {c}: {roc_auc_score(y, f_tr[c]):.5f}")
    if a.save:
        np.save(RUNS / "_ogmatch_train.npy", f_tr.to_numpy())
        np.save(RUNS / "_ogmatch_test.npy", match_features(te, og).to_numpy())

    spec = json.loads(a.stack.read_text())
    Z = np.column_stack([to_logit(np.load(RUNS / r / "oof.npy"), np.zeros(1))[0] for r in spec["runs"]])
    F = f_tr.to_numpy()
    F = (F - F.mean(axis=0)) / F.std(axis=0)
    fl = folds(y)
    for name, X in (("stack", Z), ("stack + ogm features", np.hstack([Z, F]))):
        pred = np.zeros(len(y))
        for i_tr, i_va in fl:
            lr = LogisticRegression(C=spec["C"], max_iter=1000).fit(X[i_tr], y[i_tr])
            pred[i_va] = lr.decision_function(X[i_va])
        fa = [roc_auc_score(y[i_va], pred[i_va]) for _, i_va in fl]
        print(f"  {name:22s} {roc_auc_score(y, pred):.5f}  folds {np.round(fa, 5).tolist()}", flush=True)


if __name__ == "__main__":
    main()
