"""Time and score fintfm's LinearReadoutModel (0.13.0+, entry defaults) on the starter notebook's holdout.

Same split as notebooks/fintfm-starter: a stratified 20,000-row holdout of train.csv (random_state 0),
one-hot categoricals, float32. The readout is fitted on a random sample of the remaining rows and scores
the first --hold-rows holdout rows, so its AUC is comparable with the notebook's in-context settings
(best 0.9063, balanced context of 2,000 rows) and its LightGBM reference (0.9356).

Usage (from src/, after pip install fintfm==1.0.0):
  python probe_readout.py --fit-rows 20000 --hold-rows 20000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from common import THREADS, find_data


def main() -> None:
    """Fit the readout on a sample of the pool and print its holdout AUC, AP and throughput."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-rows", type=int, default=20_000)
    ap.add_argument("--hold-rows", type=int, default=20_000)
    ap.add_argument("--ensemble", type=int, default=0, help="override n_ensemble (0 keeps the entry's)")
    a = ap.parse_args()

    import torch

    torch.set_num_threads(THREADS)
    import fintfm
    from fintfm.readout import LinearReadoutModel, entry_defaults

    tr = pd.read_csv(find_data("train.csv"))
    y = tr.pop("is_fraud").to_numpy().astype(int)
    cats = [c for c in tr.columns if tr[c].dtype == object]
    X = pd.get_dummies(tr.drop(columns="id"), columns=cats, dtype=np.float32).astype(np.float32).to_numpy()
    pool, hold = train_test_split(np.arange(len(y)), test_size=20_000, stratify=y, random_state=0)
    fit_idx = np.random.default_rng(0).choice(pool, a.fit_rows, replace=False)
    hold_idx = hold[: a.hold_rows]

    e = entry_defaults("binary")
    ck = e.pop("checkpoint")
    if a.ensemble:
        e["n_ensemble"] = a.ensemble
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"fintfm {fintfm.__version__} on {dev}; n_ensemble {e['n_ensemble']}, context {e['max_context']}")
    t = time.time()
    m = LinearReadoutModel("binary", ck, device=dev, random_state=0, **e).fit(X[fit_idx], y[fit_idx])
    tf = time.time() - t
    t = time.time()
    p = m.predict_proba(X[hold_idx])[:, 1]
    tp = time.time() - t
    yh = y[hold_idx]
    print(
        f"readout fit on {a.fit_rows:,} rows in {tf:.0f} s; {len(yh):,} holdout rows ({yh.sum()} fraud) in "
        f"{tp:.0f} s, {len(yh) / tp:.1f} rows/s; AUC {roc_auc_score(yh, p):.4f}, "
        f"AP {average_precision_score(yh, p):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
