"""FinTFM 1.0.0 linear readout as a stack member: out-of-fold scores on the shared folds, test scores.

The readout (`fintfm.readout.LinearReadoutModel`, entry defaults) is a pretrained in-context transformer
with a fitted linear head, a model family no other member shares. On the starter notebook's holdout it
scored 0.9296 (fitted on the whole pool). Inputs are the raw columns, one-hot categoricals, float32, as in
`probe_readout.py`.

Scoring runs at about 100-140 rows/s on Kaggle CPU, so the work is split over kernels: each part scores
some folds (`--folds 0,1`) and optionally the test set (`--test`, from one readout fitted on all training
rows) and writes `runs/<name>_<part>/`. `--merge` joins the parts locally into `runs/<name>/`.

Usage (from src/):
  python train_readout.py --name r20_fintfm_ro --part a --folds 0,1
  python train_readout.py --name r20_fintfm_ro --part c --folds 4 --test
  ../.venv/bin/python train_readout.py --name r20_fintfm_ro --merge a,b,c
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import RUNS, TARGET, THREADS, find_data, folds, save_run


def features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y, X_test): raw columns, one-hot categoricals over train and test, float32."""
    tr = pd.read_csv(find_data("train.csv"))
    te = pd.read_csv(find_data("test.csv"))
    y = tr.pop(TARGET).to_numpy().astype(int)
    both = pd.concat([tr, te], ignore_index=True).drop(columns="id")
    cats = [c for c in both.columns if not pd.api.types.is_numeric_dtype(both[c])]
    X = pd.get_dummies(both, columns=cats, dtype=np.float32).astype(np.float32).to_numpy()
    return X[: len(tr)], y, X[len(tr) :]


def fit_score(X: np.ndarray, y: np.ndarray, Xs: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit one readout on (X, y), score Xs; return (probabilities, fit seconds, score seconds)."""
    import torch
    from fintfm.readout import LinearReadoutModel, entry_defaults

    e = entry_defaults("binary")
    ck = e.pop("checkpoint")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = time.time()
    m = LinearReadoutModel("binary", ck, device=dev, random_state=0, **e).fit(X, y)
    tf = time.time() - t
    t = time.time()
    p = m.predict_proba(Xs)[:, 1]
    return p, tf, time.time() - t


def run_part(name: str, part: str, fold_ids: list[int], test: bool) -> None:
    """Score the given folds (and the test set) and write runs/<name>_<part>/."""
    import fintfm
    import torch

    torch.set_num_threads(THREADS)
    X, y, Xt = features()
    print(f"fintfm {fintfm.__version__}; train {X.shape}, test {Xt.shape}", flush=True)
    d = RUNS / f"{name}_{part}"
    d.mkdir(parents=True, exist_ok=True)
    aucs = []
    for k, (i_tr, i_va) in enumerate(folds(y)):
        if k not in fold_ids:
            continue
        p, tf, ts = fit_score(X[i_tr], y[i_tr], X[i_va])
        np.save(d / f"oof_f{k}.npy", p.astype(np.float64))
        aucs.append(roc_auc_score(y[i_va], p))
        print(f"fold {k}: AUC {aucs[-1]:.5f}; fit {tf:.0f} s, {len(i_va):,} rows in {ts:.0f} s", flush=True)
    if test:
        p, tf, ts = fit_score(X, y, Xt)
        np.save(d / "test.npy", p.astype(np.float64))
        print(f"test: fit {tf:.0f} s, {len(Xt):,} rows in {ts:.0f} s", flush=True)
    cv = float(np.mean(aucs)) if aucs else None
    meta = {"part": part, "fold_ids": fold_ids, "test": test, "folds": aucs, "cv": cv}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def merge(name: str, parts: list[str]) -> None:
    """Join the parts' fold and test scores into runs/<name>/{oof,test,meta}.

    Raises:
        ValueError: a fold is scored by two parts or by none, or the test set by more or fewer than one.
    """
    y = pd.read_csv(find_data("train.csv"), usecols=[TARGET])[TARGET].to_numpy().astype(int)
    oof = np.full(len(y), np.nan)
    test = None
    seen: dict[int, str] = {}
    for part in parts:
        d = RUNS / f"{name}_{part}"
        for k, (_, i_va) in enumerate(folds(y)):
            f = d / f"oof_f{k}.npy"
            if f.exists():
                if k in seen:
                    raise ValueError(f"fold {k} in parts {seen[k]} and {part}")
                seen[k] = part
                oof[i_va] = np.load(f, allow_pickle=False)
        if (d / "test.npy").exists():
            if test is not None:
                raise ValueError(f"more than one part scored the test set ({part})")
            test = np.load(d / "test.npy", allow_pickle=False)
    if not np.isfinite(oof).all():
        raise ValueError("a fold is missing")
    if test is None:
        raise ValueError("no part scored the test set")
    fa = [roc_auc_score(y[i_va], oof[i_va]) for _, i_va in folds(y)]
    model = "fintfm 1.0.0 LinearReadoutModel, entry defaults"
    meta = {"model": model, "cv": roc_auc_score(y, oof), "folds": fa}
    save_run(name, oof, test, meta)
    print(f"{name}: CV {meta['cv']:.5f} folds {np.round(fa, 5).tolist()}")


def main() -> None:
    """Score one part on this machine, or merge the parts' outputs."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--part", default="a")
    ap.add_argument("--folds", default="", help="comma-separated fold ids to score")
    ap.add_argument("--test", action="store_true", help="also fit on all rows and score the test set")
    ap.add_argument("--merge", help="comma-separated part names to join (local)")
    a = ap.parse_args()
    if a.merge:
        merge(a.name, a.merge.split(","))
    else:
        run_part(a.name, a.part, [int(k) for k in a.folds.split(",") if k], a.test)


if __name__ == "__main__":
    main()
