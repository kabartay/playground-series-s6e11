"""FinTFM zero-shot predictions on the shared folds, for stacking with the GBDTs.

FinTFM is a tabular foundation model (github.com/kabartay/fintfm). Here it is used zero-shot: each fold's
training rows (plus original rows if --orig) form the pool its context is drawn from, and it predicts the
validation rows. CPU throughput on Kaggle is about 20 rows/s, so --rows limits each validation fold to a
stratified subsample for cheap probes. Probe runs save OOF with NaN outside the scored rows.

Usage (on Kaggle, via scripts/kaggle_kernel.py, or locally with fintfm installed):
  train_fintfm.py --strategy balanced --folds 1 --rows 20000 --name tfm_bal_probe
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from common import RAW, RUNS, TARGET, THREADS, folds, load, save_run

FINTFM_COMMIT = "8a25b4c790f86feab97ae9e7be80299bf1cebc52"
CKPT_URL = (
    "https://huggingface.co/kabartay/fintfm-binary/resolve/"
    "f116bfd43a2b15c65ed3551ea8c38e3364629ddc/v4-cellattn-labels.pt"
)
CKPT_SHA256 = "d231b062483c9261aae4c503ad5359db295fc7206fa0183f51763e4953c518d2"


def ensure_fintfm() -> Path:
    """Install the pinned fintfm if missing and return the verified checkpoint path.

    A checkpoint whose SHA-256 differs from ``CKPT_SHA256`` (a truncated or altered download) is deleted
    and the script exits, so the next run downloads it again.
    """
    try:
        import fintfm  # noqa: F401
    except ImportError:
        url = f"fintfm @ git+https://github.com/kabartay/fintfm.git@{FINTFM_COMMIT}"
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", url], check=True)
    ckpt = RUNS.parent / "v4-cellattn-labels.pt"
    if not ckpt.exists():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CKPT_URL, ckpt)
    if hashlib.sha256(ckpt.read_bytes()).hexdigest() != CKPT_SHA256:
        ckpt.unlink()
        sys.exit(f"checkpoint hash mismatch: deleted {ckpt}; run again to download it afresh")
    return ckpt


def matrix(frames: list[pd.DataFrame]) -> list[np.ndarray]:
    """One-hot categoricals jointly so every frame gets the same float32 columns."""
    both = pd.concat([f[RAW] for f in frames], keys=range(len(frames)))
    both = pd.get_dummies(
        both, columns=[c for c in RAW if str(both[c].dtype) == "category"], dtype=np.float32
    )
    both = both.astype(np.float32)
    return [both.loc[i].to_numpy() for i in range(len(frames))]


def main() -> None:
    """Score the requested folds (and optionally test) zero-shot and save the run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="uniform", help="FinTFM context_strategy")
    ap.add_argument("--max-context", type=int, default=2000)
    ap.add_argument("--orig", type=int, default=1, help="1 adds the original rows to the context pool")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rows", type=int, default=0, help="score a stratified subsample of each fold (0 = all)")
    ap.add_argument("--test", type=int, default=0, help="1 also predicts test (context from all train)")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    ckpt = ensure_fintfm()
    from fintfm.inference.classifier import FinancialTFMClassifier

    tr, te, og = load()
    X, Xt, Xo = matrix([tr, te, og])
    y, yo = tr[TARGET].to_numpy(), og[TARGET].to_numpy()

    def model() -> FinancialTFMClassifier:
        return FinancialTFMClassifier(
            str(ckpt), device="cpu", max_context=a.max_context, context_strategy=a.strategy, random_state=0
        )

    import torch

    torch.set_num_threads(THREADS)
    oof = np.full(len(tr), np.nan)
    scores, t0 = [], time.time()
    for k, (i_tr, i_va) in enumerate(folds(y)[: a.folds]):
        if a.rows and a.rows < len(i_va):
            i_va, _ = train_test_split(i_va, train_size=a.rows, stratify=y[i_va], random_state=k)
        Xa, ya = X[i_tr], y[i_tr]
        if a.orig:
            Xa, ya = np.vstack([Xa, Xo]), np.concatenate([ya, yo])
        oof[i_va] = model().fit(Xa, ya).predict_proba(X[i_va])[:, 1]
        scores.append(roc_auc_score(y[i_va], oof[i_va]))
        print(f"fold {k} auc {scores[-1]:.5f} rows {len(i_va)} ({time.time() - t0:.0f}s)", flush=True)
    test = np.full(len(te), np.nan)
    if a.test:
        Xa, ya = (np.vstack([X, Xo]), np.concatenate([y, yo])) if a.orig else (X, y)
        test = model().fit(Xa, ya).predict_proba(Xt)[:, 1]
    m = ~np.isnan(oof)
    cv = roc_auc_score(y[m], oof[m])
    print(f"{a.name}: AUC {cv:.5f} on {m.sum()} rows, folds {np.round(scores, 5).tolist()}")
    save_run(
        a.name,
        oof,
        test,
        dict(
            vars(a),
            cv=cv,
            folds=scores,
            rows_scored=int(m.sum()),
            fintfm=FINTFM_COMMIT,
            seconds=round(time.time() - t0),
        ),
    )
    if Path("/kaggle/working").exists():
        ckpt.unlink()  # keep the 100+ MB checkpoint out of the kernel output


if __name__ == "__main__":
    main()
