"""Fit-free mean of independently built nested stacks on logit-ranks, then zones (D31).

Four stacks, each nested on the shared folds: ``stack_pub5`` (ours), goodpjw2008's wide LM stack,
DavutKursun's "0.9526 Wall" stack and dhananjayk2609's wide stack (both Kaggle notebook outputs, Apache-2.0).
Nothing is fitted here: each stack's OOF and test scores become logit-ranks and are averaged with equal
weights. The script checks the shared folds and each stack's per-fold AUCs against the figures its author
printed, prints the nested CV with zones 1-4 refitted per fold, and writes the unzoned OOF and test scores
for ``zones.py``.

Run from ``src/`` (after ``stack.py ... --oof-out``, which writes the ``stack_pub5`` OOF)::

    ../.venv/bin/python stack_average.py --pub5-oof ../runs/stack_pub5_oof.npy --out-oof X.npy --out-sub X.csv
    ../.venv/bin/python zones.py --oof X.npy --sub X.csv --out ../submissions/stack_pub4wz.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from common import DATA, TARGET, folds, load
from stack_ideas import hand_zones

PUB = DATA / "public"
DK = PUB / "davutkursun__s6e11-the-0-9526-wall-who-still-adds-signal"
DH = PUB / "dhananjayk2609__s6e11-fraud-part-3-final-blend-lb-0-953" / "oof"
# Per-fold nested AUCs as printed in each notebook's log (2026-11-21 versions).
DK_FOLDS = [0.95168, 0.95254, 0.95764, 0.95272, 0.94847]
DH_FOLDS = [0.95182, 0.95261, 0.95737, 0.95279, 0.94838]


def logit_rank(v: np.ndarray) -> np.ndarray:
    """Logit of the mid-rank fraction, so every stack enters on the same scale."""
    return logit((rankdata(v) - 0.5) / len(v))


def main() -> None:
    """Check the inputs, print the nested CV of the mean with zones, and write the unzoned scores."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pub5-oof", type=Path, required=True,
                    help="nested OOF of stack_pub5 (stack.py --oof-out)")
    ap.add_argument("--pub5-sub", type=Path, default=Path("../submissions/stack_pub5.csv"))
    ap.add_argument("--out-oof", type=Path, required=True)
    ap.add_argument("--out-sub", type=Path, required=True)
    a = ap.parse_args()

    tr, te, _ = load()
    y = tr[TARGET].to_numpy()
    fl = folds(y)
    fold_id = np.zeros(len(y), int)
    for k, (_, i_va) in enumerate(fl):
        fold_id[i_va] = k

    dk_o, dk_t = pd.read_csv(DK / "oof_wall_stack.csv"), pd.read_csv(DK / "test_wall_stack.csv")
    p5 = pd.read_csv(a.pub5_sub)
    if not ((dk_o.id.to_numpy() == tr.id.to_numpy()).all() and (dk_o.fold.to_numpy() == fold_id).all()):
        raise SystemExit("DavutKursun's stack: ids or folds differ from ours")
    if not ((dk_t.id.to_numpy() == te.id.to_numpy()).all() and (p5.id.to_numpy() == te.id.to_numpy()).all()):
        raise SystemExit("test ids differ")
    oof = {
        "stack_pub5": np.load(a.pub5_oof, allow_pickle=False),
        "wide LM stack": np.load(DATA.parent / "runs/pub_fa_wide_lm_stack/oof.npy", allow_pickle=False),
        "DavutKursun wall": dk_o.stack_all.to_numpy(),
        "dhananjayk2609 wide": np.load(DH / "18_wide_stack_oof.npy", allow_pickle=False),
    }
    test = {
        "stack_pub5": p5.is_fraud.to_numpy(),
        "wide LM stack": np.load(DATA.parent / "runs/pub_fa_wide_lm_stack/test.npy", allow_pickle=False),
        "DavutKursun wall": dk_t.stack_all.to_numpy(),
        "dhananjayk2609 wide": np.load(DH / "18_wide_stack_test.npy", allow_pickle=False),
    }
    for name, v in oof.items():
        fa = [roc_auc_score(y[i_va], v[i_va]) for _, i_va in fl]
        print(f"{name:22s} OOF {roc_auc_score(y, v):.5f}  folds {np.round(fa, 5).tolist()}")
        ref = {"DavutKursun wall": DK_FOLDS, "dhananjayk2609 wide": DH_FOLDS}.get(name)
        if ref and max(abs(f - r) for f, r in zip(fa, ref, strict=True)) > 1e-4:
            raise SystemExit(f"{name}: per-fold AUCs differ from the author's log")
        if len(test[name]) != len(te) or not np.isfinite(test[name]).all():
            raise SystemExit(f"{name}: bad test scores")

    mean_oof = sum(logit_rank(v) for v in oof.values()) / len(oof)
    mean_test = sum(logit_rank(v) for v in test.values()) / len(test)
    p = mean_oof.copy()
    low = p.min() - 1
    for i_tr, i_va in fl:
        p[i_va[hand_zones(tr, y, i_tr, i_va)]] = low
    fa = [roc_auc_score(y[i_va], p[i_va]) for _, i_va in fl]
    print(f"{'mean of 4 + zones':22s} OOF {roc_auc_score(y, p):.5f}  folds {np.round(fa, 5).tolist()}")
    np.save(a.out_oof, mean_oof)
    pd.DataFrame({"id": te.id, "is_fraud": expit(mean_test)}).to_csv(a.out_sub, index=False)


if __name__ == "__main__":
    main()
