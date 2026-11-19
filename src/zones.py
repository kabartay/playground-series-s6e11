"""Test the public "zero-fraud zones" on a stack's nested out-of-fold scores, and apply them to a submission.

A zone is a set of rows with no fraud in the training data; sending its test rows to the bottom of the
ranking can only help AUC if the zone also holds no fraud on unseen rows. The zones (public notebooks of
daoviet, goodpjw2008 and Weni Mohamed, 2026-11-18/19) were found on all training rows, so scoring them on
the same rows is optimistic. Here every merchant-risk edge is refitted on the four training folds (just
below the lowest fraud inside the zone's other conditions) and applied to the held-out fold:

  zone 1:  merchant_risk_score < 13.1
  zone 2:  velocity_score <= 4.1 and merchant_risk_score <= 20
  zone 34: all eight risk flags clear and (velocity_score <= 4.1 and merchant_risk_score <= 33.8,
           or velocity_score <= 6.0 and merchant_risk_score <= 25)

The velocity edges stay as published (chosen on all rows, a small optimism this test does not remove).
For the test set each edge is the lower of the published one and the one refitted on all training rows.

Usage (from src/):
  ../.venv/bin/python stack.py --no-hill --C 0.1 --oof-out X.npy --out X.csv <runs...>
  ../.venv/bin/python zones.py --oof X.npy --sub X.csv [--out X_zones.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import TARGET, folds, load

FLAGS_ZERO = ["cvv_retry_count", "prior_disputes"]
FLAGS_OFF = [
    "is_new_merchant",
    "is_foreign_transaction",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]
# (name, base condition before the merchant-risk edge, published merchant-risk edge, edge is strict "<")
ZONES = [
    ("zone 1", lambda d: np.ones(len(d), bool), 13.1, True),
    ("zone 2", lambda d: d["velocity_score"].to_numpy() <= 4.1, 20.0, False),
    ("zone 3", lambda d: clean(d) & (d["velocity_score"].to_numpy() <= 4.1), 33.8, False),
    ("zone 4", lambda d: clean(d) & (d["velocity_score"].to_numpy() <= 6.0), 25.0, False),
]


def clean(d: pd.DataFrame) -> np.ndarray:
    """Rows with all eight risk flags clear."""
    ok = np.ones(len(d), bool)
    for c in FLAGS_ZERO:
        ok &= d[c].to_numpy() == 0
    for c in FLAGS_OFF:
        ok &= d[c].to_numpy().astype(int) == 0
    return ok


def in_zone(d: pd.DataFrame, base, edge: float, strict: bool) -> np.ndarray:
    """Rows inside one zone for a given merchant-risk edge."""
    m = d["merchant_risk_score"].to_numpy()
    return base(d) & ((m < edge) if strict else (m <= edge))


def refit_edge(d: pd.DataFrame, y: np.ndarray, base, published: float, strict: bool) -> tuple[float, bool]:
    """Edge just below the lowest fraud inside the base condition, capped at the published edge.

    Returns the edge and whether it is strict.
    """
    m = d["merchant_risk_score"].to_numpy()
    fraud = base(d) & (y == 1)
    lowest = m[fraud].min() if fraud.any() else np.inf
    return (lowest, True) if lowest <= published else (published, strict)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", type=Path, required=True, help="nested OOF stack scores from stack.py --oof-out")
    ap.add_argument("--sub", type=Path, help="submission CSV of the same stack")
    ap.add_argument("--out", type=Path, help="write the submission with all zones sent to the bottom")
    a = ap.parse_args()

    tr, te, _ = load()
    y = tr[TARGET].to_numpy()
    pred = np.load(a.oof, allow_pickle=False)
    assert pred.shape == (len(tr),) and np.isfinite(pred).all()
    fl = folds(y)
    base = [roc_auc_score(y[i_va], pred[i_va]) for _, i_va in fl]
    print(f"  stack                     {roc_auc_score(y, pred):.5f}  folds {np.round(base, 5).tolist()}")

    for upto in range(1, len(ZONES) + 1):
        use = ZONES[:upto]
        z_pred = pred.copy()
        rows = frauds = 0
        for i_tr, i_va in fl:
            d_tr, d_va = tr.iloc[i_tr], tr.iloc[i_va]
            z = np.zeros(len(i_va), bool)
            for _, cond, pub, strict in use:
                z |= in_zone(d_va, cond, *refit_edge(d_tr, y[i_tr], cond, pub, strict))
            rows += z.sum()
            frauds += y[i_va][z].sum()
            z_pred[i_va[z]] = pred.min() - 1.0
        fa = [roc_auc_score(y[i_va], z_pred[i_va]) for _, i_va in fl]
        up = sum(f > b for f, b in zip(fa, base, strict=True))
        names = "+".join(n.split()[1] for n, *_ in use)
        print(
            f"  + zones {names:17s} {roc_auc_score(y, z_pred):.5f}  folds {np.round(fa, 5).tolist()}"
            f"  up {up}/5, {rows} held-out rows in zones, {frauds} frauds among them"
        )

    if a.sub is not None:
        sub = pd.read_csv(a.sub)
        assert (sub["id"].to_numpy() == te["id"].to_numpy()).all()
        z = np.zeros(len(te), bool)
        for name, cond, pub, strict in ZONES:
            edge, s = refit_edge(tr, y, cond, pub, strict)
            zt = in_zone(te, cond, edge, s)
            print(f"  test {name}: merchant_risk_score {'<' if s else '<='} {edge:g}, {zt.sum()} rows")
            z |= zt
        print(f"  test rows in any zone: {z.sum()} of {len(te)}")
        if a.out is not None:
            p = sub[TARGET].to_numpy().copy()
            p[z] *= 1e-6
            sub[TARGET] = p
            sub.to_csv(a.out, index=False)
            print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
