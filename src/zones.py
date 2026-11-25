"""Zero-fraud zones (D27): test them on a stack's nested out-of-fold scores and apply them to a submission.

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

Other scripts reuse :func:`hand_zones` (zone rows of a held-out fold) and :func:`send_to_bottom`.

Usage (from src/):
  ../.venv/bin/python stack.py --no-hill --C 0.1 --oof-out X.npy --out X.csv <runs...>
  ../.venv/bin/python zones.py --oof X.npy --sub X.csv [--out X_zones.csv]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common import TARGET, CVScore, Fold, folds, load

FLAGS_ZERO = ["cvv_retry_count", "prior_disputes"]
FLAGS_OFF = [
    "is_new_merchant",
    "is_foreign_transaction",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]


def clean(d: pd.DataFrame) -> np.ndarray:
    """Rows with all eight risk flags clear."""
    ok = np.ones(len(d), bool)
    for c in FLAGS_ZERO:
        ok &= d[c].to_numpy() == 0
    for c in FLAGS_OFF:
        ok &= d[c].to_numpy().astype(int) == 0
    return ok


@dataclass(frozen=True)
class Zone:
    """A base condition plus an upper edge on merchant_risk_score.

    Attributes:
        name: printed name.
        base: rows satisfying the conditions other than the merchant-risk edge.
        published: the edge as published.
        strict: the published edge is ``<`` rather than ``<=``.
    """

    name: str
    base: Callable[[pd.DataFrame], np.ndarray]
    published: float
    strict: bool

    def refit(self, d: pd.DataFrame, y: np.ndarray) -> tuple[float, bool]:
        """Edge just below the lowest fraud inside the base condition, capped at the published edge.

        Returns the edge and whether it is strict.
        """
        m = d["merchant_risk_score"].to_numpy()
        fraud = self.base(d) & (y == 1)
        lowest = m[fraud].min() if fraud.any() else np.inf
        return (lowest, True) if lowest <= self.published else (self.published, self.strict)

    def contains(self, d: pd.DataFrame, edge: float, strict: bool) -> np.ndarray:
        """Rows inside the zone for a given merchant-risk edge."""
        m = d["merchant_risk_score"].to_numpy()
        return self.base(d) & ((m < edge) if strict else (m <= edge))


ZONES: tuple[Zone, ...] = (
    Zone("zone 1", lambda d: np.ones(len(d), bool), 13.1, True),
    Zone("zone 2", lambda d: d["velocity_score"].to_numpy() <= 4.1, 20.0, False),
    Zone("zone 3", lambda d: clean(d) & (d["velocity_score"].to_numpy() <= 4.1), 33.8, False),
    Zone("zone 4", lambda d: clean(d) & (d["velocity_score"].to_numpy() <= 6.0), 25.0, False),
)


def hand_zones(
    tr: pd.DataFrame, y: np.ndarray, i_tr: np.ndarray, i_va: np.ndarray, zones: Sequence[Zone] = ZONES
) -> np.ndarray:
    """Mask over the held-out rows ``i_va`` of those inside any zone, edges refitted on rows ``i_tr``."""
    d_tr, d_va = tr.iloc[i_tr], tr.iloc[i_va]
    z = np.zeros(len(i_va), bool)
    for zone in zones:
        z |= zone.contains(d_va, *zone.refit(d_tr, y[i_tr]))
    return z


def send_to_bottom(pred: np.ndarray, fl: Sequence[Fold], masks: Sequence[np.ndarray]) -> np.ndarray:
    """Copy of out-of-fold scores with each fold's masked validation rows set below every score."""
    out = pred.copy()
    low = pred.min() - 1.0
    for (_, i_va), z in zip(fl, masks, strict=True):
        out[i_va[z]] = low
    return out


def zones_on_test(tr: pd.DataFrame, y: np.ndarray, te: pd.DataFrame) -> np.ndarray:
    """Mask of test rows inside any zone, each edge the lower of the published and the refitted one."""
    z = np.zeros(len(te), bool)
    for zone in ZONES:
        edge, s = zone.refit(tr, y)
        zt = zone.contains(te, edge, s)
        print(f"  test {zone.name}: merchant_risk_score {'<' if s else '<='} {edge:g}, {zt.sum()} rows")
        z |= zt
    print(f"  test rows in any zone: {z.sum()} of {len(te)}")
    return z


def main(argv: list[str] | None = None) -> int:
    """Score zones 1, 1+2, 1+2+3 and all four on the OOF scores; apply all four to the submission."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", type=Path, required=True, help="nested OOF stack scores from stack.py --oof-out")
    ap.add_argument("--sub", type=Path, help="submission CSV of the same stack")
    ap.add_argument("--out", type=Path, help="write the submission with all zones sent to the bottom")
    a = ap.parse_args(argv)

    tr, te, _ = load()
    y = tr[TARGET].to_numpy()
    pred = np.load(a.oof, allow_pickle=False)
    if pred.shape != (len(tr),) or not np.isfinite(pred).all():
        print(f"zones.py: error: {a.oof} must hold {len(tr)} finite scores", file=sys.stderr)
        return 2
    fl = folds(y)
    base = CVScore.of(y, pred, fl)
    print(f"  stack                     {base}")

    for upto in range(1, len(ZONES) + 1):
        use = ZONES[:upto]
        masks = [hand_zones(tr, y, i_tr, i_va, use) for i_tr, i_va in fl]
        rows = sum(int(z.sum()) for z in masks)
        frauds = sum(int(y[i_va][z].sum()) for (_, i_va), z in zip(fl, masks, strict=True))
        score = CVScore.of(y, send_to_bottom(pred, fl, masks), fl)
        names = "+".join(zone.name.split()[1] for zone in use)
        print(
            f"  + zones {names:17s} {score}"
            f"  up {score.up(base)}/5, {rows} held-out rows in zones, {frauds} frauds among them"
        )

    if a.sub is not None:
        sub = pd.read_csv(a.sub)
        if not np.array_equal(sub["id"].to_numpy(), te["id"].to_numpy()):
            print(f"zones.py: error: {a.sub} ids differ from test.csv", file=sys.stderr)
            return 2
        z = zones_on_test(tr, y, te)
        if a.out is not None:
            p = sub[TARGET].to_numpy().copy()
            p[z] *= 1e-6
            sub[TARGET] = p
            sub.to_csv(a.out, index=False)
            print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
