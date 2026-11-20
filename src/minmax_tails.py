"""MinMax tail selection on out-of-fold scores (D29, a null).

Where every voting member's rank fraction is above ``hi`` (below ``lo``), the row takes the members' maximum
(minimum) rank fraction; every other row keeps the base score. Zones 1-4 are applied with their edges refitted
inside each fold, as for the submissions. Nothing is fitted on labels, so the OOF score is honest.

Run from ``src/``::

    ../.venv/bin/python minmax_tails.py --base pub5_oof.npy ../runs/pub_fa_wide_lm_stack/oof.npy \
        --voters pub5_oof.npy ../runs/pub_fa_wide_lm_stack/oof.npy

``--base`` files are averaged as logit ranks (two files give the stack_pubwz score before zones); nested stack
OOFs come from ``stack.py --oof-out``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from common import TARGET, folds, load
from stack_ideas import hand_zones

THRESHOLDS: tuple[tuple[float, float], ...] = ((0.90, 0.10), (0.95, 0.05), (0.98, 0.02), (0.99, 0.01))


def rank_fraction(v: np.ndarray) -> np.ndarray:
    """Return ranks scaled to (0, 1): ``(rank - 0.5) / n``."""
    return (rankdata(v) - 0.5) / len(v)


def minmax(voters: np.ndarray, base: np.ndarray, hi: float, lo: float) -> tuple[np.ndarray, int, int]:
    """Apply MinMax selection.

    Args:
        voters: rank fractions, one column per voting member.
        base: rank fractions used for rows outside the agreed tails.
        hi: all voters above this -> the row takes the voters' maximum.
        lo: all voters below this -> the row takes the voters' minimum.

    Returns:
        The new scores and the numbers of rows in the top and bottom tails.
    """
    mn, mx = voters.min(axis=1), voters.max(axis=1)
    out = base.copy()
    top, bot = mn > hi, mx < lo
    out[top] = mx[top]
    out[bot] = mn[bot]
    return out, int(top.sum()), int(bot.sum())


def main() -> None:
    """Score every threshold pair against the base, fold by fold."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", type=Path, nargs="+", required=True, help="OOF files averaged as logit ranks")
    ap.add_argument("--voters", type=Path, nargs="+", required=True, help="OOF files that vote on the tails")
    a = ap.parse_args()

    tr, _, _ = load()
    y = tr[TARGET].to_numpy()
    fl = folds(y)
    zones = [hand_zones(tr, y, i_tr, i_va) for i_tr, i_va in fl]

    def score(p: np.ndarray) -> tuple[float, np.ndarray]:
        p = p.copy()
        low = p.min() - 1
        for (_, i_va), z in zip(fl, zones, strict=True):
            p[i_va[z]] = low
        return roc_auc_score(y, p), np.array([roc_auc_score(y[i_va], p[i_va]) for _, i_va in fl])

    load_rf = lambda f: rank_fraction(np.load(f, allow_pickle=False))  # noqa: E731
    base = rank_fraction(np.mean([logit(load_rf(f)) for f in a.base], axis=0))
    voters = np.column_stack([load_rf(f) for f in a.voters])
    ref_cv, ref_f = score(base)
    print(f"base + zones {ref_cv:.5f} {np.round(ref_f, 5).tolist()}")
    for hi, lo in THRESHOLDS:
        p, n_top, n_bot = minmax(voters, base, hi, lo)
        cv, f = score(p)
        gain = 1e5 * (cv - ref_cv)
        print(f"{hi:.2f}/{lo:.2f}  top {n_top:6d} bottom {n_bot:6d}  {cv:.5f}  {gain:+6.1f} pts  "
              f"up {int((f > ref_f).sum())} of 5", flush=True)


if __name__ == "__main__":
    main()
