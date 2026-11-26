"""D35: what does a blend fitted to the public board buy on the private board? Simulated on labelled rows.

The top of the public board is built by "board models": blends of many scored files whose weights are chosen
to raise the public score, measured on a fixed 20% of the test rows (about 60,000 rows, 1,000 frauds). Their
private effect is unknown until the close. Here the same situation is rebuilt on the training rows, whose
labels we have, with the out-of-fold predictions of every member in ``runs/`` playing the scored files:

* a random 60,000 training rows play the public part (P), the other 640,000 the private part (Q);
* the honest file is the base (``--base``, e.g. the unzoned ``stack_pub4wz`` OOF), chosen without P's labels;
* the board file is a logistic blend of the base and all members (logit ranks, standardised), fitted on P
  only, at several regularisation strengths; ``continuation`` adds 0.45 x (board - base) on top, as the
  extrapolation notebooks of 2026-11-26 do;
* every file is scored on P (what the board shows) and on Q (what the private board would show).

Fitting on P's labels is an upper bound on what 244 public scores can reveal, so the public gains here are
larger than a real board model's; the question is the sign and size of the private effect. Repeated over
``--reps`` random splits. No file is written.

    ../.venv/bin/python research_board_sim.py --base ../runs/_d35_avg4_oof.npy --reps 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import RUNS, TARGET, load, logit_rank

N_PUBLIC = 60_000
CS = (0.001, 0.01, 0.1, 1.0)
ALPHA = 0.45


def member_matrix() -> tuple[list[str], np.ndarray]:
    """Logit ranks of every run in runs/ with a complete OOF vector (rows x members, float32)."""
    names, cols = [], []
    for d in sorted(RUNS.iterdir()):
        f = d / "oof.npy"
        if not f.exists():
            continue
        v = np.load(f, allow_pickle=False)
        if v.ndim == 1 and np.isfinite(v).all():
            names.append(d.name)
            cols.append(logit_rank(v).astype(np.float32))
    return names, np.column_stack(cols)


def main() -> None:
    """Print, per board file, its mean gain over the base on P and on Q, with SDs over the splits."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", type=Path, required=True, help="OOF scores of the honest file")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    tr, _, _ = load()
    y = tr[TARGET].to_numpy()
    base = logit_rank(np.load(a.base, allow_pickle=False))
    _, M = member_matrix()
    X = np.column_stack([base, M])
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    print(f"{X.shape[1] - 1} members + base; P = {N_PUBLIC:,} random rows, Q = the rest; {a.reps} splits")

    rows: dict[str, list[tuple[float, float]]] = {}
    noise = []
    for rep in range(a.reps):
        rng = np.random.default_rng(rep)
        p = np.zeros(len(y), bool)
        p[rng.choice(len(y), N_PUBLIC, replace=False)] = True
        q = ~p
        b_p, b_q = roc_auc_score(y[p], base[p]), roc_auc_score(y[q], base[q])
        noise.append(b_p - b_q)
        for C in CS:
            lr = LogisticRegression(C=C, max_iter=2000).fit(X[p], y[p])
            board = logit_rank(lr.decision_function(X))
            cont = board + ALPHA * (board - base)
            for name, s in ((f"board blend C={C:g}", board), (f"  + continuation C={C:g}", cont)):
                rows.setdefault(name, []).append(
                    (roc_auc_score(y[p], s[p]) - b_p, roc_auc_score(y[q], s[q]) - b_q)
                )
        print(f"split {rep}: base P {b_p:.5f} Q {b_q:.5f}", flush=True)

    print(
        f"\nbase: P minus Q AUC SD {1e5 * np.std(noise):.1f} pts over splits"
        " (the public/private shake-up of an honest file)"
    )
    print(f"{'file':28s} {'gain on P (public)':>22s} {'gain on Q (private)':>22s}  Q up")
    for name, v in rows.items():
        d = np.array(v) * 1e5
        print(
            f"{name:28s} {d[:, 0].mean():+9.1f} (SD {d[:, 0].std():5.1f}) pts "
            f"{d[:, 1].mean():+9.1f} (SD {d[:, 1].std():5.1f}) pts  {int((d[:, 1] > 0).sum())}/{len(d)}"
        )


if __name__ == "__main__":
    main()
