r"""Stack saved runs and score every stacker by nested CV; optionally write a submission.

    ../.venv/bin/python stack.py RUN [RUN ...] [--method {logistic,hill}] [--C 0.01] [--no-hill] \
        [--oof-out X.npy] [--out ../submissions/X.csv]

Stackers, each fitted on four of the members' five folds and scored on the fifth, so its CV is honest:

* rank average (fit-free; printed for reference);
* logistic regression over member logits at C 0.01, 0.1, 1 and ``--C``;
* greedy hill climbing: forward selection with replacement on standardised logits, maximising AUC, then
  a one-feature logistic calibration (skip with ``--no-hill``; slow on many members).

Runs saved as logits (GLM, GLM boost) are detected and used as they are (:func:`to_logit`). Runs with NaN
OOF (probes) are scored only on their non-NaN rows and cannot be stacked into a submission. ``--method`` and
``--C`` choose the stacker that writes the submission (fitted on all rows) and ``--oof-out`` (its nested
out-of-fold scores, the input of ``zones.py``, ``stack_average.py`` and ``residual.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import TARGET, CVScore, Fold, folds, load, load_run, logit

FitScore = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""A stacker: fit on the training-row positions, return scores for the validation-row positions."""


def to_logit(oof: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Logit of a run's predictions; runs that already hold logits (GLM, GLM boost) pass through.

    A run is on the logit scale when any OOF value lies outside [0, 1]; OOF and test always share a scale.
    """
    v = oof[~np.isnan(oof)]
    if v.min() < 0 or v.max() > 1:
        return oof, test
    return logit(oof), logit(test)


def member_logits(loaded: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Stack (OOF, test) score pairs as logits: (OOF rows x members, test rows x members)."""
    pairs = [to_logit(o, t) for o, t in loaded]
    return np.column_stack([p[0] for p in pairs]), np.column_stack([p[1] for p in pairs])


def load_members(runs: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Member logits of saved runs (see :func:`member_logits`)."""
    return member_logits([load_run(r) for r in runs])


def nested(
    fl: Sequence[Fold], y: np.ndarray, fit_score: FitScore, rows: np.ndarray | None = None
) -> tuple[np.ndarray, CVScore]:
    """Out-of-fold predictions of a stacker and their score.

    Args:
        fl: the folds.
        y: labels.
        fit_score: the stacker (see :data:`FitScore`).
        rows: optional mask of usable rows; the others are left NaN and unscored.
    """
    pred = np.full(len(y), np.nan)
    for i_tr, i_va in fl:
        if rows is not None:
            i_tr, i_va = i_tr[rows[i_tr]], i_va[rows[i_va]]
        pred[i_va] = fit_score(i_tr, i_va)
    return pred, CVScore.of(y, pred, fl, rows)


def hill_climb(Z: np.ndarray, y: np.ndarray, steps: int = 40) -> np.ndarray:
    """Greedy forward selection with replacement maximising AUC; returns weights summing to 1.

    Z holds standardised logits (one column per member). Starts from the best single member and stops
    when no addition improves AUC by more than 1e-6, or after ``steps`` additions.
    """
    counts = np.zeros(Z.shape[1])
    aucs = [roc_auc_score(y, Z[:, j]) for j in range(Z.shape[1])]
    counts[int(np.argmax(aucs))] = 1
    cur, best = Z[:, int(np.argmax(aucs))].copy(), max(aucs)
    for _ in range(steps):
        trial = [roc_auc_score(y, cur + Z[:, j]) for j in range(Z.shape[1])]
        j = int(np.argmax(trial))
        if trial[j] <= best + 1e-6:
            break
        cur, best = cur + Z[:, j], trial[j]
        counts[j] += 1
    return counts / counts.sum()


def calibrate(score: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """One-feature logistic fit, so blends fitted on different folds share one scale when pooled."""
    return LogisticRegression(C=1e6, max_iter=1000).fit(score[:, None], y)


def standardise(Z: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Scale each column of Z by the mean and SD of the same column in ref."""
    return (Z - ref.mean(axis=0)) / ref.std(axis=0)


def logistic_stacker(Z: np.ndarray, y: np.ndarray, C: float) -> FitScore:
    """L2 logistic regression on member logits at regularisation ``C``."""

    def fit_score(i_tr: np.ndarray, i_va: np.ndarray) -> np.ndarray:
        return LogisticRegression(C=C, max_iter=1000).fit(Z[i_tr], y[i_tr]).decision_function(Z[i_va])

    return fit_score


def hill_stacker(Z: np.ndarray, y: np.ndarray) -> FitScore:
    """Hill climbing on standardised logits, calibrated on the training rows (:func:`hill_climb`)."""

    def fit_score(i_tr: np.ndarray, i_va: np.ndarray) -> np.ndarray:
        w = hill_climb(standardise(Z[i_tr], Z[i_tr]), y[i_tr])
        cal = calibrate(standardise(Z[i_tr], Z[i_tr]) @ w, y[i_tr])
        return cal.decision_function((standardise(Z[i_va], Z[i_tr]) @ w)[:, None])

    return fit_score


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line (``argv`` defaults to ``sys.argv[1:]``)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run names under runs/")
    ap.add_argument("--out", type=Path, help="submission CSV to write (plus a JSON with the spec)")
    ap.add_argument("--method", default="logistic", choices=["logistic", "hill"])
    ap.add_argument("--C", type=float, default=1.0, help="logistic stacker regularisation")
    ap.add_argument("--no-hill", action="store_true", help="skip hill climbing (slow on many members)")
    ap.add_argument("--oof-out", type=Path, help="save nested OOF stack scores (.npy)")
    a = ap.parse_args(argv)
    if a.method == "hill" and a.no_hill:
        ap.error("--method hill needs hill climbing (drop --no-hill)")
    return a


def main(argv: list[str] | None = None) -> int:
    """Score every stacker by nested CV and write the chosen one's outputs; return the exit status."""
    a = parse_args(argv)
    tr, te, _ = load()
    y = tr[TARGET].to_numpy()
    try:
        loaded = [load_run(r) for r in a.runs]
    except FileNotFoundError as e:
        print(f"stack.py: error: {e}", file=sys.stderr)
        return 2
    oof = np.column_stack([o for o, _ in loaded])
    Z, Zt = member_logits(loaded)
    m = ~np.isnan(oof).any(axis=1)
    print(f"rows scored: {m.sum()} of {len(y)}")
    for i, r in enumerate(a.runs):
        print(f"  {r:28s} {roc_auc_score(y[m], oof[m, i]):.5f}")

    rank = np.column_stack([rankdata(oof[m, i]) for i in range(oof.shape[1])]).mean(axis=1)
    print(f"  {'rank average':28s} {roc_auc_score(y[m], rank):.5f}")

    fl = folds(y)
    stackers: dict[str, FitScore] = {
        f"logistic C={C:g}": logistic_stacker(Z, y, C) for C in sorted({0.01, 0.1, 1.0, a.C})
    }
    if not a.no_hill:
        stackers["hill climbing"] = hill_stacker(Z, y)
    preds: dict[str, np.ndarray] = {}
    results: dict[str, CVScore] = {}
    for key, fit_score in stackers.items():
        preds[key], results[key] = nested(fl, y, fit_score, m)
        print(f"  {key + ' (nested)':28s} {results[key]}")

    lr = LogisticRegression(C=a.C, max_iter=1000).fit(Z[m], y[m])
    print("  logistic weights", dict(zip(a.runs, np.round(lr.coef_[0], 3).tolist(), strict=True)))
    if a.method == "hill":
        key = "hill climbing"
        w = hill_climb(standardise(Z[m], Z[m]), y[m])
        print("  hill weights", dict(zip(a.runs, np.round(w, 3).tolist(), strict=True)))
        cal = calibrate(standardise(Z[m], Z[m]) @ w, y[m])
        final = cal.decision_function((standardise(Zt, Z[m]) @ w)[:, None])
    else:
        key = f"logistic C={a.C:g}"
        final = lr.decision_function(Zt)

    if a.oof_out:
        np.save(a.oof_out, preds[key])
        print(f"wrote {a.oof_out}")
    if a.out:
        if not m.all() or np.isnan(Zt).any():
            print(
                "stack.py: error: cannot write a submission from probe runs with missing predictions",
                file=sys.stderr,
            )
            return 2
        sub = pd.DataFrame({"id": te["id"], TARGET: 1 / (1 + np.exp(-final))})
        a.out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(a.out, index=False)
        spec = {
            "runs": a.runs,
            "method": a.method,
            "C": a.C,
            "cv": results[key].cv,
            "folds": list(results[key].folds),
        }
        a.out.with_suffix(".json").write_text(json.dumps(spec, indent=2))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
