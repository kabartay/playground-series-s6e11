"""Stack saved runs and score every stacker by nested CV.

Usage:
  stack.py lgb_ogte_o1 xgb_ogte_o1 cat_ogte_o1 [--method hill] [--out submissions/stack1.csv]

Stackers: rank average; logistic regression over logit OOF predictions at several C; greedy hill
climbing (forward selection with replacement on standardised logits, maximising AUC, then a one-feature
logistic calibration). Runs saved as logits (GLM, GLM boost) are detected and used as they are.
Each stacker is fitted on 4 of the members' 5 folds and scored on the 5th, so its CV is honest;
`--method` and `--C` choose the one that writes the submission. Runs with NaN OOF (probes) are scored
only on their non-NaN rows and cannot be stacked into a submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import RUNS, TARGET, folds, load


def logit(p: np.ndarray) -> np.ndarray:
    """Clipped logit."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def to_logit(oof: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Logit of a run's predictions; runs that already hold logits (GLM, GLM boost) pass through.

    A run is on the logit scale when any OOF value lies outside [0, 1].
    """
    v = oof[~np.isnan(oof)]
    if v.min() < 0 or v.max() > 1:
        return oof, test
    return logit(oof), logit(test)


def hill_climb(Z: np.ndarray, y: np.ndarray, steps: int = 40) -> np.ndarray:
    """Greedy forward selection with replacement maximising AUC; returns weights summing to 1.

    Z holds standardised logits (one column per member). Starts from the best single member and stops
    when no addition improves AUC by more than 1e-6.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--method", default="logistic", choices=["logistic", "hill"])
    ap.add_argument("--C", type=float, default=1.0, help="logistic stacker regularisation")
    ap.add_argument("--no-hill", action="store_true", help="skip hill climbing (slow on many members)")
    ap.add_argument("--oof-out", type=Path, help="save nested OOF stack scores (.npy)")
    a = ap.parse_args()

    tr, te, _ = load()
    y = tr[TARGET].to_numpy()
    oof = np.column_stack([np.load(RUNS / r / "oof.npy") for r in a.runs])
    test = np.column_stack([np.load(RUNS / r / "test.npy") for r in a.runs])
    m = ~np.isnan(oof).any(axis=1)
    print(f"rows scored: {m.sum()} of {len(y)}")
    for i, r in enumerate(a.runs):
        print(f"  {r:28s} {roc_auc_score(y[m], oof[m, i]):.5f}")

    rank = np.column_stack([rankdata(oof[m, i]) for i in range(oof.shape[1])]).mean(axis=1)
    print(f"  {'rank average':28s} {roc_auc_score(y[m], rank):.5f}")

    pairs = [to_logit(oof[:, i], test[:, i]) for i in range(oof.shape[1])]
    Z = np.column_stack([p[0] for p in pairs])
    Zt = np.column_stack([p[1] for p in pairs])
    fl = folds(y)

    preds: dict[str, np.ndarray] = {}

    def nested(fit_score, key: str) -> tuple[float, list[float]]:
        """Out-of-fold scores of a stacker: fit_score(train_idx, val_idx) returns val predictions."""
        pred = np.full(len(y), np.nan)
        fold_auc = []
        for i_tr, i_va in fl:
            i_tr, i_va = i_tr[m[i_tr]], i_va[m[i_va]]
            pred[i_va] = fit_score(i_tr, i_va)
            fold_auc.append(roc_auc_score(y[i_va], pred[i_va]))
        preds[key] = pred
        return roc_auc_score(y[m], pred[m]), fold_auc

    results = {}
    for C in sorted({0.01, 0.1, 1.0, a.C}):

        def lr_fit(i_tr, i_va, C=C):
            return LogisticRegression(C=C, max_iter=1000).fit(Z[i_tr], y[i_tr]).decision_function(Z[i_va])

        results[f"logistic C={C:g}"] = nested(lr_fit, f"logistic C={C:g}")
    if not a.no_hill:

        def hill_fit(i_tr, i_va):
            w = hill_climb(standardise(Z[i_tr], Z[i_tr]), y[i_tr])
            cal = calibrate(standardise(Z[i_tr], Z[i_tr]) @ w, y[i_tr])
            return cal.decision_function((standardise(Z[i_va], Z[i_tr]) @ w)[:, None])

        results["hill climbing"] = nested(hill_fit, "hill climbing")
    for k, (cv_k, fa) in results.items():
        print(f"  {k + ' (nested)':28s} {cv_k:.5f}  folds {np.round(fa, 5).tolist()}")

    lr = LogisticRegression(C=a.C, max_iter=1000).fit(Z[m], y[m])
    print("  logistic weights", dict(zip(a.runs, np.round(lr.coef_[0], 3).tolist(), strict=True)))
    if a.method == "hill":
        if a.no_hill:
            raise SystemExit("--method hill needs hill climbing")
        w = hill_climb(standardise(Z[m], Z[m]), y[m])
        print("  hill weights", dict(zip(a.runs, np.round(w, 3).tolist(), strict=True)))
        cv, fold_auc = results["hill climbing"]
        cal = calibrate(standardise(Z[m], Z[m]) @ w, y[m])
        final = cal.decision_function((standardise(Zt, Z[m]) @ w)[:, None])
    else:
        cv, fold_auc = results[f"logistic C={a.C:g}"]
        final = lr.decision_function(Zt)

    if a.oof_out:
        key = "hill climbing" if a.method == "hill" else f"logistic C={a.C:g}"
        np.save(a.oof_out, preds[key])
        print(f"wrote {a.oof_out}")
    if a.out:
        if not m.all() or np.isnan(test).any():
            raise SystemExit("cannot write a submission from probe runs with missing predictions")
        sub = pd.DataFrame({"id": te["id"], TARGET: 1 / (1 + np.exp(-final))})
        a.out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(a.out, index=False)
        a.out.with_suffix(".json").write_text(
            json.dumps({"runs": a.runs, "method": a.method, "C": a.C, "cv": cv, "folds": fold_auc}, indent=2)
        )
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
