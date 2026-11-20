"""Three stacker-side ideas (D29), each scored by nested CV against the stack it would replace.

1. Pairwise stacker: a logistic regression on differences of member logits between a fraud row and a
   non-fraud row (a smooth surrogate of AUC; RankNet with a linear scorer). The usual stacker fits
   log-loss, but the metric only rewards ranking.
2. Segment stacker: the logistic stacker plus member-logit x segment interactions, so members can weigh
   differently in different regions (merchant-risk bands, velocity bands, or both).
3. Zone search: a decision tree on the raw columns, fitted on the four training folds; its leaves with no
   fraud there become new zones and are sent to the bottom on the held-out fold, on top of the four hand
   zones (D27, edges refitted in each fold as in zones.py).

4. Meta stacker: member logits rank-gaussianised over all rows (no labels), and/or the logistic stacker
   plus member-disagreement columns (mean, SD and range of member logits), standardised.

Members and C come from a stack's submission JSON. Every fit sees only the four training folds.

Usage (from src/):
  ../.venv/bin/python stack_ideas.py ../submissions/stack_r12z.json [--ideas pair,segment,zones]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from common import BOOL, CAT_STR, NUM, RUNS, TARGET, folds, load
from stack import to_logit
from zones import ZONES, in_zone, refit_edge

Fold = tuple[np.ndarray, np.ndarray]


def nested(fl: list[Fold], y: np.ndarray, fit_score: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    """Out-of-fold predictions of fit_score(train_idx, val_idx); returns (pred, cv, fold AUCs)."""
    pred = np.full(len(y), np.nan)
    for i_tr, i_va in fl:
        pred[i_va] = fit_score(i_tr, i_va)
    fa = [roc_auc_score(y[i_va], pred[i_va]) for _, i_va in fl]
    return pred, roc_auc_score(y, pred), fa


def report(name: str, cv: float, fa: list[float], base: list[float] | None = None) -> None:
    """Print one result line, with the folds-up count against base."""
    up = "" if base is None else f"  up {sum(f > b for f, b in zip(fa, base, strict=True))}/5"
    print(f"  {name:44s} {cv:.5f}  folds {np.round(fa, 5).tolist()}{up}", flush=True)


def pair_sample(y: np.ndarray, idx: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """n random (fraud, non-fraud) row pairs from idx."""
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    return rng.choice(pos, n), rng.choice(neg, n)


def pairwise_fit(Z: np.ndarray, y: np.ndarray, C: float, n_pairs: int) -> Callable:
    """Pairwise logistic stacker: weights w maximise sum log sigmoid(w . (z_pos - z_neg))."""

    def fit_score(i_tr: np.ndarray, i_va: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(0)
        p, q = pair_sample(y, i_tr, n_pairs, rng)
        D = Z[p] - Z[q]
        X = np.vstack([D, -D])
        t = np.r_[np.ones(len(D)), np.zeros(len(D))]
        lr = LogisticRegression(C=C, fit_intercept=False, max_iter=2000).fit(X, t)
        return Z[i_va] @ lr.coef_[0]

    return fit_score


def segment_codes(tr: pd.DataFrame, kind: str) -> np.ndarray:
    """Segment id per row: merchant-risk quartiles, velocity quartiles, or a 3 x 3 grid of both.

    Band edges are quantiles of the column over all train rows, which uses no labels.
    """

    def bands(col: str, k: int) -> np.ndarray:
        v = tr[col].to_numpy()
        edges = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
        return np.searchsorted(edges, v, side="right")

    if kind == "mr4":
        return bands("merchant_risk_score", 4)
    if kind == "vel4":
        return bands("velocity_score", 4)
    return bands("merchant_risk_score", 3) * 3 + bands("velocity_score", 3)


def segment_fit(Z: np.ndarray, y: np.ndarray, seg: np.ndarray, C: float) -> Callable:
    """Logistic stacker on [Z, Z x one-hot(segment), one-hot(segment)], standardised, L2 at C."""
    k = seg.max() + 1
    oh = np.eye(k, dtype=np.float32)[seg]
    X = np.hstack([Z, (Z[:, :, None] * oh[:, None, :]).reshape(len(Z), -1), oh]).astype(np.float32)

    def fit_score(i_tr: np.ndarray, i_va: np.ndarray) -> np.ndarray:
        mu, sd = X[i_tr].mean(axis=0), X[i_tr].std(axis=0) + 1e-6
        lr = LogisticRegression(C=C, max_iter=2000).fit((X[i_tr] - mu) / sd, y[i_tr])
        return lr.decision_function((X[i_va] - mu) / sd)

    return fit_score


def meta_fits(Z: np.ndarray, y: np.ndarray, C: float, fl: list[Fold]) -> dict[str, Callable]:
    """Rank-gaussianised members (over all rows, or within each fold as in Deotte's s6e3 stacker), and
    members plus disagreement columns; each a logistic stacker at C. No transform uses labels."""
    from scipy.stats import norm, rankdata

    G = norm.ppf((rankdata(Z, axis=0) - 0.5) / len(Z))
    F = np.empty_like(Z)
    for _, i_va in fl:
        F[i_va] = norm.ppf((rankdata(Z[i_va], axis=0) - 0.5) / len(i_va))
    M = np.column_stack([Z.mean(axis=1), Z.std(axis=1), Z.max(axis=1) - Z.min(axis=1)])

    def lr(X: np.ndarray, std: bool) -> Callable:
        def fit_score(i_tr: np.ndarray, i_va: np.ndarray) -> np.ndarray:
            mu, sd = (X[i_tr].mean(axis=0), X[i_tr].std(axis=0) + 1e-6) if std else (0.0, 1.0)
            m = LogisticRegression(C=C, max_iter=2000).fit((X[i_tr] - mu) / sd, y[i_tr])
            return m.decision_function((X[i_va] - mu) / sd)

        return fit_score

    return {
        "rank-gaussian members": lr(G, False),
        "rank-gaussian within each fold": lr(F, False),
        "members + disagreement (std)": lr(np.hstack([Z, M]), True),
        "members, standardised (control)": lr(Z, True),
    }


def hand_zones(tr: pd.DataFrame, y: np.ndarray, i_tr: np.ndarray, i_va: np.ndarray) -> np.ndarray:
    """Held-out rows inside the four hand zones, edges refitted on the training folds (as zones.py)."""
    d_tr, d_va = tr.iloc[i_tr], tr.iloc[i_va]
    z = np.zeros(len(i_va), bool)
    for _, cond, pub, strict in ZONES:
        z |= in_zone(d_va, cond, *refit_edge(d_tr, y[i_tr], cond, pub, strict))
    return z


def zone_search(tr: pd.DataFrame, y: np.ndarray, fl: list[Fold], pred: np.ndarray, leaves: list[int]) -> None:
    """Score tree-found zero-fraud leaves on top of the hand zones, fold by fold."""
    X = tr[NUM + BOOL].astype(np.float32).copy()
    for c in CAT_STR:
        X[c] = tr[c].cat.codes.astype(np.float32)
    X = X.to_numpy()
    low = pred.min() - 1.0

    hz = pred.copy()
    hand = []
    for i_tr, i_va in fl:
        z = hand_zones(tr, y, i_tr, i_va)
        hand.append(z)
        hz[i_va[z]] = low
    base = [roc_auc_score(y[i_va], hz[i_va]) for _, i_va in fl]
    report("stack + hand zones", roc_auc_score(y, hz), base)

    for leaf in leaves:
        p = hz.copy()
        rows = frauds = 0
        for (i_tr, i_va), z_hand in zip(fl, hand, strict=True):
            t = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=0).fit(X[i_tr], y[i_tr])
            lf_tr, lf_va = t.apply(X[i_tr]), t.apply(X[i_va])
            pos = np.bincount(lf_tr, weights=y[i_tr], minlength=t.tree_.node_count)
            zero = np.flatnonzero((pos == 0) & (np.bincount(lf_tr, minlength=t.tree_.node_count) > 0))
            z = np.isin(lf_va, zero) & ~z_hand
            rows += z.sum()
            frauds += int(y[i_va][z].sum())
            p[i_va[z]] = low
        fa = [roc_auc_score(y[i_va], p[i_va]) for _, i_va in fl]
        name = f"+ tree zones, min leaf {leaf} ({rows} new rows, {frauds} fraud)"
        report(name, roc_auc_score(y, p), fa, base)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stack", type=Path, help="submission JSON with runs and C")
    ap.add_argument("--ideas", default="pair,segment,zones")
    ap.add_argument("--pairs", type=int, default=300_000)
    ap.add_argument("--leaves", default="500,1000,2000,5000")
    a = ap.parse_args()
    spec = json.loads(a.stack.read_text())
    runs, C = spec["runs"], spec["C"]
    ideas = a.ideas.split(",")

    tr, _, _ = load()
    y = tr[TARGET].to_numpy()
    Z = np.column_stack([to_logit(np.load(RUNS / r / "oof.npy"), np.zeros(1))[0] for r in runs])
    fl = folds(y)
    print(f"{a.stack.name}: {len(runs)} members, C={C:g}", flush=True)

    def lr_fit(i_tr: np.ndarray, i_va: np.ndarray) -> np.ndarray:
        return LogisticRegression(C=C, max_iter=1000).fit(Z[i_tr], y[i_tr]).decision_function(Z[i_va])

    pred, cv, base = nested(fl, y, lr_fit)
    report(f"logistic stacker C={C:g} (baseline)", cv, base)

    if "pair" in ideas:
        for pc in (0.01, 0.1, 1.0):
            _, cv_p, fa = nested(fl, y, pairwise_fit(Z, y, pc, a.pairs))
            report(f"pairwise stacker C={pc:g}, {a.pairs:,} pairs", cv_p, fa, base)
    if "segment" in ideas:
        for kind in ("mr4", "vel4", "mr3xvel3"):
            seg = segment_codes(tr, kind)
            for sc in (0.01, 0.1):
                _, cv_s, fa = nested(fl, y, segment_fit(Z, y, seg, sc))
                report(f"segment stacker {kind}, C={sc:g}", cv_s, fa, base)
    if "meta" in ideas:
        for name, f in meta_fits(Z, y, C, fl).items():
            _, cv_m, fa = nested(fl, y, f)
            report(f"meta stacker: {name}", cv_m, fa, base)
    if "zones" in ideas:
        zone_search(tr, y, fl, pred, [int(v) for v in a.leaves.split(",")])


if __name__ == "__main__":
    main()
