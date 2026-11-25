"""Stacker building blocks: scale detection, nested CV and hill climbing."""

from __future__ import annotations

import numpy as np

from common import folds
from stack import hill_climb, logistic_stacker, nested, standardise, to_logit


def test_to_logit_passes_logits_through_and_converts_probabilities() -> None:
    lo, lt = np.array([-2.0, 0.5]), np.array([1.0])
    assert to_logit(lo, lt)[0] is lo
    p, pt = to_logit(np.array([0.5, 0.25]), np.array([0.75]))
    assert np.allclose(p, [0.0, np.log(1 / 3)]) and np.allclose(pt, [np.log(3)])


def test_hill_climb_returns_weights_summing_to_one_on_the_best_member() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    good, noise = y + rng.normal(0, 0.5, 2000), rng.normal(size=2000)
    w = hill_climb(standardise(np.column_stack([noise, good]), np.column_stack([noise, good])), y)
    assert np.isclose(w.sum(), 1) and w[1] >= w[0]


def test_nested_leaves_unusable_rows_unscored() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    Z = np.column_stack([y + rng.normal(0, 1, 1000), rng.normal(size=1000)])
    rows = rng.random(1000) < 0.8
    pred, score = nested(folds(y), y, logistic_stacker(Z, y, 1.0), rows)
    assert np.isnan(pred[~rows]).all() and np.isfinite(pred[rows]).all()
    assert 0.6 < score.cv < 1
