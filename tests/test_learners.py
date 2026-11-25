"""Every model family on a small synthetic fold: interface, offsets, inner rounds and option checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from common import TARGET, ConfigError, folds
from encoding import FoldEncoder
from features import FEATURE_SETS, Frames, te_keys
from learners import LEARNERS, CatBoostLearner, FoldData, Learner, fit_glm, glm_design, make_learner

FAST = {
    "lgb": {"n_estimators": 60, "num_leaves": 4},
    "xgb": {"n_estimators": 60, "max_depth": 2},
    "cat": {"iterations": 60, "depth": 3},
    "glm": {"splines": 4},
    "mlp": {"max_iter": 40, "hidden": [16], "lr": 0.01},
}


def fold(frames: Frames, offset: bool = False, y_val: bool = True) -> FoldData:
    """Fold 0 of the synthetic frames, target-encoded, optionally with GLM-boost offsets."""
    tr, te, og = frames
    y = tr[TARGET].to_numpy()
    i_tr, i_va = folds(y)[0]
    enc = FoldEncoder(te_keys(tr, te, og, pairs=False), y, ["20"])
    Xa, Xva, Xte, _ = enc.transform(FEATURE_SETS["raw"](tr, te, og), i_tr, i_va)
    init = None
    if offset:
        logits, _ = fit_glm({"C": 0.01}, Xa, y[i_tr], [Xa, Xva, Xte])
        init = (logits[0], logits[1], logits[2])
    return FoldData(Xa, y[i_tr], np.ones(len(i_tr)), Xva, y[i_va] if y_val else None, Xte, init)


def test_registry_and_unknown_model() -> None:
    assert set(LEARNERS) == {"glm", "mlp", "realmlp", "lgb", "xgb", "cat"}
    assert all(issubclass(c, Learner) and c.key == k for k, c in LEARNERS.items())
    with pytest.raises(ConfigError, match="unknown model"):
        make_learner("svm")


def test_glm_design_has_one_logit_per_encoding(frames: Frames) -> None:
    d = fold(frames)
    n_te = sum(c.startswith("te_") for c in d.X.columns)
    # logits of the encodings, 5 categoricals x 3 levels one-hot, 6 flags
    assert glm_design(d.X).shape == (len(d.X), n_te + 15 + 6)


@pytest.mark.parametrize("key", ["glm", "mlp", "lgb", "xgb", "cat"])
def test_every_family_learns_the_signal(frames: Frames, key: str) -> None:
    d = fold(frames)
    p = make_learner(key, FAST[key]).fit_predict(d, seed=0)
    assert p.val.shape == (len(d.X_val),) and p.test.shape == (len(d.X_test),)
    assert roc_auc_score(d.y_val, p.val) > 0.7


@pytest.mark.parametrize("key", ["lgb", "xgb", "cat"])
def test_boosters_with_offsets_predict_logits(frames: Frames, key: str) -> None:
    d = fold(frames, offset=True)
    p = make_learner(key, FAST[key]).fit_predict(d, seed=0)
    assert not ((p.val >= 0) & (p.val <= 1)).all()  # logits, not probabilities


@pytest.mark.parametrize("key", ["lgb", "xgb", "cat"])
def test_inner_rounds_never_read_validation_labels(frames: Frames, key: str) -> None:
    d = fold(frames, y_val=False)
    learner = make_learner(key, FAST[key])
    n = learner.inner_rounds(d, seed=0, fold=0)
    assert n >= 1
    assert learner.fit_predict(d, seed=0, n_iter=n).n_iter == n


def test_inner_rounds_refuse_non_boosters(frames: Frames) -> None:
    with pytest.raises(ConfigError, match="booster"):
        make_learner("glm").inner_rounds(fold(frames), seed=0, fold=0)


def test_catboost_reports_a_zero_based_best_iteration() -> None:
    assert CatBoostLearner.best_iter_base == 0
    assert all(c.best_iter_base == 1 for k, c in LEARNERS.items() if k != "cat")


def test_ranking_objective_needs_unit_weights(frames: Frames) -> None:
    d = fold(frames)
    weighted = FoldData(d.X, d.y, np.full(len(d.y), 2.0), d.X_val, d.y_val, d.X_test)
    with pytest.raises(ConfigError, match="unit weights"):
        make_learner("xgb", {**FAST["xgb"], "objective": "rank:pairwise"}).fit_predict(weighted)


def test_training_subset_keeps_validation_and_slices_offsets(frames: Frames) -> None:
    d = fold(frames, offset=True)
    rows = np.arange(0, len(d.y), 2)
    s = d.training_subset(rows)
    assert len(s.X) == len(rows) and np.array_equal(s.init[0], d.init[0][rows])
    assert s.X_val is d.X_val and s.init[1] is d.init[1]
    assert isinstance(s.X, pd.DataFrame)
