"""Model families behind ``train_gbdt.py --model``: one class per family, one interface for all.

Every learner trains on one fold and predicts that fold's validation rows and the test set:

    learner = make_learner("lgb", {"num_leaves": 4})
    pred = learner.fit_predict(fold_data, seed=0)          # Prediction(val, test, n_iter)

Families (``LEARNERS``):

==========  ======================================================  =========================
key         model                                                   output scale
==========  ======================================================  =========================
``glm``     L2 logistic regression on :func:`glm_design`            logit
``mlp``     scikit-learn MLP on the same design                     probability
``realmlp`` pytabkit RealMLP (or another pytabkit classifier)       probability
``lgb``     LightGBM                                                probability, logit if GLM boost
``xgb``     XGBoost                                                 probability, logit if GLM boost
``cat``     CatBoost                                                probability, logit if GLM boost
==========  ======================================================  =========================

**GLM boost.** When :attr:`FoldData.init` holds a (train, validation, test) tuple of logits, the boosters
start from it (LightGBM ``init_score``, XGBoost ``base_margin``, CatBoost ``baseline``) and predict that
logit plus the trees' raw score.

**Early stopping.** Boosters early-stop on the validation fold unless ``n_iter`` is given, in which case
they train exactly that many rounds and never see its labels. :meth:`Learner.inner_rounds` chooses
``n_iter`` on a holdout inside the training fold (``--es inner``, honest out-of-fold scores).

Stack consumers tell the two output scales apart by range (``stack.to_logit``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, ClassVar, NamedTuple

import numpy as np
import pandas as pd

from common import BOOL, NUM, THREADS, ConfigError
from features import CATS

Offsets = tuple[np.ndarray, np.ndarray, np.ndarray]
"""Starting logits of GLM boost for (training, validation, test) rows."""
Params = dict[str, Any]


@dataclass(frozen=True)
class FoldData:
    """Everything one fold's model sees.

    Attributes:
        X, y, w: training rows, labels and sample weights.
        X_val, y_val: validation rows and labels; ``y_val`` is None when the labels must stay unseen
            (``--es inner``).
        X_test: test rows.
        init: optional GLM-boost starting logits.
    """

    X: pd.DataFrame
    y: np.ndarray
    w: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray | None
    X_test: pd.DataFrame
    init: Offsets | None = None

    def training_subset(self, rows: np.ndarray) -> FoldData:
        """The same fold restricted to some training rows (validation and test unchanged)."""
        init = None if self.init is None else (self.init[0][rows], self.init[1], self.init[2])
        return replace(self, X=self.X.iloc[rows], y=self.y[rows], w=self.w[rows], init=init)


class Prediction(NamedTuple):
    """Validation and test predictions of one fold, and the boosting rounds (or solver iterations) used."""

    val: np.ndarray
    test: np.ndarray
    n_iter: int


# ------------------------------------------------------------------------------------- GLM design


def glm_design(X: pd.DataFrame) -> np.ndarray:
    """Logit of every target-encoded column, one-hot categoricals, the 0/1 flags and any art_ features."""
    te_cols = [c for c in X.columns if c.startswith(("te_", "tej_"))]
    q = np.clip(X[te_cols].to_numpy(dtype="float64"), 1e-5, 1 - 1e-5)
    parts = [
        np.log(q / (1 - q)),
        pd.get_dummies(X[[c for c in CATS if c in X.columns]], dtype="float32").to_numpy(),
    ]
    flags = [c for c in BOOL if c in X.columns]
    parts.append(X[flags].to_numpy(dtype="float32"))
    art = [c for c in X.columns if c.startswith("art_")]
    if art:
        parts.append(X[art].to_numpy(dtype="float32"))
    return np.hstack(parts)


def make_design(params: Params, Xtr: pd.DataFrame) -> Callable[[pd.DataFrame], np.ndarray]:
    """Return a function mapping a frame to the GLM design: :func:`glm_design` plus optional splines.

    Args:
        params: ``splines`` is the number of knots per numeric column (0, the default, adds none).
        Xtr: training frame; the quantile transform and the splines are fitted on it only.
    """
    from sklearn.preprocessing import QuantileTransformer, SplineTransformer

    knots = int(params.get("splines", 0))
    nums = [c for c in NUM if c in Xtr.columns]
    if knots <= 0 or not nums:
        return glm_design
    qt = QuantileTransformer(n_quantiles=1000, subsample=200_000, random_state=0)
    st = SplineTransformer(n_knots=knots, degree=3)
    st.fit(qt.fit_transform(Xtr[nums].to_numpy(dtype="float64")))

    def design(X: pd.DataFrame) -> np.ndarray:
        return np.hstack([glm_design(X), st.transform(qt.transform(X[nums].to_numpy(dtype="float64")))])

    return design


def fit_glm(
    params: Params, Xtr: pd.DataFrame, ytr: np.ndarray, Xs: list[pd.DataFrame]
) -> tuple[list[np.ndarray], int]:
    """L2 logistic regression on the GLM design; return logits for each frame in ``Xs`` and the iterations.

    Also the starting model of GLM boost (``train_gbdt.py --offset glm``).

    Args:
        params: ``C`` (default 0.01), ``splines`` (see :func:`make_design`), ``solver`` and ``tol``
            (default lbfgs and 1e-4, scikit-learn's defaults; D29 checked lbfgs convergence).
        Xtr, ytr: training rows and labels.
        Xs: frames to score.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    design = make_design(params, Xtr)
    Dtr = design(Xtr)
    sc = StandardScaler().fit(Dtr)
    solver, tol = params.get("solver", "lbfgs"), params.get("tol", 1e-4)
    m = LogisticRegression(C=params.get("C", 0.01), solver=solver, tol=tol, max_iter=3000)
    m.fit(sc.transform(Dtr), ytr)
    return [m.decision_function(sc.transform(design(X))) for X in Xs], int(m.n_iter_[0])


# ---------------------------------------------------------------------------------------- learners


class Learner(ABC):
    """One model family. Subclasses set :attr:`key` and implement :meth:`fit_predict`.

    Args:
        params: model options from ``--params``, overriding the family's defaults.
    """

    key: ClassVar[str]
    """Name on the command line (``--model``)."""
    booster: ClassVar[bool] = False
    """Accepts ``n_iter`` and GLM-boost offsets; ``--es inner`` applies."""
    seeded: ClassVar[bool] = True
    """Results depend on the seed, so ``--seeds K`` averages K models."""
    on_glm_design: ClassVar[bool] = False
    """Reads the GLM design, so it keeps the GLM-only joint encodings (``tej_`` columns)."""
    best_iter_base: ClassVar[int] = 1
    """1 if the reported best iteration is a round count, 0 if it is a 0-based index (CatBoost)."""

    def __init__(self, params: Params | None = None) -> None:
        self.params: Params = dict(params or {})

    @abstractmethod
    def fit_predict(self, d: FoldData, seed: int = 0, n_iter: int | None = None) -> Prediction:
        """Train on ``d``'s training rows and predict its validation and test rows.

        Args:
            d: the fold.
            seed: the model's random seed.
            n_iter: boosters only: train exactly this many rounds and never read ``d.y_val``.
        """

    def inner_rounds(self, d: FoldData, seed: int, fold: int) -> int:
        """Boosting rounds for one fold, chosen without the validation fold.

        Early-stops on a stratified 10% holdout of the training rows, then scales the best round count by
        1/0.9 for the refit on all training rows. With GLM boost the holdout's starting logits come from
        the fold's GLM, which saw those rows; that can only shift the round count, never the validation fold.

        Raises:
            ConfigError: the family has no boosting rounds.
        """
        from sklearn.model_selection import train_test_split

        if not self.booster:
            raise ConfigError(f"--es inner needs a booster, not {self.key}")
        inn, hold = train_test_split(
            np.arange(len(d.y)), test_size=0.1, stratify=d.y, random_state=100 + fold
        )
        init = None if d.init is None else (d.init[0][inn], d.init[0][hold], d.init[0][hold])
        X_hold = d.X.iloc[hold]
        holdout = FoldData(d.X.iloc[inn], d.y[inn], d.w[inn], X_hold, d.y[hold], X_hold, init)
        best = int(self.fit_predict(holdout, seed=seed).n_iter) + (1 - self.best_iter_base)
        return max(1, round(best / 0.9))


class GLMLearner(Learner):
    """L2 logistic regression on the GLM design (:func:`fit_glm`); predicts logits."""

    key = "glm"
    seeded = False
    on_glm_design = True

    def fit_predict(self, d: FoldData, seed: int = 0, n_iter: int | None = None) -> Prediction:
        """See :meth:`Learner.fit_predict`; ``seed`` and ``n_iter`` do not apply."""
        (lv, lt), it = fit_glm(self.params, d.X, d.y, [d.X_val, d.X_test])
        return Prediction(lv, lt, it)


class MLPLearner(Learner):
    """Small ReLU network (scikit-learn) on the standardised GLM design; predicts probabilities.

    params: ``splines`` (see :func:`make_design`), ``hidden`` (layer sizes, default [64, 32]), ``alpha``
    (L2, default 1e-3), ``lr`` (default 1e-3), ``max_iter`` (default 40). Early stopping on an internal 10%
    split of the training rows.
    """

    key = "mlp"
    on_glm_design = True

    def fit_predict(self, d: FoldData, seed: int = 0, n_iter: int | None = None) -> Prediction:
        """See :meth:`Learner.fit_predict`; ``n_iter`` does not apply."""
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        p = self.params
        design = make_design(p, d.X)
        Dtr = design(d.X)
        sc = StandardScaler().fit(Dtr)
        m = MLPClassifier(
            hidden_layer_sizes=tuple(p.get("hidden", [64, 32])),
            alpha=p.get("alpha", 1e-3),
            learning_rate_init=p.get("lr", 1e-3),
            batch_size=2048,
            max_iter=p.get("max_iter", 40),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=seed,
        )
        m.fit(sc.transform(Dtr), d.y)
        val, test = (m.predict_proba(sc.transform(design(X)))[:, 1] for X in (d.X_val, d.X_test))
        return Prediction(val, test, int(m.n_iter_))


class RealMLPLearner(Learner):
    """pytabkit's RealMLP_TD_Classifier on the fold's feature frame; predicts probabilities.

    pytabkit picks its best epoch (and early-stops, if asked) on a validation set. That set is a stratified
    10% holdout of the training rows, never the validation fold, so the OOF stays honest; the network trains
    on the other 90%. ``params`` go to the classifier unchanged (n_ens, n_epochs, batch_size, lr,
    hidden_sizes, ...), except:

    * ``cls``: another pytabkit classifier with the same interface (e.g. ``TabM_D_Classifier``);
    * ``holdout`` (default 0.1): the holdout share. 0 trains on all training rows and hands pytabkit a 5%
      stratified subset of those same rows as its validation set, so any epoch choice is made on training
      rows (still never the validation fold); meant for fixed-epoch runs without early stopping.

    Categoricals are the pandas category columns. Needs ``pip install pytabkit``; sample weights are not
    supported (``--orig 0``).
    """

    key = "realmlp"

    def fit_predict(self, d: FoldData, seed: int = 0, n_iter: int | None = None) -> Prediction:
        """See :meth:`Learner.fit_predict`; ``n_iter`` does not apply (the epoch count is a param)."""
        import pytabkit
        import torch
        from sklearn.model_selection import train_test_split

        torch.set_num_threads(THREADS)
        params = dict(self.params)
        share = float(params.pop("holdout", 0.1))
        idx = np.arange(len(d.y))
        if share > 0:
            inn, hold = train_test_split(idx, test_size=share, stratify=d.y, random_state=100 + seed)
        else:
            inn = idx
            hold = train_test_split(idx, test_size=0.05, stratify=d.y, random_state=100 + seed)[1]
        Xtr = d.X.reset_index(drop=True)
        cls = getattr(pytabkit, params.pop("cls", "RealMLP_TD_Classifier"))
        p = dict(device="cpu", n_threads=THREADS, random_state=seed, verbosity=0)
        p.update(params)
        m = cls(**p)
        m.fit(Xtr.iloc[inn], d.y[inn], Xtr.iloc[hold], d.y[hold])
        return Prediction(
            m.predict_proba(d.X_val)[:, 1], m.predict_proba(d.X_test)[:, 1], int(p.get("n_epochs", 0))
        )


class LightGBMLearner(Learner):
    """LightGBM (``LGBMClassifier``); with GLM-boost offsets it predicts logits."""

    key = "lgb"
    booster = True
    DEFAULTS: ClassVar[Params] = dict(
        objective="binary",
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        feature_fraction=0.5,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=5.0,
        cat_smooth=20,
        n_estimators=5000,
    )

    def fit_predict(self, d: FoldData, seed: int = 0, n_iter: int | None = None) -> Prediction:
        """See :meth:`Learner.fit_predict`; early-stops after 200 rounds without AUC gain."""
        import lightgbm as lgb

        p = dict(self.DEFAULTS, num_threads=THREADS, verbose=-1, random_state=seed)
        p.update(self.params)
        if n_iter is not None:
            p["n_estimators"] = n_iter
        m = lgb.LGBMClassifier(**p)
        if n_iter is None:
            extra = {} if d.init is None else {"init_score": d.init[0], "eval_init_score": [d.init[1]]}
            m.fit(
                d.X,
                d.y,
                sample_weight=d.w,
                eval_set=[(d.X_val, d.y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(200, verbose=False)],
                **extra,
            )
            best = m.best_iteration_
        else:
            m.fit(d.X, d.y, sample_weight=d.w, **({} if d.init is None else {"init_score": d.init[0]}))
            best = n_iter
        if d.init is not None:
            return Prediction(
                m.predict(d.X_val, raw_score=True) + d.init[1],
                m.predict(d.X_test, raw_score=True) + d.init[2],
                best,
            )
        return Prediction(m.predict_proba(d.X_val)[:, 1], m.predict_proba(d.X_test)[:, 1], best)


class XGBoostLearner(Learner):
    """XGBoost (``XGBClassifier``, hist, native categoricals); with GLM-boost offsets it predicts logits.

    A ``rank:*`` objective (D30) takes one weight per query group, so it needs unit sample weights.
    """

    key = "xgb"
    booster = True
    DEFAULTS: ClassVar[Params] = dict(
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_lambda=5.0,
        n_estimators=5000,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,
        eval_metric="auc",
        early_stopping_rounds=200,
    )

    def fit_predict(self, d: FoldData, seed: int = 0, n_iter: int | None = None) -> Prediction:
        """See :meth:`Learner.fit_predict`.

        Raises:
            ConfigError: a ranking objective with non-unit sample weights (``--orig`` above 0).
        """
        import xgboost as xgb

        p = dict(self.DEFAULTS, n_jobs=THREADS, random_state=seed)
        p.update(self.params)
        if n_iter is not None:
            p.update(n_estimators=n_iter, early_stopping_rounds=None)
        m = xgb.XGBClassifier(**p)
        w: np.ndarray | None = d.w
        if str(p.get("objective", "")).startswith("rank:"):
            # Ranking objectives take one weight per query group (here the whole fold is one group).
            if w is not None and not np.all(w == 1):
                raise ConfigError("ranking objectives need unit weights (--orig 0)")
            w = None
        if n_iter is None:
            extra = {} if d.init is None else {"base_margin": d.init[0], "base_margin_eval_set": [d.init[1]]}
            m.fit(d.X, d.y, sample_weight=w, eval_set=[(d.X_val, d.y_val)], verbose=False, **extra)
            it = m.best_iteration + 1
        else:
            extra = {} if d.init is None else {"base_margin": d.init[0]}
            m.fit(d.X, d.y, sample_weight=w, verbose=False, **extra)
            it = n_iter
        if d.init is not None:
            return Prediction(
                m.predict(d.X_val, output_margin=True, base_margin=d.init[1], iteration_range=(0, it)),
                m.predict(d.X_test, output_margin=True, base_margin=d.init[2], iteration_range=(0, it)),
                it,
            )
        return Prediction(
            m.predict_proba(d.X_val, iteration_range=(0, it))[:, 1],
            m.predict_proba(d.X_test, iteration_range=(0, it))[:, 1],
            it,
        )


class CatBoostLearner(Learner):
    """CatBoost with native categoricals; with GLM-boost offsets (``baseline``) it predicts logits.

    CatBoost reports its best iteration as a 0-based index; :attr:`best_iter_base` corrects for that when
    the rounds are chosen inside the fold, and run metadata keeps CatBoost's own number.
    """

    key = "cat"
    booster = True
    best_iter_base = 0
    DEFAULTS: ClassVar[Params] = dict(
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5,
        iterations=5000,
        eval_metric="AUC",
        od_type="Iter",
        od_wait=200,
    )

    def fit_predict(self, d: FoldData, seed: int = 0, n_iter: int | None = None) -> Prediction:
        """See :meth:`Learner.fit_predict`."""
        from catboost import CatBoostClassifier, Pool

        p = dict(self.DEFAULTS, thread_count=THREADS, verbose=0, random_seed=seed)
        p.update(self.params)
        if n_iter is not None:
            p["iterations"] = n_iter
            p.pop("od_type")
            p.pop("od_wait")
        cats = [c for c in CATS if c in d.X.columns]

        def pool(
            X: pd.DataFrame,
            y: np.ndarray | None = None,
            w: np.ndarray | None = None,
            base: np.ndarray | None = None,
        ) -> Pool:
            X = X.copy()
            for c in cats:
                X[c] = X[c].astype(str)
            return Pool(X, y, weight=w, cat_features=cats, baseline=base)

        raw = "RawFormulaVal"
        m = CatBoostClassifier(**p)
        if n_iter is not None:
            m.fit(pool(d.X, d.y, d.w, None if d.init is None else d.init[0]))
            best = n_iter
        elif d.init is not None:
            m.fit(
                pool(d.X, d.y, d.w, d.init[0]),
                eval_set=pool(d.X_val, d.y_val, base=d.init[1]),
                use_best_model=True,
            )
            best = m.get_best_iteration()
        else:
            m.fit(pool(d.X, d.y, d.w), eval_set=pool(d.X_val, d.y_val), use_best_model=True)
            best = m.get_best_iteration()
        if d.init is not None:
            return Prediction(
                m.predict(pool(d.X_val), prediction_type=raw) + d.init[1],
                m.predict(pool(d.X_test), prediction_type=raw) + d.init[2],
                best,
            )
        return Prediction(m.predict_proba(pool(d.X_val))[:, 1], m.predict_proba(pool(d.X_test))[:, 1], best)


LEARNERS: dict[str, type[Learner]] = {
    cls.key: cls
    for cls in (GLMLearner, MLPLearner, RealMLPLearner, LightGBMLearner, XGBoostLearner, CatBoostLearner)
}


def make_learner(key: str, params: Params | None = None) -> Learner:
    """Instantiate the family named ``key`` (one of ``LEARNERS``).

    Raises:
        ConfigError: unknown family.
    """
    try:
        return LEARNERS[key](params)
    except KeyError:
        raise ConfigError(f"unknown model {key!r}; choose from {sorted(LEARNERS)}") from None
