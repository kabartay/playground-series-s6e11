"""5-fold CV for LightGBM / XGBoost / CatBoost / GLM. Saves OOF and test predictions to runs/<name>/.

Usage: .venv/bin/python src/train_gbdt.py --model lgb --fe ogte --orig 1 --name lgb_ogte

`--seeds K` trains K models per fold with different random seeds and averages them (same folds);
`--rows N` runs on a stratified subsample of N training rows as a smoke test and saves nothing.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import BOOL, NUM, TARGET, folds, load, save_run
from features import CATS, FEATURE_SETS, te_keys

THREADS = int(os.environ.get("S6E11_THREADS", "4"))  # shared Mac: keep at 4 or below


def glm_design(X: pd.DataFrame) -> np.ndarray:
    """Logit of every target-encoded column plus one-hot categoricals and the 0/1 flags."""
    te_cols = [c for c in X.columns if c.startswith("te_")]
    q = np.clip(X[te_cols].to_numpy(dtype="float64"), 1e-5, 1 - 1e-5)
    parts = [
        np.log(q / (1 - q)),
        pd.get_dummies(X[[c for c in CATS if c in X.columns]], dtype="float32").to_numpy(),
    ]
    flags = [c for c in BOOL if c in X.columns]
    parts.append(X[flags].to_numpy(dtype="float32"))
    return np.hstack(parts)


def fit_glm(params: dict, Xtr, ytr, Xs: list) -> tuple[list[np.ndarray], int]:
    """L2 logistic regression on glm_design; returns logits for each frame in Xs and the iteration count.

    params: `C` (default 0.01) and `splines` (knots per numeric column; 0, the default, adds none). Splines
    are fitted on quantile-transformed numeric columns of the training frame only.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler

    knots = int(params.get("splines", 0))
    nums = [c for c in NUM if c in Xtr.columns]
    spl = None
    if knots > 0 and nums:
        qt = QuantileTransformer(n_quantiles=1000, subsample=200_000, random_state=0)
        st = SplineTransformer(n_knots=knots, degree=3)
        st.fit(qt.fit_transform(Xtr[nums].to_numpy(dtype="float64")))
        spl = (qt, st)

    def design(X) -> np.ndarray:
        D = glm_design(X)
        if spl is None:
            return D
        return np.hstack([D, spl[1].transform(spl[0].transform(X[nums].to_numpy(dtype="float64")))])

    Dtr = design(Xtr)
    sc = StandardScaler().fit(Dtr)
    m = LogisticRegression(C=params.get("C", 0.01), max_iter=3000)
    m.fit(sc.transform(Dtr), ytr)
    return [m.decision_function(sc.transform(design(X))) for X in Xs], int(m.n_iter_[0])


def fit_predict(model: str, params: dict, Xtr, ytr, wtr, Xva, yva, Xte, init=None, seed: int = 0):
    """Train one fold with early stopping on the validation fold; return (val_pred, test_pred, best_iter).

    `init` is an optional (train, val, test) tuple of logits that the trees start from (GLM boost);
    predictions are then that logit plus the trees' raw score. `seed` sets the model's random seed.
    """
    if model == "glm":
        (lv, lt), it = fit_glm(params, Xtr, ytr, [Xva, Xte])
        return lv, lt, it
    if model == "lgb":
        import lightgbm as lgb

        p = dict(
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
            num_threads=THREADS,
            verbose=-1,
            random_state=seed,
        )
        p.update(params)
        m = lgb.LGBMClassifier(**p)
        extra = {} if init is None else {"init_score": init[0], "eval_init_score": [init[1]]}
        m.fit(
            Xtr,
            ytr,
            sample_weight=wtr,
            eval_set=[(Xva, yva)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(200, verbose=False)],
            **extra,
        )
        if init is not None:
            return (
                m.predict(Xva, raw_score=True) + init[1],
                m.predict(Xte, raw_score=True) + init[2],
                m.best_iteration_,
            )
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], m.best_iteration_
    if model == "xgb":
        import xgboost as xgb

        p = dict(
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
            n_jobs=THREADS,
            random_state=seed,
        )
        p.update(params)
        m = xgb.XGBClassifier(**p)
        extra = {} if init is None else {"base_margin": init[0], "base_margin_eval_set": [init[1]]}
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False, **extra)
        it = m.best_iteration + 1
        if init is not None:
            return (
                m.predict(Xva, output_margin=True, base_margin=init[1], iteration_range=(0, it)),
                m.predict(Xte, output_margin=True, base_margin=init[2], iteration_range=(0, it)),
                it,
            )
        return (
            m.predict_proba(Xva, iteration_range=(0, it))[:, 1],
            m.predict_proba(Xte, iteration_range=(0, it))[:, 1],
            it,
        )
    if model == "cat":
        from catboost import CatBoostClassifier, Pool

        p = dict(
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5,
            iterations=5000,
            eval_metric="AUC",
            od_type="Iter",
            od_wait=200,
            thread_count=THREADS,
            verbose=0,
            random_seed=seed,
        )
        p.update(params)
        cats = [c for c in CATS if c in Xtr.columns]

        def pool(X, y=None, w=None, base=None):
            X = X.copy()
            for c in cats:
                X[c] = X[c].astype(str)
            return Pool(X, y, weight=w, cat_features=cats, baseline=base)

        m = CatBoostClassifier(**p)
        if init is not None:
            m.fit(pool(Xtr, ytr, wtr, init[0]), eval_set=pool(Xva, yva, base=init[1]), use_best_model=True)
            raw = "RawFormulaVal"
            return (
                m.predict(pool(Xva), prediction_type=raw) + init[1],
                m.predict(pool(Xte), prediction_type=raw) + init[2],
                m.get_best_iteration(),
            )
        m.fit(pool(Xtr, ytr, wtr), eval_set=pool(Xva, yva), use_best_model=True)
        return m.predict_proba(pool(Xva))[:, 1], m.predict_proba(pool(Xte))[:, 1], m.get_best_iteration()
    raise ValueError(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lgb", "xgb", "cat", "glm"])
    ap.add_argument("--fe", default="raw", choices=list(FEATURE_SETS))
    ap.add_argument("--orig", type=float, default=1.0, help="weight of original rows; 0 drops them")
    ap.add_argument("--name", required=True)
    ap.add_argument("--params", default="{}", help="JSON overrides of model params")
    ap.add_argument("--folds", type=int, default=5, help="run only the first k folds (quick checks)")
    ap.add_argument(
        "--te",
        default="none",
        choices=["none", "single", "pairs"],
        help="in-fold target encoding of exact-value keys (single) and also top-10 pairs (pairs)",
    )
    ap.add_argument("--te-smooth", default="auto", help='"auto" or an m-estimate weight such as 20')
    ap.add_argument(
        "--keys", default="basic", choices=["basic", "full"], help="full adds the digit-key family"
    )
    ap.add_argument(
        "--offset",
        default="none",
        choices=["none", "glm"],
        help="glm: the trees start from an in-fold GLM's logit (GLM boost; lgb, xgb or cat)",
    )
    ap.add_argument("--te-cv", type=int, default=5, help="inner cross-fitting folds of the target encoder")
    ap.add_argument("--seeds", type=int, default=1, help="models per fold with different seeds, averaged")
    ap.add_argument("--rows", type=int, default=0, help="smoke test on a stratified subsample; saves nothing")
    a = ap.parse_args()
    import json

    params = json.loads(a.params)
    glm_params = {"C": params.pop("glm_C", 0.01), "splines": params.pop("glm_splines", 0)}

    tr, te, og = load()
    if a.rows:
        from sklearn.model_selection import train_test_split

        tr = train_test_split(tr, train_size=a.rows, stratify=tr[TARGET], random_state=0)[0]
        tr, te = tr.reset_index(drop=True), te.iloc[:1000].reset_index(drop=True)
    Xtr_all, Xte, Xog = FEATURE_SETS[a.fe](tr, te, og)
    y = tr[TARGET].values
    if a.te != "none":
        from sklearn.preprocessing import TargetEncoder

        Ktr, Kte, Kog = te_keys(tr, te, og, pairs=a.te == "pairs", digit_keys=a.keys == "full")
        print(f"target-encoding {Ktr.shape[1]} keys in each fold", flush=True)

    def add_te(i_tr, i_va):
        """Fit the encoder on the fold's competition training rows only (cross-fitted), apply elsewhere."""
        if a.te == "none":
            return Xtr_all.iloc[i_tr], Xtr_all.iloc[i_va], Xte, Xog
        smooth = a.te_smooth if a.te_smooth == "auto" else float(a.te_smooth)
        enc = TargetEncoder(target_type="binary", smooth=smooth, cv=a.te_cv, random_state=0)
        cols = [f"te_{c}" for c in Ktr.columns]
        A = enc.fit_transform(Ktr.iloc[i_tr], y[i_tr]).astype("float32")
        parts = []
        for X, K, enc_rows in (
            (Xtr_all.iloc[i_tr], None, A),
            (Xtr_all.iloc[i_va], Ktr.iloc[i_va], None),
            (Xte, Kte, None),
            (Xog, Kog, None),
        ):
            E = enc_rows if enc_rows is not None else enc.transform(K).astype("float32")
            parts.append(pd.concat([X.reset_index(drop=True), pd.DataFrame(E, columns=cols)], axis=1))
        return tuple(parts)

    oof = np.full(len(tr), np.nan)
    test = np.zeros(len(te))
    scores, iters = [], []
    t0 = time.time()
    fl = folds(y)[: a.folds]
    for k, (i_tr, i_va) in enumerate(fl):
        Xa, Xva, Xte_k, Xog_k = add_te(i_tr, i_va)
        ya = y[i_tr]
        wa = np.ones(len(i_tr))
        if a.orig > 0:
            Xa = pd.concat([Xa, Xog_k], ignore_index=True)
            ya = np.concatenate([ya, og[TARGET].values])
            wa = np.concatenate([wa, np.full(len(og), a.orig)])
        init = None
        if a.offset == "glm":
            if a.orig > 0 or a.te == "none":
                raise SystemExit("--offset glm needs --orig 0 and target encoding")
            init, _ = fit_glm(glm_params, Xa, ya, [Xa, Xva, Xte_k])
        n_seeds = 1 if a.model == "glm" else a.seeds
        pv, pt, it = 0.0, 0.0, 0
        for s in range(n_seeds):
            v, t, i = fit_predict(a.model, params, Xa, ya, wa, Xva, y[i_va], Xte_k, init, seed=s)
            pv, pt, it = pv + v / n_seeds, pt + t / n_seeds, it + int(i) // n_seeds
        oof[i_va] = pv
        test += pt / len(fl)
        scores.append(roc_auc_score(y[i_va], pv))
        iters.append(int(it))
        print(f"fold {k} auc {scores[-1]:.5f} iter {it} ({time.time() - t0:.0f}s)", flush=True)
    mask = ~np.isnan(oof)
    cv = roc_auc_score(y[mask], oof[mask])
    print(
        f"{a.name}: CV {cv:.5f}  folds {np.round(scores, 5).tolist()}  "
        f"mean {np.mean(scores):.5f} sd {np.std(scores):.5f}"
    )
    if a.folds == 5 and not a.rows:
        save_run(
            a.name,
            oof,
            test,
            dict(vars(a), cv=cv, folds=scores, iters=iters, seconds=round(time.time() - t0)),
        )


if __name__ == "__main__":
    main()
