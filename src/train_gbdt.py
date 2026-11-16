"""5-fold CV for LightGBM / XGBoost / CatBoost. Saves OOF and test predictions to runs/<name>/.

Usage: .venv/bin/python src/train_gbdt.py --model lgb --fe ogte --orig 1 --name lgb_ogte
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import TARGET, folds, load, save_run
from features import CATS, FEATURE_SETS

THREADS = int(os.environ.get("S6E11_THREADS", "4"))  # shared Mac: keep at 4 or below


def fit_predict(model: str, params: dict, Xtr, ytr, wtr, Xva, yva, Xte):
    """Train one fold with early stopping on the validation fold; return (val_pred, test_pred, best_iter)."""
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
        )
        p.update(params)
        m = lgb.LGBMClassifier(**p)
        m.fit(
            Xtr,
            ytr,
            sample_weight=wtr,
            eval_set=[(Xva, yva)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(200, verbose=False)],
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
        )
        p.update(params)
        m = xgb.XGBClassifier(**p)
        m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False)
        it = m.best_iteration + 1
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
            random_seed=0,
        )
        p.update(params)
        cats = [c for c in CATS if c in Xtr.columns]

        def pool(X, y=None, w=None):
            X = X.copy()
            for c in cats:
                X[c] = X[c].astype(str)
            return Pool(X, y, weight=w, cat_features=cats)

        m = CatBoostClassifier(**p)
        m.fit(pool(Xtr, ytr, wtr), eval_set=pool(Xva, yva), use_best_model=True)
        return m.predict_proba(pool(Xva))[:, 1], m.predict_proba(pool(Xte))[:, 1], m.get_best_iteration()
    raise ValueError(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lgb", "xgb", "cat"])
    ap.add_argument("--fe", default="raw", choices=list(FEATURE_SETS))
    ap.add_argument("--orig", type=float, default=1.0, help="weight of original rows; 0 drops them")
    ap.add_argument("--name", required=True)
    ap.add_argument("--params", default="{}", help="JSON overrides of model params")
    ap.add_argument("--folds", type=int, default=5, help="run only the first k folds (quick checks)")
    a = ap.parse_args()
    import json

    params = json.loads(a.params)

    tr, te, og = load()
    Xtr_all, Xte, Xog = FEATURE_SETS[a.fe](tr, te, og)
    y = tr[TARGET].values
    oof = np.full(len(tr), np.nan)
    test = np.zeros(len(te))
    scores, iters = [], []
    t0 = time.time()
    fl = folds(y)[: a.folds]
    for k, (i_tr, i_va) in enumerate(fl):
        Xa, ya = Xtr_all.iloc[i_tr], y[i_tr]
        wa = np.ones(len(i_tr))
        if a.orig > 0:
            Xa = pd.concat([Xa, Xog], ignore_index=True)
            ya = np.concatenate([ya, og[TARGET].values])
            wa = np.concatenate([wa, np.full(len(og), a.orig)])
        pv, pt, it = fit_predict(a.model, params, Xa, ya, wa, Xtr_all.iloc[i_va], y[i_va], Xte)
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
    if a.folds == 5:
        save_run(
            a.name,
            oof,
            test,
            dict(vars(a), cv=cv, folds=scores, iters=iters, seconds=round(time.time() - t0)),
        )


if __name__ == "__main__":
    main()
