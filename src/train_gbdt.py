"""5-fold CV for LightGBM / XGBoost / CatBoost / GLM / MLP / RealMLP; saves OOF and test to runs/<name>/.

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
from features import CATS, DIGIT_COLS, FEATURE_SETS, te_keys

THREADS = int(os.environ.get("S6E11_THREADS", "4"))  # shared Mac: keep at 4 or below


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


def make_design(params: dict, Xtr):
    """Return a function mapping a frame to the GLM design: glm_design plus optional numeric splines.

    params: `splines` (knots per numeric column; 0, the default, adds none). Splines are fitted on
    quantile-transformed numeric columns of the training frame Xtr only.
    """
    from sklearn.preprocessing import QuantileTransformer, SplineTransformer

    knots = int(params.get("splines", 0))
    nums = [c for c in NUM if c in Xtr.columns]
    if knots <= 0 or not nums:
        return glm_design
    qt = QuantileTransformer(n_quantiles=1000, subsample=200_000, random_state=0)
    st = SplineTransformer(n_knots=knots, degree=3)
    st.fit(qt.fit_transform(Xtr[nums].to_numpy(dtype="float64")))

    def design(X) -> np.ndarray:
        return np.hstack([glm_design(X), st.transform(qt.transform(X[nums].to_numpy(dtype="float64")))])

    return design


def fit_glm(params: dict, Xtr, ytr, Xs: list) -> tuple[list[np.ndarray], int]:
    """L2 logistic regression on the GLM design; returns logits for each frame in Xs and the iteration count.

    params: `C` (default 0.01) and `splines` (see make_design).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    design = make_design(params, Xtr)
    Dtr = design(Xtr)
    sc = StandardScaler().fit(Dtr)
    m = LogisticRegression(C=params.get("C", 0.01), max_iter=3000)
    m.fit(sc.transform(Dtr), ytr)
    return [m.decision_function(sc.transform(design(X))) for X in Xs], int(m.n_iter_[0])


def fit_mlp(params: dict, Xtr, ytr, Xs: list, seed: int = 0) -> tuple[list[np.ndarray], int]:
    """Small ReLU network on the standardised GLM design; returns probabilities for each frame in Xs.

    params: `splines` (see make_design), `hidden` (layer sizes, default [64, 32]), `alpha` (L2, default
    1e-3), `lr` (default 1e-3). Early stopping on an internal 10% split of the training rows.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    design = make_design(params, Xtr)
    Dtr = design(Xtr)
    sc = StandardScaler().fit(Dtr)
    m = MLPClassifier(
        hidden_layer_sizes=tuple(params.get("hidden", [64, 32])),
        alpha=params.get("alpha", 1e-3),
        learning_rate_init=params.get("lr", 1e-3),
        batch_size=2048,
        max_iter=params.get("max_iter", 40),
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=seed,
    )
    m.fit(sc.transform(Dtr), ytr)
    return [m.predict_proba(sc.transform(design(X)))[:, 1] for X in Xs], int(m.n_iter_)


def fit_realmlp(params: dict, Xtr, ytr, Xs: list, seed: int = 0) -> tuple[list[np.ndarray], int]:
    """RealMLP (pytabkit's RealMLP_TD_Classifier) on the fold's feature frame; probabilities for each frame.

    pytabkit picks its best epoch (and early-stops, if asked) on a validation set. That set is a stratified
    10% holdout of the training rows, never the validation fold, so the OOF stays honest; the network trains
    on the other 90%. `params` go to RealMLP_TD_Classifier unchanged (n_ens, n_epochs, batch_size, lr,
    hidden_sizes, ...). Categoricals are the pandas category columns. Needs `pip install pytabkit`.
    """
    import torch
    from pytabkit import RealMLP_TD_Classifier
    from sklearn.model_selection import train_test_split

    torch.set_num_threads(THREADS)
    inn, hold = train_test_split(np.arange(len(ytr)), test_size=0.1, stratify=ytr, random_state=100 + seed)
    Xtr = Xtr.reset_index(drop=True)
    p = dict(device="cpu", n_threads=THREADS, random_state=seed, verbosity=0)
    p.update(params)
    m = RealMLP_TD_Classifier(**p)
    m.fit(Xtr.iloc[inn], ytr[inn], Xtr.iloc[hold], ytr[hold])
    return [m.predict_proba(X)[:, 1] for X in Xs], int(p.get("n_epochs", 0))


def fit_predict(
    model: str,
    params: dict,
    Xtr,
    ytr,
    wtr,
    Xva,
    yva,
    Xte,
    init=None,
    seed: int = 0,
    n_iter: int | None = None,
):
    """Train one fold; return (val_pred, test_pred, best_iter).

    Boosters early-stop on (Xva, yva) unless `n_iter` is given, in which case they train exactly that many
    rounds and never see yva (used by `--es inner`, which picks n_iter on a holdout inside the training
    fold). `init` is an optional (train, val, test) tuple of logits that the trees start from (GLM boost);
    predictions are then that logit plus the trees' raw score. `seed` sets the model's random seed.
    """
    if model == "glm":
        (lv, lt), it = fit_glm(params, Xtr, ytr, [Xva, Xte])
        return lv, lt, it
    if model == "mlp":
        (pv, pt), it = fit_mlp(params, Xtr, ytr, [Xva, Xte], seed=seed)
        return pv, pt, it
    if model == "realmlp":
        (pv, pt), it = fit_realmlp(params, Xtr, ytr, [Xva, Xte], seed=seed)
        return pv, pt, it
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
        if n_iter is not None:
            p["n_estimators"] = n_iter
        m = lgb.LGBMClassifier(**p)
        if n_iter is None:
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
            best = m.best_iteration_
        else:
            m.fit(Xtr, ytr, sample_weight=wtr, **({} if init is None else {"init_score": init[0]}))
            best = n_iter
        if init is not None:
            return (
                m.predict(Xva, raw_score=True) + init[1],
                m.predict(Xte, raw_score=True) + init[2],
                best,
            )
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], best
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
        if n_iter is not None:
            p.update(n_estimators=n_iter, early_stopping_rounds=None)
        m = xgb.XGBClassifier(**p)
        if n_iter is None:
            extra = {} if init is None else {"base_margin": init[0], "base_margin_eval_set": [init[1]]}
            m.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xva, yva)], verbose=False, **extra)
            it = m.best_iteration + 1
        else:
            extra = {} if init is None else {"base_margin": init[0]}
            m.fit(Xtr, ytr, sample_weight=wtr, verbose=False, **extra)
            it = n_iter
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
        if n_iter is not None:
            p["iterations"] = n_iter
            p.pop("od_type")
            p.pop("od_wait")
        cats = [c for c in CATS if c in Xtr.columns]

        def pool(X, y=None, w=None, base=None):
            X = X.copy()
            for c in cats:
                X[c] = X[c].astype(str)
            return Pool(X, y, weight=w, cat_features=cats, baseline=base)

        m = CatBoostClassifier(**p)
        if n_iter is not None:
            m.fit(pool(Xtr, ytr, wtr, None if init is None else init[0]))
            if init is not None:
                raw = "RawFormulaVal"
                return (
                    m.predict(pool(Xva), prediction_type=raw) + init[1],
                    m.predict(pool(Xte), prediction_type=raw) + init[2],
                    n_iter,
                )
            return m.predict_proba(pool(Xva))[:, 1], m.predict_proba(pool(Xte))[:, 1], n_iter
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


HIER_LEVELS = ("ten", "floor", "dec", "")
"""Coarse-to-fine keys of one float column for the hierarchical encoding: tens, integer part, 0.1 steps,
exact value ("" is the exact-value key k_<col>). Each level nests inside the one before it."""


def chain_rates(Kfit: np.ndarray, yfit: np.ndarray, Kapp: np.ndarray, m: float) -> np.ndarray:
    """Fraud rate of each applied row at every level of a key chain, shrunk towards the parent level.

    Kfit, Kapp: integer codes (rows x levels, coarse to fine); level l's rate is
    (frauds + m * parent rate) / (rows + m) over the fit rows sharing the key, starting from the overall
    rate. Returns rates (applied rows x levels).
    """
    r = np.full(len(Kapp), yfit.mean())
    out = np.empty((len(Kapp), Kfit.shape[1]))
    for lvl in range(Kfit.shape[1]):
        size = int(max(Kfit[:, lvl].max(), Kapp[:, lvl].max())) + 1
        s = np.bincount(Kfit[:, lvl], weights=yfit, minlength=size)
        n = np.bincount(Kfit[:, lvl], minlength=size)
        c = Kapp[:, lvl]
        r = (s[c] + m * r) / (n[c] + m)
        out[:, lvl] = r
    return out


def hier_te(Ktr: pd.DataFrame, ytr: np.ndarray, others: list, m: float, cv: int) -> list[pd.DataFrame]:
    """Hierarchical target encoding of the DIGIT_COLS key chains (HIER_LEVELS), fitted on the training rows.

    Training rows are cross-fitted over `cv` inner folds; the other frames use all training rows. Columns
    te_h_<col>_<level>. Idea after busyaprime's public "hier" members; written here independently.
    """
    from sklearn.model_selection import StratifiedKFold

    blocks: list[list[np.ndarray]] = [[] for _ in range(1 + len(others))]
    names = []
    for col in DIGIT_COLS:
        keys = [f"k_{col}_{lv}" if lv else f"k_{col}" for lv in HIER_LEVELS]
        keys = [k for k in keys if k in Ktr.columns]
        if len(keys) < 2:
            continue
        names += [f"te_h_{k[2:]}" for k in keys]
        A = Ktr[keys].to_numpy()
        inner = np.empty((len(A), len(keys)))
        for i_fit, i_app in StratifiedKFold(cv, shuffle=True, random_state=0).split(A, ytr):
            inner[i_app] = chain_rates(A[i_fit], ytr[i_fit], A[i_app], m)
        blocks[0].append(inner)
        for j, K in enumerate(others, start=1):
            blocks[j].append(chain_rates(A, ytr, K[keys].to_numpy(), m))
    return [pd.DataFrame(np.hstack(b).astype("float32"), columns=names) for b in blocks]


def inner_rounds(model: str, params: dict, Xa, ya, wa, init, seed: int, fold: int) -> int:
    """Boosting rounds for one fold, chosen without the validation fold.

    Early-stops on a stratified 10% holdout of the training rows, then scales the best round count by
    1/0.9 for the refit on all training rows. With GLM boost the holdout's starting logits come from the
    fold's GLM, which saw those rows; that can only shift the round count, never the validation fold.
    """
    from sklearn.model_selection import train_test_split

    pos = np.arange(len(ya))
    inn, hold = train_test_split(pos, test_size=0.1, stratify=ya, random_state=100 + fold)
    sub = None if init is None else (init[0][inn], init[0][hold], init[0][hold])
    Xi = Xa.iloc[inn] if hasattr(Xa, "iloc") else Xa[inn]
    Xh = Xa.iloc[hold] if hasattr(Xa, "iloc") else Xa[hold]
    _, _, best = fit_predict(model, params, Xi, ya[inn], wa[inn], Xh, ya[hold], Xh, sub, seed=seed)
    best = int(best) + (1 if model == "cat" else 0)  # CatBoost reports a 0-based index
    return max(1, round(best / 0.9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lgb", "xgb", "cat", "glm", "mlp", "realmlp"])
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
    ap.add_argument("--te-smooth", default="auto", help='"auto", m-estimate weight, or list 5,20,100')
    ap.add_argument(
        "--keys",
        default="basic",
        choices=["basic", "full", "tokens"],
        help="full adds the digit-key family; tokens adds GPT-2 tokens and 3-digit chunks on top",
    )
    ap.add_argument(
        "--offset",
        default="none",
        choices=["none", "glm"],
        help="glm: the trees start from an in-fold GLM's logit (GLM boost; lgb, xgb or cat)",
    )
    ap.add_argument(
        "--joint",
        default="none",
        choices=["none", "all", "glm"],
        help="joint keys of non-additive column pairs: for every model (all) or the GLM only (glm)",
    )
    ap.add_argument(
        "--key-counts", action="store_true", help="log count of every key's value over train + test"
    )
    ap.add_argument("--te-cv", type=int, default=5, help="inner cross-fitting folds of the target encoder")
    ap.add_argument(
        "--hier",
        type=float,
        default=0,
        help="m > 0 adds hierarchical encodings of each float column's key chain (needs --keys full)",
    )
    ap.add_argument(
        "--es",
        default="val",
        choices=["val", "inner"],
        help="boosters: early-stop on the validation fold (val, optimistic OOF; runs before D22) or pick the"
        " rounds on a 10%% holdout of the training fold and refit on all of it (inner, honest OOF)",
    )
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

        Ktr, Kte, Kog = te_keys(
            tr,
            te,
            og,
            pairs=a.te == "pairs",
            digit_keys=a.keys in ("full", "tokens"),
            joint=a.joint != "none",
            token_keys=a.keys == "tokens",
        )
        if a.key_counts:
            # Unsupervised: how many train + test rows share each key's value (no labels involved).
            KK = pd.concat([Ktr, Kte], ignore_index=True)
            for c in Ktr.columns:
                vc = KK[c].value_counts()
                for X, K in ((Xtr_all, Ktr), (Xte, Kte), (Xog, Kog)):
                    X[f"art_kc_{c}"] = np.log1p(K[c].map(vc).fillna(0).to_numpy()).astype("float32")
        print(f"target-encoding {Ktr.shape[1]} keys in each fold", flush=True)

    def add_te(i_tr, i_va):
        """Fit the encoder on the fold's competition training rows only (cross-fitted), apply elsewhere.

        `--te-smooth` may list several m-estimate weights (e.g. 5,20,100): one encoding per weight, the
        first named te_<key>, the others te_<key>_m<weight>.
        """
        if a.te == "none":
            return Xtr_all.iloc[i_tr], Xtr_all.iloc[i_va], Xte, Xog
        frames = [[], [], [], []]
        for j, sm in enumerate(a.te_smooth.split(",")):
            smooth = sm if sm == "auto" else float(sm)
            enc = TargetEncoder(target_type="binary", smooth=smooth, cv=a.te_cv, random_state=0)
            suffix = "" if j == 0 else f"_m{sm}"
            pre = ["tej_" if a.joint == "glm" and c.startswith("j_") else "te_" for c in Ktr.columns]
            cols = [f"{p}{c}{suffix}" for p, c in zip(pre, Ktr.columns, strict=True)]
            A = enc.fit_transform(Ktr.iloc[i_tr], y[i_tr]).astype("float32")
            for k, K in enumerate((None, Ktr.iloc[i_va], Kte, Kog)):
                E = A if K is None else enc.transform(K).astype("float32")
                frames[k].append(pd.DataFrame(E, columns=cols))
        if a.hier > 0:
            H = hier_te(Ktr.iloc[i_tr], y[i_tr], [Ktr.iloc[i_va], Kte, Kog], a.hier, a.te_cv)
            for k, h in enumerate(H):
                frames[k].append(h)
        parts = []
        for X, F in zip((Xtr_all.iloc[i_tr], Xtr_all.iloc[i_va], Xte, Xog), frames, strict=True):
            parts.append(pd.concat([X.reset_index(drop=True), *F], axis=1))
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
        if a.model == "realmlp" and a.orig > 0:
            raise SystemExit("--model realmlp needs --orig 0 (no sample weights)")
        if a.model not in ("glm", "mlp"):
            # tej_ columns (joint keys for the GLM only) feed the GLM offset but not the trees.
            drop = [c for c in Xa.columns if c.startswith("tej_")]
            Xa, Xva, Xte_k = (X.drop(columns=drop) for X in (Xa, Xva, Xte_k))
        n_seeds = 1 if a.model == "glm" else a.seeds
        pv, pt, it = 0.0, 0.0, 0
        for s in range(n_seeds):
            if a.es == "inner" and a.model in ("lgb", "xgb", "cat"):
                n = inner_rounds(a.model, params, Xa, ya, wa, init, seed=s, fold=k)
                v, t, i = fit_predict(a.model, params, Xa, ya, wa, Xva, None, Xte_k, init, seed=s, n_iter=n)
            else:
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
