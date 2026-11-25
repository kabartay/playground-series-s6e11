r"""Five-fold CV of one model on the shared folds; saves out-of-fold and test predictions to ``runs/<name>/``.

The workhorse of the repository: every own model in docs/DESIGN.md is one command line of this script.

    ../.venv/bin/python train_gbdt.py --model lgb --offset glm --fe art --orig 0 --te single --keys full \
        --te-smooth 5,20,100 --joint glm --key-counts --es inner --seeds 3 --name r11_gb_all_kc

Pipeline per fold (see the README's diagram):

1. feature set (``--fe``, ``features.FEATURE_SETS``) and, with ``--te``, target-encoding keys
   (``features.te_keys``) built once over train, test and original rows, without labels;
2. in-fold target encoding (``encoding.FoldEncoder``): fitted on the fold's training rows only;
3. optional original rows appended with weight ``--orig``; optional GLM-boost offset (``--offset glm``);
4. the model (``--model``, ``learners.LEARNERS``), ``--seeds`` seeds averaged, rounds chosen on the
   validation fold (``--es val``) or inside the training fold (``--es inner``, honest).

Outputs ``runs/<name>/{oof.npy,test.npy,meta.json}`` only for full five-fold runs on all rows; ``--rows``,
``--folds`` below 5 and ``--train-frac`` below 1 are checks that save nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from common import TARGET, ConfigError, folds, load, save_run
from encoding import FoldEncoder, key_counts
from features import FEATURE_SETS, Frames, te_keys
from learners import LEARNERS, FoldData, Learner, Params, Prediction, fit_glm, make_learner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line (``argv`` defaults to ``sys.argv[1:]``)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=list(LEARNERS))
    ap.add_argument("--fe", default="raw", choices=list(FEATURE_SETS))
    ap.add_argument("--orig", type=float, default=1.0, help="weight of original rows; 0 drops them")
    ap.add_argument("--name", required=True)
    ap.add_argument(
        "--params",
        default="{}",
        help="JSON overrides of model params; glm_C and glm_splines go to the GLM-boost offset",
    )
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
        "--enc-seed", type=int, default=0, help="seed of the encoders' inner cross-fitting split (D29)"
    )
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
    ap.add_argument("--seed0", type=int, default=0, help="first seed; seeds run seed0 .. seed0 + seeds - 1")
    ap.add_argument(
        "--neg-frac",
        type=float,
        default=1.0,
        help="undersampling bag: each seed trains on all frauds and this random share of legitimate rows",
    )
    ap.add_argument("--rows", type=int, default=0, help="smoke test on a stratified subsample; saves nothing")
    ap.add_argument(
        "--train-frac",
        type=float,
        default=1.0,
        help="learning curve: train each fold on this stratified fraction of its training rows, score on the"
        " full validation fold; saves nothing below 1",
    )
    return ap.parse_args(argv)


def check_args(a: argparse.Namespace) -> None:
    """Refuse option combinations that cannot work, before any data is read.

    Raises:
        ConfigError: with the reason.
    """
    learner = LEARNERS[a.model]
    if not 1 <= a.folds <= 5:
        raise ConfigError("--folds must be 1 to 5")
    if a.seeds < 1:
        raise ConfigError("--seeds must be at least 1")
    if not 0 < a.neg_frac <= 1:
        raise ConfigError("--neg-frac must be in (0, 1]")
    if not 0 < a.train_frac <= 1:
        raise ConfigError("--train-frac must be in (0, 1]")
    if a.orig < 0:
        raise ConfigError("--orig must be 0 or more")
    if a.offset == "glm" and (a.orig > 0 or a.te == "none"):
        raise ConfigError("--offset glm needs --orig 0 and target encoding")
    if a.offset == "glm" and not learner.booster:
        raise ConfigError(f"--offset glm needs a booster (lgb, xgb or cat), not {a.model}")
    if a.model == "realmlp" and a.orig > 0:
        raise ConfigError("--model realmlp needs --orig 0 (no sample weights)")
    if a.hier > 0 and (a.te == "none" or a.keys == "basic"):
        raise ConfigError("--hier needs target encoding and --keys full or tokens")
    if a.key_counts and a.te == "none":
        raise ConfigError("--key-counts needs target encoding (--te)")
    try:
        params = json.loads(a.params)
    except json.JSONDecodeError as e:
        raise ConfigError(f"--params is not valid JSON: {e}") from None
    if not isinstance(params, dict):
        raise ConfigError("--params must be a JSON object")


@dataclass
class CVResult:
    """Out-of-fold and test predictions of a run, with per-fold scores and rounds."""

    oof: np.ndarray
    test: np.ndarray
    scores: list[float] = field(default_factory=list)
    iters: list[int] = field(default_factory=list)
    seconds: float = 0.0


class Experiment:
    """One ``train_gbdt.py`` run: data, encoder and model, trained fold by fold.

    Args:
        a: parsed and checked command line.
        tr, te, og: frames from ``common.load`` (train possibly subsampled by ``--rows``).
    """

    def __init__(self, a: argparse.Namespace, tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame) -> None:
        self.a = a
        self.y: np.ndarray = tr[TARGET].values
        self.y_og: np.ndarray = og[TARGET].values
        params: Params = json.loads(a.params)
        self.glm_params: Params = {"C": params.pop("glm_C", 0.01), "splines": params.pop("glm_splines", 0)}
        self.learner: Learner = make_learner(a.model, params)
        self.features: Frames = FEATURE_SETS[a.fe](tr, te, og)
        self.encoder: FoldEncoder | None = None
        if a.te != "none":
            keys = te_keys(
                tr,
                te,
                og,
                pairs=a.te == "pairs",
                digit_keys=a.keys in ("full", "tokens"),
                joint=a.joint != "none",
                token_keys=a.keys == "tokens",
            )
            if a.key_counts:
                self.features = key_counts(self.features, keys)
            self.encoder = FoldEncoder(
                keys, self.y, a.te_smooth.split(","), a.te_cv, a.enc_seed, a.joint == "glm", a.hier
            )
            print(f"target-encoding {self.encoder.n_keys} keys in each fold", flush=True)

    def fold_data(self, i_tr: np.ndarray, i_va: np.ndarray, labels_visible: bool) -> FoldData:
        """Training, validation and test frames of one fold, encoded, with original rows and GLM offset."""
        a = self.a
        if self.encoder is None:
            Xtr, Xte, Xog = self.features
            Xa, Xva, Xte_k, Xog_k = Xtr.iloc[i_tr], Xtr.iloc[i_va], Xte, Xog
        else:
            Xa, Xva, Xte_k, Xog_k = self.encoder.transform(self.features, i_tr, i_va)
        ya = self.y[i_tr]
        wa = np.ones(len(i_tr))
        if a.orig > 0:
            Xa = pd.concat([Xa, Xog_k], ignore_index=True)
            ya = np.concatenate([ya, self.y_og])
            wa = np.concatenate([wa, np.full(len(self.y_og), a.orig)])
        init = None
        if a.offset == "glm":
            logits, _ = fit_glm(self.glm_params, Xa, ya, [Xa, Xva, Xte_k])
            init = (logits[0], logits[1], logits[2])
        if not self.learner.on_glm_design:
            # tej_ columns (joint keys for the GLM only) feed the GLM offset but not the trees.
            drop = [c for c in Xa.columns if c.startswith("tej_")]
            Xa, Xva, Xte_k = (X.drop(columns=drop) for X in (Xa, Xva, Xte_k))
        return FoldData(Xa, ya, wa, Xva, self.y[i_va] if labels_visible else None, Xte_k, init)

    def predict_fold(self, k: int, d: FoldData) -> Prediction:
        """Average the fold's predictions over the seeds (each seed with its own undersampling bag)."""
        a, learner = self.a, self.learner
        n_seeds = a.seeds if learner.seeded else 1
        inner = a.es == "inner" and learner.booster
        pv, pt, it = 0.0, 0.0, 0
        for s in range(a.seed0, a.seed0 + n_seeds):
            ds = d
            if a.neg_frac < 1:
                # Undersampling bag (D32): all frauds, a fresh random share of legitimate rows per seed.
                rng = np.random.default_rng(1000 * k + s)
                ds = d.training_subset(np.flatnonzero((d.y == 1) | (rng.random(len(d.y)) < a.neg_frac)))
            n = learner.inner_rounds(ds, seed=s, fold=k) if inner else None
            v, t, i = learner.fit_predict(ds, seed=s, n_iter=n)
            pv, pt, it = pv + v / n_seeds, pt + t / n_seeds, it + int(i) // n_seeds
        return Prediction(np.asarray(pv), np.asarray(pt), int(it))

    def run(self) -> CVResult:
        """Train every requested fold; print each fold's AUC as it finishes."""
        a, y = self.a, self.y
        fl = folds(y)[: a.folds]
        res = CVResult(np.full(len(y), np.nan), np.zeros(len(self.features[1])))
        inner = a.es == "inner" and self.learner.booster
        t0 = time.time()
        for k, (i_tr, i_va) in enumerate(fl):
            if a.train_frac < 1:
                sub = train_test_split(i_tr, train_size=a.train_frac, stratify=y[i_tr], random_state=k)[0]
                i_tr = np.sort(sub)
            p = self.predict_fold(k, self.fold_data(i_tr, i_va, labels_visible=not inner))
            res.oof[i_va] = p.val
            res.test += p.test / len(fl)
            res.scores.append(roc_auc_score(y[i_va], p.val))
            res.iters.append(p.n_iter)
            print(f"fold {k} auc {res.scores[-1]:.5f} iter {p.n_iter} ({time.time() - t0:.0f}s)", flush=True)
        res.seconds = time.time() - t0
        return res


def main(argv: list[str] | None = None) -> int:
    """Run the CV, print the scores and save the run when it is a full one; return the exit status."""
    a = parse_args(argv)
    try:
        check_args(a)
        tr, te, og = load()
        if a.rows:
            tr = train_test_split(tr, train_size=a.rows, stratify=tr[TARGET], random_state=0)[0]
            tr, te = tr.reset_index(drop=True), te.iloc[:1000].reset_index(drop=True)
        exp = Experiment(a, tr, te, og)
        res = exp.run()
    except (ConfigError, FileNotFoundError) as e:
        print(f"train_gbdt.py: error: {e}", file=sys.stderr)
        return 2
    y = exp.y
    mask = ~np.isnan(res.oof)
    cv = roc_auc_score(y[mask], res.oof[mask])
    print(
        f"{a.name}: CV {cv:.5f}  folds {np.round(res.scores, 5).tolist()}  "
        f"mean {np.mean(res.scores):.5f} sd {np.std(res.scores):.5f}"
    )
    if a.folds == 5 and not a.rows and a.train_frac == 1:
        meta = dict(vars(a), cv=cv, folds=res.scores, iters=res.iters, seconds=round(res.seconds))
        save_run(a.name, res.oof, res.test, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
