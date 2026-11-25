r"""D34: should test predictions come from one refit on all rows or from the mean of the fold models?

Our test predictions are the mean of five fold models, each with encoders and trees fitted on 80% of the
training rows. Some public notebooks refit each model once on all rows instead (more rows behind every rare
value's encoding); cross-validation cannot see that choice, because it never changes the out-of-fold rows.

This script measures it inside the training data, one level down. Each outer fold plays the test set; its
other four folds play the training rows:

* **A (current practice)**: four models, each trained on three of the four folds (inner split
  ``StratifiedKFold(4, shuffle=True, random_state=42)``), their outputs averaged on the held-out fold;
* **B (refit)**: one model trained on all four folds, rounds chosen the same way (``--es inner``).

Both read the same recipe as ``train_gbdt.py`` (same command line). The held-out fold is never used to fit
anything, and both arms predict the same rows, so the five paired AUC differences are a fair test.

    ../.venv/bin/python refit_sim.py --model lgb --offset glm --fe art --orig 0 --te single --keys full \
        --te-smooth 5,20,100 --joint glm --key-counts --es inner --name r11_gb_all_kc --params '{...}'

Writes ``runs/<name>_refitsim/meta.json`` (``cv`` and ``folds`` are arm B's, for ``kaggle_kernel.py fetch``).
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import RUNS, TARGET, ConfigError, folds, load
from train_gbdt import Experiment, check_args, parse_args

INNER_SEED = 42


def simulate(exp: Experiment, n_outer: int) -> dict[str, list[float]]:
    """Paired AUCs of arm A (mean of four inner models) and arm B (one refit) on each outer fold."""
    y = exp.y
    out: dict[str, list[float]] = {"a": [], "b": [], "rounds_a": [], "rounds_b": []}
    t0 = time.time()
    for k, (outer_tr, held) in enumerate(folds(y)[:n_outer]):
        inner = StratifiedKFold(4, shuffle=True, random_state=INNER_SEED).split(outer_tr, y[outer_tr])
        pa, rounds = 0.0, []
        for j, (i_fit, _) in enumerate(inner):
            p = exp.predict_fold(10 * k + j, exp.fold_data(outer_tr[i_fit], held, labels_visible=False))
            pa, rounds = pa + p.val / 4, [*rounds, p.n_iter]
        pb = exp.predict_fold(k, exp.fold_data(outer_tr, held, labels_visible=False))
        out["a"].append(float(roc_auc_score(y[held], pa)))
        out["b"].append(float(roc_auc_score(y[held], pb.val)))
        out["rounds_a"].append(float(np.mean(rounds)))
        out["rounds_b"].append(float(pb.n_iter))
        print(
            f"outer fold {k}: mean of 4 {out['a'][-1]:.5f}  refit {out['b'][-1]:.5f}  "
            f"diff {1e5 * (out['b'][-1] - out['a'][-1]):+.1f} pts  rounds {out['rounds_a'][-1]:.0f} / "
            f"{out['rounds_b'][-1]:.0f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the simulation for one recipe and write its metadata; return the exit status."""
    a = parse_args(argv)
    try:
        check_args(a)
        if a.train_frac < 1:
            raise ConfigError("refit_sim.py compares training-set sizes itself (no --train-frac)")
        tr, te, og = load()
        if a.rows:  # smoke test on a stratified subsample
            tr = train_test_split(tr, train_size=a.rows, stratify=tr[TARGET], random_state=0)[0].reset_index(
                drop=True
            )
        res = simulate(Experiment(a, tr, te, og), a.folds)
    except (ConfigError, FileNotFoundError) as e:
        print(f"refit_sim.py: error: {e}", file=sys.stderr)
        return 2
    diff = np.array(res["b"]) - np.array(res["a"])
    print(
        f"{a.name}: refit minus mean-of-4 {1e5 * diff.mean():+.1f} pts (SD {1e5 * diff.std():.1f}), "
        f"refit ahead on {int((diff > 0).sum())} of {len(diff)} outer folds"
    )
    if not a.rows:
        d = RUNS / f"{a.name}_refitsim"
        d.mkdir(parents=True, exist_ok=True)
        sim = {f"sim_{k}": v for k, v in res.items()}
        meta = dict(vars(a), cv=float(np.mean(res["b"])), folds=res["b"], **sim)
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
