"""Fit-free mean of independently built nested stacks on logit-ranks, then zones (D31).

Four stacks, each nested on the shared folds: ``stack_pub5`` (ours), goodpjw2008's wide LM stack,
DavutKursun's "0.9526 Wall" stack and dhananjayk2609's wide stack (both Kaggle notebook outputs, Apache-2.0).
Nothing is fitted here: each stack's OOF and test scores become logit-ranks and are averaged with equal
weights. The script checks the shared folds and each stack's per-fold AUCs against the figures its author
printed, prints the nested CV with zones 1-4 refitted per fold, and writes the unzoned OOF and test scores
for ``zones.py``.

Run from ``src/`` (after ``stack.py ... --oof-out``, which writes the ``stack_pub5`` OOF)::

    ../.venv/bin/python stack_average.py --pub5-oof ../runs/stack_pub5_oof.npy --out-oof X.npy --out-sub X.csv
    ../.venv/bin/python zones.py --oof X.npy --sub X.csv --out ../submissions/stack_pub4wz.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from common import DATA, RUNS, TARGET, CVScore, fold_ids, folds, load, logit_rank
from zones import hand_zones, send_to_bottom

PUB = DATA / "public"
DK = PUB / "davutkursun__s6e11-the-0-9526-wall-who-still-adds-signal"
DH = PUB / "dhananjayk2609__s6e11-fraud-part-3-final-blend-lb-0-953" / "oof"
# Per-fold nested AUCs as printed in each notebook's log (2026-11-21 versions).
DK_FOLDS = [0.95168, 0.95254, 0.95764, 0.95272, 0.94847]
DH_FOLDS = [0.95182, 0.95261, 0.95737, 0.95279, 0.94838]
AUTHOR_FOLDS = {"DavutKursun wall": DK_FOLDS, "dhananjayk2609 wide": DH_FOLDS}


class InputError(Exception):
    """An input stack that does not line up with ours (ids, folds, per-fold AUCs or test scores)."""


def load_stacks(pub5_oof: Path, pub5_sub: Path, tr: pd.DataFrame, te: pd.DataFrame) -> tuple[dict, dict]:
    """OOF and test scores of the four stacks, by name, after checking ids and folds.

    Raises:
        InputError: ids or folds differ from ours.
    """
    fold_id = fold_ids(tr[TARGET].to_numpy())
    dk_o, dk_t = pd.read_csv(DK / "oof_wall_stack.csv"), pd.read_csv(DK / "test_wall_stack.csv")
    p5 = pd.read_csv(pub5_sub)
    if not (
        np.array_equal(dk_o.id.to_numpy(), tr.id.to_numpy()) and np.array_equal(dk_o.fold.to_numpy(), fold_id)
    ):
        raise InputError("DavutKursun's stack: ids or folds differ from ours")
    if not (
        np.array_equal(dk_t.id.to_numpy(), te.id.to_numpy())
        and np.array_equal(p5.id.to_numpy(), te.id.to_numpy())
    ):
        raise InputError("test ids differ")
    wide = RUNS / "pub_fa_wide_lm_stack"
    oof = {
        "stack_pub5": np.load(pub5_oof, allow_pickle=False),
        "wide LM stack": np.load(wide / "oof.npy", allow_pickle=False),
        "DavutKursun wall": dk_o.stack_all.to_numpy(),
        "dhananjayk2609 wide": np.load(DH / "18_wide_stack_oof.npy", allow_pickle=False),
    }
    test = {
        "stack_pub5": p5.is_fraud.to_numpy(),
        "wide LM stack": np.load(wide / "test.npy", allow_pickle=False),
        "DavutKursun wall": dk_t.stack_all.to_numpy(),
        "dhananjayk2609 wide": np.load(DH / "18_wide_stack_test.npy", allow_pickle=False),
    }
    return oof, test


def main(argv: list[str] | None = None) -> int:
    """Check the inputs, print the nested CV of the mean with zones, and write the unzoned scores."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--pub5-oof", type=Path, required=True, help="nested OOF of stack_pub5 (stack.py --oof-out)"
    )
    ap.add_argument("--pub5-sub", type=Path, default=Path("../submissions/stack_pub5.csv"))
    ap.add_argument("--out-oof", type=Path, required=True)
    ap.add_argument("--out-sub", type=Path, required=True)
    a = ap.parse_args(argv)

    tr, te, _ = load()
    y = tr[TARGET].to_numpy()
    fl = folds(y)
    try:
        oof, test = load_stacks(a.pub5_oof, a.pub5_sub, tr, te)
        for name, v in oof.items():
            score = CVScore.of(y, v, fl)
            print(f"{name:22s} OOF {score.cv:.5f}  folds {score.fold_list()}")
            ref = AUTHOR_FOLDS.get(name)
            if ref and max(abs(f - r) for f, r in zip(score.folds, ref, strict=True)) > 1e-4:
                raise InputError(f"{name}: per-fold AUCs differ from the author's log")
            if len(test[name]) != len(te) or not np.isfinite(test[name]).all():
                raise InputError(f"{name}: bad test scores")
    except (InputError, FileNotFoundError) as e:
        print(f"stack_average.py: error: {e}", file=sys.stderr)
        return 2

    mean_oof = sum(logit_rank(v) for v in oof.values()) / len(oof)
    mean_test = sum(logit_rank(v) for v in test.values()) / len(test)
    zoned = send_to_bottom(mean_oof, fl, [hand_zones(tr, y, i_tr, i_va) for i_tr, i_va in fl])
    score = CVScore.of(y, zoned, fl)
    print(f"{'mean of 4 + zones':22s} OOF {score.cv:.5f}  folds {score.fold_list()}")
    np.save(a.out_oof, mean_oof)
    pd.DataFrame({"id": te.id, "is_fraud": expit(mean_test)}).to_csv(a.out_sub, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
