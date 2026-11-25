"""In-fold target encoding: correctness of the shrinkage chain and absence of leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common import TARGET, ConfigError, folds
from encoding import FoldEncoder, chain_rates, hier_te, key_counts
from features import FEATURE_SETS, Frames, te_keys


def test_chain_rates_shrinks_each_level_towards_its_parent() -> None:
    # Two levels; fit rows: key (0, 0) x2 with labels 1, 0 and key (1, 1) with label 1.
    K = np.array([[0, 0], [0, 0], [1, 1]])
    y = np.array([1.0, 0.0, 1.0])
    out = chain_rates(K, y, np.array([[0, 0], [1, 1]]), m=1.0)
    prior = 2 / 3
    lvl0 = [(1 + prior) / 3, (1 + prior) / 2]
    lvl1 = [(1 + lvl0[0]) / 3, (1 + lvl0[1]) / 2]
    assert np.allclose(out[:, 0], lvl0) and np.allclose(out[:, 1], lvl1)


def test_hier_te_needs_the_digit_key_family() -> None:
    K = pd.DataFrame({"k_x": [0, 1, 0, 1]})
    with pytest.raises(ConfigError, match="--keys full"):
        hier_te(K, np.array([0, 1, 0, 1]), [K], m=10, cv=2)


def test_key_counts_are_log_counts_over_train_and_test() -> None:
    keys = [pd.DataFrame({"k": [1, 1, 2]}), pd.DataFrame({"k": [1, 3]}), pd.DataFrame({"k": [2]})]
    feats = [pd.DataFrame({"a": [0.0] * len(k)}) for k in keys]
    out = key_counts(feats, keys)
    assert np.allclose(out[0]["art_kc_k"], np.log1p([3, 3, 1]))
    assert np.allclose(out[2]["art_kc_k"], np.log1p([1]))
    assert list(out[1].columns) == ["a", "art_kc_k"]


def test_fold_encoder_never_reads_validation_labels(frames: Frames) -> None:
    tr, te, og = frames
    y = tr[TARGET].to_numpy()
    keys = te_keys(tr, te, og, pairs=False, digit_keys=True)
    feats = FEATURE_SETS["raw"](tr, te, og)
    i_tr, i_va = folds(y)[0]
    flipped = y.copy()
    flipped[i_va] = 1 - flipped[i_va]
    a = FoldEncoder(keys, y, ["5", "20"], hier=10).transform(feats, i_tr, i_va)
    b = FoldEncoder(keys, flipped, ["5", "20"], hier=10).transform(feats, i_tr, i_va)
    for x, z in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(x, z)
    assert any(c.endswith("_m20") for c in a[0].columns)
    assert any(c.startswith("te_h_") for c in a[0].columns)
    assert len(a[0]) == len(i_tr) and len(a[1]) == len(i_va) and len(a[2]) == len(te)


def test_fold_encoder_names_glm_only_joint_keys(frames: Frames) -> None:
    tr, te, og = frames
    keys = te_keys(tr, te, og, pairs=False, joint=True)
    enc = FoldEncoder(keys, tr[TARGET].to_numpy(), ["auto"], glm_only_joint=True)
    cols = enc.encodings(*folds(tr[TARGET].to_numpy())[0])[0][0].columns
    assert any(c.startswith("tej_j_") for c in cols) and not any(c.startswith("te_j_") for c in cols)


def test_fold_encoder_refuses_bad_input() -> None:
    with pytest.raises(ConfigError):
        FoldEncoder([pd.DataFrame()], np.zeros(1), ["auto"])
    with pytest.raises(ConfigError):
        FoldEncoder([pd.DataFrame()] * 3, np.zeros(1), [])
