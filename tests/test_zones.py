"""Zero-fraud zones: edge refitting, membership and the shared bottom-sending helper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import folds
from features import Frames
from zones import ZONES, Zone, clean, hand_zones, send_to_bottom

ALL = Zone("all", lambda d: np.ones(len(d), bool), 10.0, True)


def test_refit_lowers_the_edge_below_the_lowest_fraud() -> None:
    d = pd.DataFrame({"merchant_risk_score": [1.0, 5.0, 8.0, 12.0]})
    assert ALL.refit(d, np.array([0, 0, 1, 0])) == (8.0, True)
    assert ALL.refit(d, np.array([0, 0, 0, 1])) == (10.0, True)  # capped at the published edge


def test_contains_respects_strictness() -> None:
    d = pd.DataFrame({"merchant_risk_score": [7.9, 8.0]})
    assert ALL.contains(d, 8.0, True).tolist() == [True, False]
    assert ALL.contains(d, 8.0, False).tolist() == [True, True]


def test_clean_needs_all_eight_flags_off(frames: Frames) -> None:
    tr = frames[0]
    ok = clean(tr)
    assert ok.dtype == bool and not ok[tr["used_vpn"].to_numpy() == 1].any()


def test_hand_zones_hold_no_training_fraud(frames: Frames) -> None:
    tr = frames[0]
    y = tr["is_fraud"].to_numpy()
    i_tr, i_va = folds(y)[0]
    z_tr = hand_zones(tr, y, i_tr, i_tr)
    assert y[i_tr][z_tr].sum() == 0
    assert hand_zones(tr, y, i_tr, i_va, ZONES[:1]).sum() <= hand_zones(tr, y, i_tr, i_va).sum()


def test_send_to_bottom_only_moves_masked_rows() -> None:
    y = np.r_[np.zeros(8), np.ones(2)].astype(int)
    fl = folds(y, n=2)
    pred = np.arange(10.0)
    masks = [np.zeros(len(v), bool) for _, v in fl]
    masks[0][0] = True
    out = send_to_bottom(pred, fl, masks)
    moved = fl[0][1][0]
    assert out[moved] == -1.0 and np.array_equal(np.delete(out, moved), np.delete(pred, moved))
