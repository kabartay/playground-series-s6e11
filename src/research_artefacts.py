"""Look for generator artefacts beyond per-column exact values and digit keys.

Prints, for the competition training rows:
  1. per numeric column: share of values that also occur in the original data, and the fraud rate of
     rows whose value does and does not occur there;
  2. how many rows repeat an original row exactly (all 24 features), or on column pairs;
  3. duplicate feature rows within train + test, and label agreement among duplicated train rows;
  4. per numeric column: fraud rate by how often the exact value occurs in train + test.

Read-only; nothing is saved. Usage: .venv/bin/python src/research_artefacts.py
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from common import BOOL, CAT_STR, NUM, RAW, TARGET, load


def rate(mask: np.ndarray, y: np.ndarray) -> str:
    """Row count and fraud rate of the rows in mask."""
    n = int(mask.sum())
    return f"{n:>8,} rows, fraud {100 * y[mask].mean():6.3f}%" if n else f"{0:>8} rows"


def main() -> None:
    tr, te, og = load()
    y = tr[TARGET].to_numpy()
    print(f"train {len(tr):,}, test {len(te):,}, original {len(og):,}; base fraud {100 * y.mean():.3f}%\n")

    print("1. Numeric values that also occur in the original data")
    for c in NUM:
        inog = tr[c].isin(set(og[c].unique())).to_numpy()
        share = f"{100 * inog.mean():5.1f}% in original"
        print(f"  {c:24s} {share} | in: {rate(inog, y)} | out: {rate(~inog, y)}")

    print("\n2. Rows repeating an original row")
    key = [c for c in RAW]
    ogk = og[key].astype(str).agg("|".join, axis=1)
    trk = tr[key].astype(str).agg("|".join, axis=1)
    full = trk.isin(set(ogk)).to_numpy()
    print(f"  all 24 features: {rate(full, y)}")
    conts = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd"]
    for a, b in itertools.combinations(conts + ["velocity_score", "merchant_risk_score"], 2):
        ok = set(zip(og[a], og[b], strict=True))
        m = np.fromiter((p in ok for p in zip(tr[a], tr[b], strict=True)), bool, len(tr))
        print(f"  pair {a} + {b}: {rate(m, y)}")

    print("\n3. Duplicate feature rows (train + test)")
    allk = pd.concat([trk, te[key].astype(str).agg("|".join, axis=1)], ignore_index=True)
    cnt = allk.map(allk.value_counts())
    dup = (cnt.iloc[: len(tr)] > 1).to_numpy()
    print(f"  train rows whose features occur more than once: {rate(dup, y)}")
    if dup.any():
        g = pd.DataFrame({"k": trk[dup], "y": y[dup]}).groupby("k")["y"].agg(["mean", "size"])
        g = g[g["size"] > 1]
        pure = 100 * g["mean"].isin([0, 1]).mean()
        print(f"  duplicated train groups: {len(g):,}; label-pure groups {pure:.1f}%")

    print("\n4. Fraud rate by how often the exact value occurs in train + test")
    for c in NUM:
        v = pd.concat([tr[c], te[c]], ignore_index=True)
        f = tr[c].map(v.value_counts()).to_numpy()
        bins = [0, 1, 2, 5, 20, 100, np.inf]
        out = []
        for lo, hi in itertools.pairwise(bins):
            m = (f > lo) & (f <= hi)
            if m.sum() >= 500:
                out.append(f"({lo:g},{hi:g}] {100 * y[m].mean():.2f}% n={m.sum():,}")
        print(f"  {c:24s} " + " | ".join(out))

    print("\n5. Categorical and flag levels absent from the original")
    for c in CAT_STR + BOOL:
        miss = ~tr[c].isin(set(og[c].unique())).to_numpy()
        if miss.any():
            print(f"  {c}: {rate(miss, y)}")


if __name__ == "__main__":
    main()
