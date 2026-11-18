"""Feature sets. Each builder takes (train, test, original) and returns frames with identical columns.

Fold-safe by construction: nothing here looks at the competition target. Target statistics come only
from the original dataset, which never appears in validation.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from common import BOOL, CAT_STR, NUM, RAW, TARGET


def fe_raw(tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame) -> list[pd.DataFrame]:
    """Raw columns only."""
    return [df[RAW].copy() for df in (tr, te, og)]


def fe_orig_te(tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame) -> list[pd.DataFrame]:
    """Raw columns plus, for every column, the original dataset's fraud rate and count at that value.

    Unseen values get NaN. Smoothing towards the global rate with weight 20 rows.
    """
    out = fe_raw(tr, te, og)
    prior = og[TARGET].mean()
    for c in RAW:
        key = og[c].astype(str)
        g = og.groupby(key, observed=True)[TARGET].agg(["sum", "count"])
        rate = (g["sum"] + 20 * prior) / (g["count"] + 20)
        for src, dst in zip((tr, te, og), out, strict=True):
            k = src[c].astype(str)
            dst[f"ogte_{c}"] = k.map(rate).astype("float32").values
            dst[f"ogcnt_{c}"] = k.map(g["count"]).astype("float32").values
    return out


def fe_combo(tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame) -> list[pd.DataFrame]:
    """fe_orig_te plus simple ratios and a count of risk flags."""
    out = fe_orig_te(tr, te, og)
    for src, dst in zip((tr, te, og), out, strict=True):
        dst["flags"] = src[BOOL].sum(axis=1).astype("int8")
        dst["amt_over_bal"] = (src.amount_usd / (src.account_balance_usd + 1)).astype("float32")
        dst["txn_per_hour"] = (src.txn_count_last_24h / (src.hours_since_last_txn + 0.1)).astype("float32")
        dst["risk_x_vel"] = (src.merchant_risk_score * src.velocity_score).astype("float32")
    return out


def fe_count(tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame) -> list[pd.DataFrame]:
    """fe_combo plus frequency of each numeric value over train+test+original (unsupervised)."""
    out = fe_combo(tr, te, og)
    allv = pd.concat([tr[NUM], te[NUM], og[NUM]])
    for c in NUM:
        vc = allv[c].value_counts()
        for src, dst in zip((tr, te, og), out, strict=True):
            dst[f"freq_{c}"] = src[c].map(vc).astype("float32").values
    return out


def fe_ogmodel(tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame) -> list[pd.DataFrame]:
    """fe_count plus the prediction of a LightGBM trained on the original dataset only.

    Train and test get a model fitted on all original rows; original rows get 5-fold out-of-fold
    predictions so none sees its own label. The competition target is never read.
    """
    import lightgbm as lgb
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    out = fe_count(tr, te, og)
    params = dict(
        objective="binary",
        learning_rate=0.03,
        n_estimators=400,
        num_leaves=15,
        min_child_samples=50,
        feature_fraction=0.7,
        verbose=-1,
        num_threads=4,
    )
    y = og[TARGET].to_numpy()
    oof = np.zeros(len(og))
    for i_tr, i_va in StratifiedKFold(5, shuffle=True, random_state=0).split(og, y):
        m = lgb.LGBMClassifier(**params).fit(og[RAW].iloc[i_tr], y[i_tr])
        oof[i_va] = m.predict_proba(og[RAW].iloc[i_va])[:, 1]
    m = lgb.LGBMClassifier(**params).fit(og[RAW], y)
    out[0]["ogmodel"] = m.predict_proba(tr[RAW])[:, 1].astype("float32")
    out[1]["ogmodel"] = m.predict_proba(te[RAW])[:, 1].astype("float32")
    out[2]["ogmodel"] = oof.astype("float32")
    return out


CONT = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd"]
DIGIT_COLS = [
    "merchant_risk_score",
    "velocity_score",
    "hours_since_last_txn",
    "amount_usd",
    "distance_from_home_km",
    "account_balance_usd",
]
TE_TOP = [
    "cvv_retry_count",
    "ip_country_mismatch",
    "is_foreign_transaction",
    "is_new_merchant",
    "billing_shipping_mismatch",
    "auth_method",
    "used_vpn",
    "is_ai_generated_scam_attempt",
    "merchant_category",
    "txn_count_last_24h",
]


def digits(df: pd.DataFrame) -> pd.DataFrame:
    """First and second decimal digit of continuous columns: a modulo artefact of the synthesiser."""
    import numpy as np

    out = {}
    for c in DIGIT_COLS:
        cents = np.rint(df[c].to_numpy() * 100).astype("int64")
        out[f"d1_{c}"] = ((cents // 10) % 10).astype("int8")
        out[f"d2_{c}"] = (cents % 10).astype("int8")
    return pd.DataFrame(out, index=df.index)


def fe_digits(tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame) -> list[pd.DataFrame]:
    """Raw columns plus decimal digits."""
    return [
        pd.concat([r, digits(src)], axis=1) for r, src in zip(fe_raw(tr, te, og), (tr, te, og), strict=True)
    ]


JOINT = [
    ("category_risk", ("merchant_category", "merchant_risk_score")),
    ("foreign_ip", ("is_foreign_transaction", "ip_country_mismatch")),
    ("category_amount", ("merchant_category", "q:amount_usd")),
    ("txn_velocity", ("txn_count_last_24h", "q:velocity_score")),
    ("category_age", ("merchant_category", "customer_age")),
    ("amount_risk", ("q:amount_usd", "q:merchant_risk_score")),
    ("hours_velocity", ("q:hours_since_last_txn", "q:velocity_score")),
    ("age_risk", ("customer_age", "q:merchant_risk_score")),
]
"""Column pairs whose effect on fraud is not additive, as measured and published in megayak's notebook
"S6E11 | Per-Value + Digit TE | Shallow-Tree Stack" (Apache-2.0); "q:" means deciles of a float column."""


def token_frame(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Text keys of each float column's integer part: its first two GPT-2 tokens and its 3-digit chunks.

    The integer part is written as GPT-2 would see a number in running text (" 1371" for 1371.37), so
    the tokeniser's own digit grouping (" 13", "71") becomes a pair of keys; the 3-digit chunks ("137",
    "1") group the same digits a fixed way. Idea after aphoristicartist and busyaprime's public notebooks;
    written here independently. Tokens are computed once per distinct integer (tiktoken, MIT).
    """
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    out: dict[str, np.ndarray] = {}
    for c in DIGIT_COLS:
        ints = pd.Series(np.floor(df[c].to_numpy()).astype("int64")).astype(str)
        uniq = pd.unique(ints)
        toks = {u: [enc.decode([t]) for t in enc.encode(" " + u)] for u in uniq}
        first = {u: t[0] for u, t in toks.items()}
        second = {u: (t[1] if len(t) > 1 else "") for u, t in toks.items()}
        out[f"k_{c}_g2a"] = ints.map(first).to_numpy()
        out[f"k_{c}_g2b"] = ints.map(second).to_numpy()
        out[f"k_{c}_c3a"] = ints.str[:3].to_numpy()
        out[f"k_{c}_c3b"] = ints.str[3:].to_numpy()
    return out


def te_keys(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    og: pd.DataFrame,
    pairs: bool,
    digit_keys: bool = False,
    joint: bool = False,
    token_keys: bool = False,
) -> list[pd.DataFrame]:
    """Integer category keys for in-fold target encoding (train, test, original), built on the union of rows.

    Every column's exact value is a key (when it has at most half as many levels as rows); continuous
    columns are also keyed on rounded values; decimal endings are keys; with `pairs`, every pair among
    TE_TOP is a key; with `joint`, every pair in JOINT is a key named j_<name>; with `token_keys`, the first
    two GPT-2 tokens and the 3-digit chunks of each float column's integer part are keys (`token_frame`).
    The encoding itself happens inside each fold in train_gbdt.py.
    """
    import numpy as np

    both = pd.concat([tr[RAW], te[RAW], og[RAW]], ignore_index=True)
    keys = {}
    for c in RAW:
        codes, uniq = pd.factorize(both[c].astype(str))
        if len(uniq) <= len(both) // 2:
            keys[f"k_{c}"] = codes
    for c in CONT:
        keys[f"k_{c}_int"] = pd.factorize(np.round(both[c]).astype("int64"))[0]
        keys[f"k_{c}_r10"] = pd.factorize(np.round(both[c] / 10).astype("int64"))[0]
    for c in ["merchant_risk_score", "velocity_score"]:
        keys[f"k_{c}_int"] = pd.factorize(np.round(both[c]).astype("int64"))[0]
    for c in DIGIT_COLS:
        keys[f"k_{c}_end"] = np.rint(both[c].to_numpy() * 100).astype("int64") % 100
    if digit_keys:
        # The digit-key family of the six float columns: integer part, tens, value at 0.1 resolution,
        # first one and two significant digits, and the length of the printed value.
        for c in DIGIT_COLS:
            x = both[c].to_numpy()
            text = both[c].astype(str)
            sig = text.str.replace(".", "", regex=False).str.lstrip("0-")
            keys[f"k_{c}_floor"] = pd.factorize(np.floor(x).astype("int64"))[0]
            keys[f"k_{c}_ten"] = pd.factorize(np.floor(x / 10).astype("int64"))[0]
            keys[f"k_{c}_dec"] = pd.factorize(np.floor(x * 10 + 1e-6).astype("int64"))[0]
            keys[f"k_{c}_lead1"] = pd.factorize(sig.str[:1])[0]
            keys[f"k_{c}_lead2"] = pd.factorize(sig.str[:2])[0]
            keys[f"k_{c}_len"] = text.str.len().to_numpy()
    if token_keys:
        for name, col in token_frame(both).items():
            keys[name] = pd.factorize(col)[0]
    if pairs:
        from itertools import combinations

        for a, b in combinations(TE_TOP, 2):
            keys[f"k_{a}__{b}"] = pd.factorize(both[a].astype(str) + "_" + both[b].astype(str))[0]
    if joint:
        for name, cols in JOINT:
            parts = [
                pd.qcut(both[c[2:]], 10, labels=False, duplicates="drop").astype(str)
                if c.startswith("q:")
                else both[c].astype(str)
                for c in cols
            ]
            keys[f"j_{name}"] = pd.factorize(parts[0] + "_" + parts[1])[0]
    k = pd.DataFrame(keys).astype("int32")
    n1, n2 = len(tr), len(tr) + len(te)
    return [
        k.iloc[:n1].reset_index(drop=True),
        k.iloc[n1:n2].reset_index(drop=True),
        k.iloc[n2:].reset_index(drop=True),
    ]


ART_PAIR_COLS = [
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
]


def fe_art(tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame) -> list[pd.DataFrame]:
    """Raw columns plus pooled generator artefacts (unsupervised, no labels used).

    For each numeric column with more than 100 distinct values: log1p of how often the exact value occurs
    in train + test (`art_cnt_*`) and whether it occurs in the original data (`art_inog_*`). For each pair
    of ART_PAIR_COLS: whether the pair of values occurs together in one original row (`art_pair_*`).
    Measured in `research_artefacts.py` (D14): such pooled effects are invisible to per-value encodings.
    """
    out = fe_raw(tr, te, og)
    allv = pd.concat([tr[NUM], te[NUM]])
    for c in NUM:
        if allv[c].nunique() <= 100:
            continue
        vc = allv[c].value_counts()
        ogv = set(og[c].unique())
        for src, dst in zip((tr, te, og), out, strict=True):
            dst[f"art_cnt_{c}"] = np.log1p(src[c].map(vc).fillna(0).to_numpy()).astype("float32")
            dst[f"art_inog_{c}"] = src[c].isin(ogv).to_numpy().astype("int8")
    for a, b in itertools.combinations(ART_PAIR_COLS, 2):
        ok = set(zip(og[a], og[b], strict=True))
        for src, dst in zip((tr, te, og), out, strict=True):
            hit = np.fromiter((p in ok for p in zip(src[a], src[b], strict=True)), bool, len(src))
            dst[f"art_pair_{a}__{b}"] = hit.astype("int8")
    return out


def fe_emb(tr: pd.DataFrame, te: pd.DataFrame, og: pd.DataFrame) -> list[pd.DataFrame]:
    """Raw columns plus categorical copies of every numeric column, for networks that embed categories.


    `emb_<col>`: the integer part of each numeric column as a category (levels pooled over train + test +
    original); for each DIGIT_COLS column also `emb_dec_<col>`, its two decimals as a category, and
    `dec_<col>`, the decimals as a number. Unsupervised. Idea from yekenot's public RealMLP notebook (read as
    text): the network learns an embedding per value bucket instead of seeing target encodings.
    """
    out = fe_raw(tr, te, og)
    srcs = (tr, te, og)
    for c in NUM:
        keys = [np.floor(s[c].to_numpy()).astype("int64") for s in srcs]
        levels = np.unique(np.concatenate(keys))
        for k, dst in zip(keys, out, strict=True):
            dst[f"emb_{c}"] = pd.Categorical(k, categories=levels)
    for c in DIGIT_COLS:
        keys = [np.rint((s[c].to_numpy() % 1) * 100).astype("int64") for s in srcs]
        levels = np.unique(np.concatenate(keys))
        for k, dst in zip(keys, out, strict=True):
            dst[f"emb_dec_{c}"] = pd.Categorical(k, categories=levels)
            dst[f"dec_{c}"] = (k / 100).astype("float32")
    return out


FEATURE_SETS = {
    "raw": fe_raw,
    "art": fe_art,
    "ogte": fe_orig_te,
    "combo": fe_combo,
    "count": fe_count,
    "ogmodel": fe_ogmodel,
    "digits": fe_digits,
    "emb": fe_emb,
}
CATS = CAT_STR
