"""In-fold target encoding of the keys from ``features.te_keys``, the core of every strong model here.

The data is synthetic, and its generator reuses the original dataset's values: the exact value of a column,
its cents, its integer part and its leading digits all carry fraud signal that trees cannot express with
range splits. Each such key is target-encoded **inside the fold**:

* the encoder is fitted on the fold's competition training rows only;
* the training rows' own encodings are cross-fitted (``cv`` inner folds), so no row sees its own label;
* validation, test and original rows are encoded by the encoder fitted on all of the fold's training rows.

:class:`FoldEncoder` does this for one or several m-estimate smoothing weights and, optionally, adds
hierarchical encodings of each float column's coarse-to-fine key chain (:func:`hier_te`). :func:`key_counts`
adds the unsupervised log count of every key's value.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from common import ConfigError
from features import DIGIT_COLS, Frames

HIER_LEVELS = ("ten", "floor", "dec", "")
"""Coarse-to-fine keys of one float column for the hierarchical encoding: tens, integer part, 0.1 steps,
exact value ("" is the exact-value key k_<col>). Each level nests inside the one before it."""


def chain_rates(Kfit: np.ndarray, yfit: np.ndarray, Kapp: np.ndarray, m: float) -> np.ndarray:
    """Fraud rate of each applied row at every level of a key chain, shrunk towards the parent level.

    Args:
        Kfit: integer codes of the fit rows (rows x levels, coarse to fine).
        yfit: labels of the fit rows.
        Kapp: integer codes of the rows to encode, same levels.
        m: m-estimate weight. Level l's rate is (frauds + m * parent rate) / (rows + m) over the fit rows
            sharing the key, starting from the overall rate.

    Returns:
        Rates, applied rows x levels.
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


def hier_te(
    Ktr: pd.DataFrame, ytr: np.ndarray, others: Sequence[pd.DataFrame], m: float, cv: int, seed: int = 0
) -> Frames:
    """Hierarchical target encoding of the DIGIT_COLS key chains (HIER_LEVELS), fitted on the training rows.

    Training rows are cross-fitted over ``cv`` inner folds; the ``others`` frames use all training rows.
    Columns are named te_h_<col>_<level>. Idea after busyaprime's public "hier" members; written here
    independently.

    Raises:
        ConfigError: no column has a key chain (the digit-key family is missing: use ``--keys full``).
    """
    from sklearn.model_selection import StratifiedKFold

    blocks: list[list[np.ndarray]] = [[] for _ in range(1 + len(others))]
    names: list[str] = []
    for col in DIGIT_COLS:
        keys = [f"k_{col}_{lv}" if lv else f"k_{col}" for lv in HIER_LEVELS]
        keys = [k for k in keys if k in Ktr.columns]
        if len(keys) < 2:
            continue
        names += [f"te_h_{k[2:]}" for k in keys]
        A = Ktr[keys].to_numpy()
        inner = np.empty((len(A), len(keys)))
        for i_fit, i_app in StratifiedKFold(cv, shuffle=True, random_state=seed).split(A, ytr):
            inner[i_app] = chain_rates(A[i_fit], ytr[i_fit], A[i_app], m)
        blocks[0].append(inner)
        for j, K in enumerate(others, start=1):
            blocks[j].append(chain_rates(A, ytr, K[keys].to_numpy(), m))
    if not names:
        raise ConfigError("hierarchical encoding needs the digit-key family (--keys full)")
    return [pd.DataFrame(np.hstack(b).astype("float32"), columns=names) for b in blocks]


def key_counts(features: Frames, keys: Frames) -> Frames:
    """Append ``art_kc_<key>``: log1p of how many train + test rows share each key's value. Unsupervised.

    Args:
        features: (train, test, original) feature frames.
        keys: the matching (train, test, original) key frames.

    Returns:
        New (train, test, original) frames with one count column per key, in key order.
    """
    pooled = pd.concat([keys[0], keys[1]], ignore_index=True)
    counts = {c: pooled[c].value_counts() for c in keys[0].columns}
    return [
        pd.concat(
            [
                X,
                pd.DataFrame(
                    {
                        f"art_kc_{c}": np.log1p(K[c].map(vc).fillna(0).to_numpy()).astype("float32")
                        for c, vc in counts.items()
                    },
                    index=X.index,
                ),
            ],
            axis=1,
        )
        for X, K in zip(features, keys, strict=True)
    ]


class FoldEncoder:
    """Target-encode the key frames inside one fold and join the encodings to the feature frames.

    Args:
        keys: (train, test, original) key frames from ``features.te_keys``.
        y: labels of the training rows.
        smooths: m-estimate weights as given on the command line ("auto" or numbers). The first gives
            columns te_<key>, the others te_<key>_m<weight>.
        cv: inner cross-fitting folds for the training rows' own encodings.
        seed: seed of the inner split.
        glm_only_joint: name joint keys (j_*) tej_<key>, so that only GLM-design models read them.
        hier: m-estimate weight of the hierarchical encodings; 0 adds none.
    """

    def __init__(
        self,
        keys: Frames,
        y: np.ndarray,
        smooths: Sequence[str],
        cv: int = 5,
        seed: int = 0,
        glm_only_joint: bool = False,
        hier: float = 0.0,
    ) -> None:
        if len(keys) != 3:
            raise ConfigError("FoldEncoder needs (train, test, original) key frames")
        if not smooths:
            raise ConfigError("at least one smoothing weight is needed (--te-smooth)")
        self.Ktr, self.Kte, self.Kog = keys
        self.y = y
        self.smooths = list(smooths)
        self.cv = cv
        self.seed = seed
        self.glm_only_joint = glm_only_joint
        self.hier = hier

    @property
    def n_keys(self) -> int:
        """Number of key columns encoded per smoothing weight."""
        return self.Ktr.shape[1]

    def _names(self, suffix: str) -> list[str]:
        pre = ["tej_" if self.glm_only_joint and c.startswith("j_") else "te_" for c in self.Ktr.columns]
        return [f"{p}{c}{suffix}" for p, c in zip(pre, self.Ktr.columns, strict=True)]

    def encodings(self, i_tr: np.ndarray, i_va: np.ndarray) -> list[Frames]:
        """Encoding frames for (training, validation, test, original) rows of one fold."""
        from sklearn.preprocessing import TargetEncoder

        frames: list[Frames] = [[], [], [], []]
        for j, sm in enumerate(self.smooths):
            smooth = sm if sm == "auto" else float(sm)
            enc = TargetEncoder(target_type="binary", smooth=smooth, cv=self.cv, random_state=self.seed)
            cols = self._names("" if j == 0 else f"_m{sm}")
            A = enc.fit_transform(self.Ktr.iloc[i_tr], self.y[i_tr]).astype("float32")
            for k, K in enumerate((None, self.Ktr.iloc[i_va], self.Kte, self.Kog)):
                E = A if K is None else enc.transform(K).astype("float32")
                frames[k].append(pd.DataFrame(E, columns=cols))
        if self.hier > 0:
            others = [self.Ktr.iloc[i_va], self.Kte, self.Kog]
            H = hier_te(self.Ktr.iloc[i_tr], self.y[i_tr], others, self.hier, self.cv, self.seed)
            for k, h in enumerate(H):
                frames[k].append(h)
        return frames

    def transform(
        self, features: Frames, i_tr: np.ndarray, i_va: np.ndarray
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Feature frames of one fold with its encodings appended: (training, validation, test, original).

        Args:
            features: (train, test, original) feature frames.
            i_tr, i_va: the fold's training and validation row positions.
        """
        Xtr, Xte, Xog = features
        bases = (Xtr.iloc[i_tr], Xtr.iloc[i_va], Xte, Xog)
        parts = [
            pd.concat([X.reset_index(drop=True), *F], axis=1)
            for X, F in zip(bases, self.encodings(i_tr, i_va), strict=True)
        ]
        return parts[0], parts[1], parts[2], parts[3]
