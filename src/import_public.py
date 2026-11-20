"""Import public OOF + test predictions as runs, after checking they line up with our folds.

Sources live in data/public/<owner>__<dataset>/ (downloaded with `kaggle datasets download`, never
committed). Each member becomes runs/pub_<tag>_<name>/{oof.npy,test.npy,meta.json} and can then be
stacked with our own runs by stack.py. Checks before import:

  * 699,960 OOF and 299,984 test values, all finite (files loaded with allow_pickle=False);
  * the source's fold file equals our folds (`common.folds`) row by row, when the source ships one;
    sources without one must state the same split (StratifiedKFold(5, shuffle=True, random_state=42));
  * the OOF AUC we compute matches the AUC the source reports, within 0.0001.

A member that fails any check is skipped with the reason printed. Usage: .venv/bin/python src/import_public.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from common import DATA, RUNS, TARGET, folds, load

PUB = DATA / "public"
# Bundle streams imported elsewhere: masaya_gpt2 (tag mk), gbn_bag5_w512 (tag gp), and the two LMs whose
# bundle test predictions come from the fold-0 model only (taken from goodpjw2008's datasets with 5-fold
# test files instead).
BUNDLE_SKIP = {"masaya_gpt2", "gbn_bag5_w512", "lm7_5fold", "gpt2_seed2026"}
# OOF AUC stated in each LM dataset's README. Their folds are stated equal to ours and confirmed by the q17
# bundle, whose fold array equals ours and whose copies of these two members have the same per-fold AUCs.
LM_AUC = {"lm7": 0.949448, "gpt2_s2026": 0.950140}
# Sources with no fold file whose description states StratifiedKFold(5, shuffle=True, random_state=42) on
# is_fraud in train.csv order: by construction the same split as `common.folds`.
STATED = "stated"
# dariushafshar's library members that early-stop on the validation fold itself (disclosed in its manifest):
# their OOF is optimistic, which would leak into a stacker.
DH_SKIP = {"a", "h", "i"}


def our_fold_ids(y: np.ndarray) -> np.ndarray:
    """Fold index of every training row under the shared split."""
    f = np.empty(len(y), dtype=np.int8)
    for k, (_, i_va) in enumerate(folds(y)):
        f[i_va] = k
    return f


def members() -> list[dict]:
    """Every candidate member: tag, name, oof and test paths, reported AUC, fold file, licence note."""
    out = []
    d = PUB / "goodpjw2008__s6e11-oof-bank"
    if d.exists():
        for r in csv.DictReader(open(d / "manifest.csv")):
            if r["core"] == "True":
                out.append(
                    dict(
                        tag="gp",
                        name=r["name"],
                        oof=d / f"oof_{r['name']}.npy",
                        test=d / f"test_{r['name']}.npy",
                        auc=float(r["oof_auc"]),
                        folds=d / "folds.csv",
                        source="goodpjw2008/s6e11-oof-bank",
                        note=f"{r['family']}; test {r['test_kind']}",
                    )
                )
    d = PUB / "mathsonchain__s6e11-fraud-oof-test-17-single-models"
    if d.exists():
        for p in sorted(d.glob("oof_*.npy")):
            name = p.stem[4:]
            out.append(
                dict(
                    tag="mc",
                    name=name,
                    oof=p,
                    test=d / f"test_{name}.npy",
                    auc=None,
                    folds=d / "fold_ids.npy",
                    source="mathsonchain/s6e11-fraud-oof-test-17-single-models",
                    note="",
                )
            )
    d = PUB / "masayakawamata__s6e11-gpt2-target-first-oof"
    if d.exists():
        out.append(
            dict(
                tag="mk",
                name="gpt2",
                oof=d / "oof.npy",
                test=d / "test.npy",
                auc=0.950027203793,
                folds=d / "fold_ids.npy",
                source="masayakawamata/s6e11-gpt2-target-first-oof",
                note="GPT-2 target-first classifier, log-odds; CC BY 4.0, attribute Masaya Kawamata",
            )
        )
    d = PUB / "amanatar__s6e11-rank1-q17-meta-bundle"
    if d.exists():
        z = d / "rank1_q17_streams.npz"
        with np.load(z, allow_pickle=False) as npz:
            names = [str(n) for n in npz["stream_names"]]
        for n in names:
            if n in BUNDLE_SKIP:
                continue
            out.append(
                dict(
                    tag="q17",
                    name=n,
                    oof=(z, f"oof_{n}"),
                    test=(z, f"test_{n}"),
                    auc=None,
                    folds=(z, "folds"),
                    source="amanatar/s6e11-rank1-q17-meta-bundle",
                    note="logit ranks; CC0",
                )
            )
    for ds, name, oof, test in (
        ("s6e11-compact-lm-seed7", "lm7", "lm7_oof_5fold.npy", "lm7_test_5fold.npy"),
        (
            "s6e11-target-first-gpt2-seed-2026",
            "gpt2_s2026",
            "gpt2_s2026_oof_5fold.npy",
            "gpt2_s2026_test_5fold.npy",
        ),
    ):
        d = PUB / f"goodpjw2008__{ds}"
        if d.exists():
            out.append(
                dict(
                    tag="gw",
                    name=name,
                    oof=d / oof,
                    test=d / test,
                    auc=LM_AUC[name],
                    folds=None,
                    source=f"goodpjw2008/{ds}",
                    note="target-first LM log-odds, test = mean of the 5 fold models; CC0",
                )
            )
    d = PUB / "busyaprime__s6e11-33-own-members"
    if d.exists():
        for col, r in enumerate(csv.DictReader(open(d / "manifest.csv"))):
            out.append(
                dict(
                    tag="bp",
                    name=r["name"],
                    oof=(d / "oof.npy", col),
                    test=(d / "test.npy", col),
                    auc=float(r["oof_auc"]),
                    folds=STATED,
                    source="busyaprime/s6e11-33-own-members",
                    note=f"{r['family']}; CC0",
                )
            )
    lib = PUB / "dariushafshar__s6e11-golem-oof-library"
    for d, tag in ((lib, "dh"), (PUB / "dariushafshar__s6e11-gpu-oof-members", "dg")):
        if d.exists() and lib.exists():
            for r in csv.DictReader(open(d / "manifest.csv")):
                if r["member"] in DH_SKIP:
                    continue
                out.append(
                    dict(
                        tag=tag,
                        name=r["member"],
                        oof=d / f"oof_{r['member']}.npy",
                        test=d / f"test_{r['member']}.npy",
                        auc=float(r["oof_auc"]),
                        folds=lib / "folds_seed42.npy",
                        source=f"dariushafshar/{d.name.split('__')[1]}",
                        note=f"{r.get('model_family') or r.get('model')}; CC0",
                    )
                )
    d = PUB / "alexchilton__s6e11-oof-predictions"
    if d.exists():
        man = json.loads((d / "oof_manifest.json").read_text())
        for h, m in man.items():
            if (d / f"oof_{h}.npy").exists():
                out.append(
                    dict(
                        tag="ac",
                        name=f"{m['name']} {m.get('inputs', '')}".strip(),
                        oof=d / f"oof_{h}.npy",
                        test=d / f"test_{h}.npy",
                        auc=m.get("oof_auc"),
                        folds=STATED,
                        source="alexchilton/s6e11-oof-predictions",
                        note=f"{m.get('version', '')} {h}",
                    )
                )
    # goodpjw2008's series artifacts: the two nested wide stacks only (single models duplicate the bank).
    d = PUB / "goodpjw2008__s6e11-fraud-series-artifacts"
    if d.exists():
        for name in ("wide_lm_stack", "wide_stack"):
            out.append(
                dict(
                    tag="fa",
                    name=name,
                    oof=d / f"{name}_nested.npy",
                    test=d / f"{name}_test.npy",
                    auc=None,
                    folds=STATED,
                    source="goodpjw2008/s6e11-fraud-series-artifacts",
                    note="logistic stack of about 90 public members, nested on the shared split; CC0",
                )
            )
    # yekenot's RealMLP outputs (seed-42 split; pytabkit: 3 epochs, no early stopping).
    for kind, auc in (("pytabkit", 0.95031), ("pytorch", 0.95002)):
        d = PUB / f"yekenot__realmlp-{kind}"
        if d.exists():
            out.append(
                dict(
                    tag="ym",
                    name=f"realmlp_{kind}",
                    oof=(d / "oof_preds.csv", "is_fraud"),
                    test=(d / "submission.csv", "is_fraud"),
                    auc=auc,
                    folds=STATED,
                    source=f"yekenot/ps-s6-e11-realmlp-{kind} (notebook output)",
                    note="RealMLP, 5-fold OOF and fold-mean test; ids checked against train/sample order",
                )
            )
    # Weni Mohamed's diverse OOF pack (CC0; BPE-digit boosters, a GLM, RealMLP, TabM). No fold file; the
    # description states StratifiedKFold(5, shuffle=True, random_state=42), and model_summary.csv gives
    # per-fold AUCs, which must match ours (checked in main).
    d = PUB / "wenimohamed__s6e11-diverse-bpe-realmlp-tabm-oof-pack"
    if d.exists():
        for r in csv.DictReader(open(d / "model_summary.csv")):
            out.append(
                dict(
                    tag="wm",
                    name=r["model_name"].removeprefix("weni_"),
                    oof=d / r["oof_file"],
                    test=d / r["test_file"],
                    auc=float(r["oof_roc_auc"]),
                    fold_aucs=[float(r[f"fold{k}_auc"]) for k in range(5)],
                    folds=STATED,
                    source="wenimohamed/s6e11-diverse-bpe-realmlp-tabm-oof-pack",
                    note=r["architecture"] + "; CC0",
                )
            )
    # yekenot's TabM (pytabkit TabM_D, one epoch, so no early stopping on the scored fold; seed-42 split).
    d = PUB / "yekenot__tabm-pytabkit"
    if d.exists():
        out.append(
            dict(
                tag="ym",
                name="tabm_pytabkit",
                oof=(d / "oof_preds.csv", "is_fraud"),
                test=(d / "submission.csv", "is_fraud"),
                auc=0.94900,
                fold_aucs=[0.94912, 0.94980, 0.95373, 0.94920, 0.94351],
                folds=STATED,
                source="yekenot/ps-s6-e11-tabm-pytabkit (notebook output)",
                note="TabM_D, 1 epoch, in-fold target encoding; per-fold AUCs from the notebook log",
            )
        )
    # megayak's own target-first GPT-2 (CC0, 2026-11-19, all five folds) and RealMLP (CC0, 2026-11-18,
    # 8 seeds, last epoch, nothing chosen on the scored fold). No fold file or AUC; both state the shared
    # split. The LM's OOF holds log-odds but its test file rank fractions, so its OOF becomes ranks too.
    for ds, name, oof, test, rank, note in (
        ("s6e11-target-first-lm", "lm", "lm_oof.npy", "lm_test.npy", True, "GPT-2 target-first"),
        ("s6e11-realmlp-member", "realmlp_8seeds", "mlp_oof.npy", "mlp_test.npy", False, "RealMLP, 8 seeds"),
    ):
        d = PUB / f"megayak__{ds}"
        if d.exists():
            out.append(
                dict(
                    tag="mg",
                    name=name,
                    oof=d / oof,
                    test=d / test,
                    auc=None,
                    folds=STATED,
                    rank_oof=rank,
                    source=f"megayak/{ds}",
                    note=note + "; CC0",
                )
            )
    # DavutKursun's XGBoost-on-GLM bag (notebook output, pinned by goodpjw2008, Apache-2.0); ships its folds.
    d = PUB / "goodpjw2008__s6e11-board-model-pinned-inputs"
    if d.exists():
        out.append(
            dict(
                tag="dk",
                name="xgb_glm_bag",
                oof=(d / "davutkursun_xgb_glm_bag_oof.csv", "pred"),
                test=(d / "davutkursun_xgb_glm_bag_test.csv", "pred"),
                auc=0.95069,
                folds=(d / "davutkursun_xgb_glm_bag_oof.csv", "fold"),
                source="davutkursun/s6e11-one-model-no-public-inputs (output pinned by goodpjw2008)",
                note="XGBoost started from a GLM, bagged; Apache-2.0",
            )
        )
    return out


def load_array(src: Path | tuple[Path, str] | tuple[Path, int]) -> np.ndarray:
    """A .npy file, one array of an .npz archive as (path, key), one column of a 2-D .npy as (path, col),
    or one column of a CSV as (path, column).

    Never unpickles.
    """
    if isinstance(src, tuple) and isinstance(src[1], int):
        return np.asarray(np.load(src[0], mmap_mode="r", allow_pickle=False)[:, src[1]])
    if isinstance(src, tuple) and src[0].suffix == ".csv":
        return pd.read_csv(src[0])[src[1]].to_numpy()
    if isinstance(src, tuple):
        with np.load(src[0], allow_pickle=False) as z:
            return z[src[1]]
    return np.load(src, allow_pickle=False)


def load_folds(p: Path | tuple[Path, str]) -> np.ndarray:
    """A source's fold ids in train.csv order."""
    if isinstance(p, tuple) or p.suffix == ".npy":
        return load_array(p).astype(np.int8)
        return np.load(p, allow_pickle=False).astype(np.int8)
    return pd.read_csv(p)["fold"].to_numpy().astype(np.int8)


def slug(s: str) -> str:
    """Lower-case run-name fragment."""
    return "".join(ch if ch.isalnum() else "_" for ch in s.lower()).strip("_")[:40]


def main() -> None:
    tr, te, _ = load()
    y = tr[TARGET].to_numpy()
    ours = our_fold_ids(y)
    ok = 0
    for m in members():
        label = f"{m['tag']}:{m['name']}"
        try:
            oof = load_array(m["oof"]).astype("float64").ravel()
            test = load_array(m["test"]).astype("float64").ravel()
        except (OSError, ValueError, KeyError) as e:
            print(f"  skip {label}: {e}")
            continue
        if m.get("rank_oof"):
            oof = (rankdata(oof) - 0.5) / len(oof)
        if oof.shape != (len(tr),) or test.shape != (len(te),):
            print(f"  skip {label}: shapes {oof.shape} {test.shape}")
            continue
        if not (np.isfinite(oof).all() and np.isfinite(test).all()):
            print(f"  skip {label}: non-finite values")
            continue
        if m["folds"] not in (None, STATED) and not np.array_equal(load_folds(m["folds"]), ours):
            print(f"  skip {label}: folds differ from ours")
            continue
        auc = roc_auc_score(y, oof)
        if m["auc"] is not None and abs(auc - m["auc"]) > 1e-4:
            print(f"  skip {label}: AUC {auc:.5f} but source reports {m['auc']:.5f} (row order?)")
            continue
        if m.get("fold_aucs") is not None:
            got = [roc_auc_score(y[ours == k], oof[ours == k]) for k in range(5)]
            if max(abs(a - b) for a, b in zip(got, m["fold_aucs"], strict=True)) > 1e-4:
                print(f"  skip {label}: fold AUCs {[round(g, 5) for g in got]} differ from the source's")
                continue
        run = f"pub_{m['tag']}_{slug(m['name'])}"
        d = RUNS / run
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "oof.npy", oof)
        np.save(d / "test.npy", test)
        meta = dict(
            source=m["source"],
            member=m["name"],
            note=m["note"],
            cv=auc,
            folds=(
                ("stated, fold AUCs match" if m.get("fold_aucs") else "stated")
                if m["folds"] == STATED
                else ("checked" if m["folds"] is not None else "unknown")
            ),
        )
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        ok += 1
        print(f"  {run:44s} {auc:.5f}  folds {meta['folds']}")
    print(f"imported {ok} members")


if __name__ == "__main__":
    main()
