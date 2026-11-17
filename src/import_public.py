"""Import public OOF + test predictions as runs, after checking they line up with our folds.

Sources live in data/public/<owner>__<dataset>/ (downloaded with `kaggle datasets download`, never
committed). Each member becomes runs/pub_<tag>_<name>/{oof.npy,test.npy,meta.json} and can then be
stacked with our own runs by stack.py. Checks before import:

  * 699,960 OOF and 299,984 test values, all finite (files loaded with allow_pickle=False);
  * the source's fold file equals our folds (`common.folds`) row by row, when the source ships one;
  * the OOF AUC we compute matches the AUC the source reports, within 0.0001.

A member that fails any check is skipped with the reason printed. Usage: .venv/bin/python src/import_public.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import DATA, RUNS, TARGET, folds, load

PUB = DATA / "public"


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
                        folds=None,
                        source="alexchilton/s6e11-oof-predictions",
                        note=f"{m.get('version', '')} {h}",
                    )
                )
    return out


def load_folds(p: Path) -> np.ndarray:
    """A source's fold ids in train.csv order."""
    if p.suffix == ".npy":
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
            oof = np.load(m["oof"], allow_pickle=False).astype("float64").ravel()
            test = np.load(m["test"], allow_pickle=False).astype("float64").ravel()
        except (OSError, ValueError) as e:
            print(f"  skip {label}: {e}")
            continue
        if oof.shape != (len(tr),) or test.shape != (len(te),):
            print(f"  skip {label}: shapes {oof.shape} {test.shape}")
            continue
        if not (np.isfinite(oof).all() and np.isfinite(test).all()):
            print(f"  skip {label}: non-finite values")
            continue
        if m["folds"] is not None and not np.array_equal(load_folds(m["folds"]), ours):
            print(f"  skip {label}: folds differ from ours")
            continue
        auc = roc_auc_score(y, oof)
        if m["auc"] is not None and abs(auc - m["auc"]) > 1e-4:
            print(f"  skip {label}: AUC {auc:.5f} but source reports {m['auc']:.5f} (row order?)")
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
            folds_checked=m["folds"] is not None,
        )
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        ok += 1
        print(f"  {run:44s} {auc:.5f}  folds {'checked' if m['folds'] is not None else 'unknown'}")
    print(f"imported {ok} members")


if __name__ == "__main__":
    main()
