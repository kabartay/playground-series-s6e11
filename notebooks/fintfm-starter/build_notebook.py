"""Generate fintfm-starter-s6e11.ipynb from the cell sources below.

Run with any Python that has nbformat:  python build_notebook.py
The notebook itself runs on Kaggle (CPU or GPU) or locally; FINTFM_SMOKE=1 shrinks every size for a quick check.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
cells: list[nbf.NotebookNode] = []


def md(text: str) -> None:
    """Append a markdown cell; a "## N · Title" heading gets an anchor `sN` for the contents links."""
    text = text.strip("\n")
    m = re.match(r"## (\d+) · ", text)
    if m:
        text = f'<a id="s{m.group(1)}"></a>\n\n' + text
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    """Append a code cell."""
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# FinTFM Starter: In-Context Learning on Credit Card Fraud

*Playground Series, Season 6 Episode 11: Predicting Credit Card Fraud*

This notebook predicts card fraud with **[FinTFM](https://github.com/kabartay/fintfm)**, a small tabular
foundation model, and uses the competition as a classroom: what "in-context learning" means for a table,
which settings matter, and how far it gets against a gradient-boosted tree on exactly the same split.

There is **no feature engineering**. Every number below is measured in this notebook, on a stratified
holdout carved out of `train.csv`, and the final cell writes `submission.csv` with the best setting found.

**What to expect.** FinTFM is a research model. Its own README states plainly that on real credit panels
gradient boosting still wins. Here the interesting questions are *how much context it needs*, *which
rows to show it*, and *whether it adds anything to a tree model*. The tables answer them.

**New in version 2 (fintfm 1.0.0).** FinTFM's final release adds a *trained linear readout*
(`fintfm.readout`, from 0.13.0): instead of the network's own head, a logistic regression on the network's
internal representation of each row decides. It is the configuration behind FinTFM's TabArena-Lite entry.
Section 9 measures it on the same holdout. The weights are unchanged, so sections 3 to 8 reproduce
version 1.

| | |
|---|---|
| **Code** | [github.com/kabartay/fintfm](https://github.com/kabartay/fintfm), Apache-2.0 |
| **Weights** | [kabartay/fintfm-binary](https://huggingface.co/kabartay/fintfm-binary), Apache-2.0, 885K parameters |
| **Install** | `pip install fintfm==1.0.0` (final release; development has ended) |
| **Inspired by** | [TabPFN-3.5 Starter](https://www.kaggle.com/code/philippsinger/tabpfn-3-5-starter-playground-series-s6e11) by Psi |

**Contents**

1. [Setup](#s1)
2. [The data](#s2)
3. [How FinTFM predicts](#s3)
4. [A reference point](#s4)
5. [How much context?](#s5)
6. [Which rows to show?](#s6)
7. [Partitioned contexts](#s7)
8. [Ensembling](#s8)
9. [A trained linear readout *(new in version 2)*](#s9)
10. [Does it help a tree model?](#s10)
11. [Submission](#s11)
12. [Takeaways](#s12)
""")

md(r"""
## 1 · Setup

Install the released package (pinned) and download the published checkpoint at a fixed revision, checked
by SHA-256 so the weights are exactly the ones the FinTFM project measured. The notebook runs on CPU;
on a GPU session it picks up CUDA automatically and every section runs faster.
""")

code(r"""
import os, subprocess, sys, time, hashlib, warnings
warnings.filterwarnings("ignore")

def pip_install(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

try:
    import fintfm
    assert fintfm.__version__ == "1.0.0"
except (ImportError, AssertionError):
    try:
        pip_install("fintfm==1.0.0")
    except subprocess.CalledProcessError:   # PyPI not yet updated: the same release from its git tag
        pip_install("fintfm @ git+https://github.com/kabartay/fintfm.git@v1.0.0")
import fintfm
print("fintfm", fintfm.__version__)
""")

code(r"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from fintfm.inference import FinancialTFMClassifier
from fintfm.inference.large_n import PartitionedTFMClassifier

SMOKE = os.environ.get("FINTFM_SMOKE") == "1"          # tiny sizes for a quick local check
HOLDOUT_ROWS = 400 if SMOKE else 20_000                 # rows scored in every experiment
TEST_LIMIT = 600 if SMOKE else None                     # None = the full test set
BUDGET_HOURS = 11.0                                     # Kaggle stops a session at 12 h
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count())
T0 = time.time()
TARGET = "is_fraud"

CKPT = hf_hub_download("kabartay/fintfm-binary", "v4-cellattn-labels.pt",
                       revision="f116bfd43a2b15c65ed3551ea8c38e3364629ddc")
assert hashlib.sha256(Path(CKPT).read_bytes()).hexdigest() == \
    "d231b062483c9261aae4c503ad5359db295fc7206fa0183f51763e4953c518d2", "checkpoint changed"
print(f"device {DEVICE}, {os.cpu_count()} CPU threads, holdout {HOLDOUT_ROWS:,} rows, smoke={SMOKE}")

# One look for every chart
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
INK, FTFM, TREE, ACCENT = "#1f2933", "#2f6fdb", "#e07a2e", "#16a394"
""")

md(r"""
## 2 · The data

One row is one card transaction; `is_fraud` is 1 for fraud. The data is synthetic, generated from the
[Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026)
dataset. Fraud is rare, which matters for an in-context model: a context of a few thousand random rows
holds only a few dozen fraud examples.
""")

code(r"""
root = min(Path("/kaggle/input").rglob("train.csv")).parent if Path("/kaggle/input").exists() else Path("../../data")
train = pd.read_csv(root / "train.csv")
test = pd.read_csv(root / "test.csv")
if TEST_LIMIT:
    test = test.iloc[:TEST_LIMIT]
y_all = train.pop(TARGET).astype(int).to_numpy()
print(f"train {train.shape[0]:,} rows, test {test.shape[0]:,} rows, {train.shape[1] - 1} features")
print(f"fraud rate {y_all.mean():.4%} ({y_all.sum():,} fraud rows)")
print(f"fraud rows expected in a random context of 2,000: {2000 * y_all.mean():.0f}")
train.head(3)
""")

code(r"""
flags = [c for c in train.columns if train[c].dtype == bool]
cats = [c for c in train.columns if train[c].dtype == object or str(train[c].dtype).startswith("str")]
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
rate = pd.Series({c: y_all[train[c].to_numpy()].mean() for c in flags}).sort_values()
ax[0].barh(rate.index, 100 * rate.values, color=FTFM)
ax[0].axvline(100 * y_all.mean(), color=INK, ls=":", lw=1)
ax[0].set_xlabel("fraud rate when the flag is True, %"); ax[0].set_title("Six warning flags")
by_auth = pd.Series(y_all).groupby(train["auth_method"].to_numpy()).mean().sort_values()
ax[1].barh(by_auth.index, 100 * by_auth.values, color=ACCENT)
ax[1].axvline(100 * y_all.mean(), color=INK, ls=":", lw=1)
ax[1].set_xlabel("fraud rate, %"); ax[1].set_title("By authentication method")
fig.suptitle("Fraud concentrates in a few signals (dotted line: overall rate)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); plt.show()
""")

md(r"""
## 3 · How FinTFM predicts

FinTFM never trains on your data. `fit()` only **stores labelled rows as context**; `predict_proba()`
passes each new row through a transformer that attends to those labelled rows and reads off the fraud
probability, in one forward pass. The model learned *how to read a table* during pretraining, on
synthetic tables only.

Two consequences shape everything below:

- **Context size is the main dial.** Attention cost grows with the square of the context, so the model
  cannot see all 700k rows at once. It sees a subsample, and which rows (and how many fraud rows) are in
  it matters.
- **The checkpoint takes numbers.** Categoricals are one-hot encoded and booleans become 0/1, which gives
  54 columns, within the checkpoint's limit of 136.

We hold out a stratified sample of `train.csv`, give FinTFM the remaining rows as its pool of possible
context, and score every setting on the same holdout.
""")

code(r"""
def to_matrix(*frames):
    both = pd.concat([f.drop(columns="id") for f in frames], keys=range(len(frames)))
    both = pd.get_dummies(both, columns=cats, dtype=np.float32).astype(np.float32)
    return [both.loc[i].to_numpy() for i in range(len(frames))]

X_all, X_test = to_matrix(train, test)
pool_idx, hold_idx = train_test_split(np.arange(len(y_all)), test_size=HOLDOUT_ROWS,
                                      stratify=y_all, random_state=0)
X_pool, y_pool, X_hold, y_hold = X_all[pool_idx], y_all[pool_idx], X_all[hold_idx], y_all[hold_idx]
print(f"{X_all.shape[1]} numeric columns; context pool {len(y_pool):,} rows; "
      f"holdout {len(y_hold):,} rows with {y_hold.sum()} fraud")

RESULTS = []      # one row per experiment

def hours_left():
    return BUDGET_HOURS - (time.time() - T0) / 3600

def evaluate(section, name, make, rows_per_s_guess=None):
    # Fit on the pool, score the holdout, record AUC, AP and speed. Skips if the budget would run out.
    if rows_per_s_guess and len(y_hold) / rows_per_s_guess / 3600 > hours_left() - 5:
        print(f"skip {name}: not enough time left for it and the test predictions")
        return None
    t = time.time()
    p = make().fit(X_pool, y_pool).predict_proba(X_hold)[:, 1]
    s = time.time() - t
    row = dict(section=section, setting=name, auc=roc_auc_score(y_hold, p),
               ap=average_precision_score(y_hold, p), seconds=round(s), rows_per_s=len(y_hold) / s)
    RESULTS.append({**row, "pred": p})
    print(f"{name:42s} AUC {row['auc']:.4f}  AP {row['ap']:.4f}  {s:6.0f} s  ({row['rows_per_s']:.0f} rows/s)")
    return p

def table(section=None):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "pred"} for r in RESULTS])
    return (df if section is None else df[df.section == section]).round(4)

def fintfm(**kw):
    kw.setdefault("max_context", 2000)
    return lambda: FinancialTFMClassifier(CKPT, device=DEVICE, random_state=0, **kw)
""")

md(r"""
## 4 · A reference point

Before tuning FinTFM, the number to beat: a default LightGBM (500 rounds, no tuning) on the same split.
It sees all 680k pool rows, which no in-context model here can.
""")

code(r"""
from lightgbm import LGBMClassifier

# On macOS, torch and LightGBM bring two OpenMP runtimes and multithreaded LightGBM crashes the process;
# Linux (and Kaggle) is fine. One thread there costs only a minute here.
LGBM_JOBS = 1 if sys.platform == "darwin" else -1
lgbm = lambda: LGBMClassifier(n_estimators=500, learning_rate=0.05, verbose=-1, random_state=0, n_jobs=LGBM_JOBS)
p_lgbm = evaluate("reference", "LightGBM, 500 rounds, default", lgbm)
""")

md(r"""
## 5 · How much context?

The same model, uniform random contexts of growing size. This is FinTFM's learning curve: more labelled
rows, and more fraud rows among them, at a growing cost per prediction.
""")

code(r"""
sizes = [250, 500, 1000] if SMOKE else [250, 500, 1000, 2000, 4000]
prev = None                                   # (rows/s, context) of the last run, to guess the next one's speed
for n in sizes:
    guess = prev[0] * prev[1] / n if prev else None
    if evaluate("context size", f"uniform, {n:,} rows", fintfm(max_context=n), guess) is not None:
        RESULTS[-1]["ctx"] = n
        prev = (RESULTS[-1]["rows_per_s"], n)

cs = table("context size").assign(ctx=[r["ctx"] for r in RESULTS if r["section"] == "context size"])
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(cs.ctx, cs.auc, "o-", color=FTFM, label="FinTFM, uniform context")
ax[0].axhline(table("reference").auc.iloc[0], color=TREE, ls="--", label="LightGBM reference")
ax[0].set_xscale("log"); ax[0].set_xlabel("context rows"); ax[0].set_ylabel("holdout ROC-AUC"); ax[0].legend()
ax[0].set_title("Accuracy grows with context")
ax[1].plot(cs.ctx, cs.rows_per_s, "o-", color=ACCENT)
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("context rows"); ax[1].set_ylabel("rows scored per second")
ax[1].set_title(f"…and so does the cost ({DEVICE})")
fig.tight_layout(); plt.show()
cs
""")

md(r"""
## 6 · Which rows to show?

With about 1.7% fraud, a uniform context of 2,000 rows holds roughly 33 fraud rows. FinTFM can build the
context differently:

- **`uniform`**: a random sample (the default).
- **`balanced`**: equal numbers of fraud and non-fraud rows. FinTFM reads class balance from its context,
  so it then corrects the probabilities back to the true base rate (`correct_prior=True`); ranking, and so
  ROC-AUC, is unaffected by that correction.
- **`retrieval`**: for each group of query rows, the most similar labelled rows (nearest neighbours),
  with a minimum number of fraud rows.
- **`hybrid`**: part retrieved neighbours, part random rows.

All at 2,000 context rows, so only the choice of rows changes.
""")

code(r"""
base_speed = next((r["rows_per_s"] for r in RESULTS if r.get("ctx") == 2000), None)
for strategy in ["balanced", "retrieval", "hybrid"]:
    evaluate("context rows", f"{strategy}, 2,000 rows", fintfm(context_strategy=strategy), base_speed)

t_all = table()
strat = t_all[(t_all.section == "context rows") | (t_all.setting == "uniform, 2,000 rows")].sort_values("auc")
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.barh(strat.setting, strat.auc, color=FTFM)
ax.axvline(table("reference").auc.iloc[0], color=TREE, ls="--", label="LightGBM reference")
ax.set_xlim(max(0.5, strat.auc.min() - 0.03), max(strat.auc.max(), table("reference").auc.iloc[0]) + 0.01)
ax.set_xlabel("holdout ROC-AUC"); ax.legend(loc="lower right"); ax.set_title("Same size, different rows")
fig.tight_layout(); plt.show()
strat
""")

md(r"""
## 7 · Partitioned contexts

One context for 700k rows is a compromise. `PartitionedTFMClassifier` first splits the pool with a shallow
decision tree (depth 2, so up to four leaves), then gives each leaf its own FinTFM with a context drawn
from that leaf. A query row is answered by the FinTFM of the leaf it falls in, so its context is made of
rows that look like it. In FinTFM's own measurements this was its strongest zero-shot lever on a credit
benchmark.
""")

code(r"""
def best_fintfm_kw():
    cands = [r for r in RESULTS if r["section"] in ("context size", "context rows") and r.get("ctx", 2000) == 2000]
    best = max(cands, key=lambda r: r["auc"])
    strategy = best["setting"].split(",")[0]
    return {"context_strategy": strategy}, best

kw, best = best_fintfm_kw()
print("partitioning the best 2,000-row setting:", best["setting"])
evaluate("partitioned", f"partitioned (depth 2) + {best['setting']}",
         lambda: PartitionedTFMClassifier(fintfm(**kw)(), max_depth=2, random_state=0), best["rows_per_s"])
table("partitioned")
""")

md(r"""
## 8 · Ensembling

`n_ensemble=4` averages four forward passes, each with its own context draw and feature order. It costs
four times as much; worth it only if the gain shows on the holdout.
""")

code(r"""
evaluate("ensemble", f"{best['setting']}, n_ensemble=4", fintfm(n_ensemble=4, **kw), best["rows_per_s"] / 4)
table("ensemble")
""")

md(r"""
## 9 · A trained linear readout (new in 0.13.0)

Everything so far used the network's own output head, which was trained on synthetic tasks and only
*reads* the context. `fintfm.readout.LinearReadoutModel` keeps the network but **trains a logistic
regression on its internal representation** of each row (with the row's own features beside it), so the
final decision is fitted to this table. Two details keep it honest:

- **Cross-fitting.** The rows the readout learns from are also context rows. Each is represented by the
  network with its own fold of rows left out of the context, so no row ever sees its own label, exactly
  as a test row cannot.
- **Settings by internal cross-validation** on the training rows only; nothing is chosen on our holdout.

`entry_defaults("binary")` is the measured configuration of FinTFM's TabArena-Lite entry (8 ensemble
members, 1,000 context rows, partitioned contexts whose depth scales with the row count). It is fitted on
the whole pool: the readout itself represents at most 100,000 of those rows (its internal budget), so the
fit takes about 15 minutes on Kaggle CPU whatever the pool size. Measured while preparing this version, on
the same holdout: fitted on 20,000 rows 0.9112, on 100,000 rows 0.9295, on 300,000 rows 0.9308.
""")

code(r"""
from fintfm.readout import LinearReadoutModel, entry_defaults

RO = entry_defaults("binary"); RO_CKPT = RO.pop("checkpoint")
READOUT_FIT_ROWS = 2_000 if SMOKE else None                 # None: every row it is given
RO_SPEED_GUESS = None if SMOKE else 135                      # holdout rows/s, measured on Kaggle CPU

class Readout:
    # LinearReadoutModel, optionally fitted on a random sample of the rows it is given
    def __init__(self, rows=None):
        self.rows = rows
    def fit(self, X, y):
        i = np.arange(len(y)) if self.rows is None or self.rows >= len(y) else \
            np.random.default_rng(0).choice(len(y), self.rows, replace=False)
        self.m = LinearReadoutModel("binary", RO_CKPT, device=DEVICE, random_state=0, **RO).fit(X[i], y[i])
        return self
    def predict_proba(self, X):
        return self.m.predict_proba(X)

ro_name = "linear readout, whole pool" if READOUT_FIT_ROWS is None else f"linear readout, {READOUT_FIT_ROWS:,} rows"
evaluate("readout", ro_name, lambda: Readout(READOUT_FIT_ROWS), RO_SPEED_GUESS)
table("readout")
""")

md(r"""
## 10 · Does it help a tree model?

A weaker model can still improve a stronger one if its mistakes are different. We blend the best FinTFM
predictions with LightGBM's on the holdout, averaging their **ranks** (ROC-AUC only sees order), with the
FinTFM share going from 0 to 1.

One caution: the blend weight is chosen on the same holdout it is scored on, so a small gain here is an
upper bound, not a promise.
""")

code(r"""
from scipy.stats import rankdata
ftfm_rows = [r for r in RESULTS if r["section"] != "reference"]
best_row = max(ftfm_rows, key=lambda r: r["auc"])
rk = lambda v: rankdata(v) / len(v)
w = np.linspace(0, 1, 21)
blend = [roc_auc_score(y_hold, (1 - a) * rk(p_lgbm) + a * rk(best_row["pred"])) for a in w]
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(w, blend, "o-", color=ACCENT)
ax.set_xlabel("FinTFM share of the rank blend"); ax.set_ylabel("holdout ROC-AUC")
ax.set_title(f"LightGBM + {best_row['setting']}")
fig.tight_layout(); plt.show()
print(f"LightGBM alone {blend[0]:.4f}; best blend {max(blend):.4f} at FinTFM share {w[int(np.argmax(blend))]:.2f}; "
      f"FinTFM alone {blend[-1]:.4f}")
print(f"rank correlation of the two models on the holdout: {np.corrcoef(rk(p_lgbm), rk(best_row['pred']))[0, 1]:.3f}")
""")

md(r"""
## 11 · Submission

The best FinTFM setting on the holdout, now with the whole of `train.csv` as its context pool, predicts
the test set. If the remaining time would not cover the test set at that setting's measured speed, the
cell falls back to the fastest setting that fits.
""")

code(r"""
def make_from(row):
    s = row["setting"]
    if s.startswith("linear readout"):
        return lambda: Readout(READOUT_FIT_ROWS)
    if s.startswith("partitioned"):
        return lambda: PartitionedTFMClassifier(fintfm(**kw)(), max_depth=2, random_state=0)
    if "n_ensemble" in s:
        return fintfm(n_ensemble=4, **kw)
    n = row.get("ctx", 2000)
    return fintfm(max_context=n, context_strategy=s.split(",")[0])

ranked = sorted(ftfm_rows, key=lambda r: -r["auc"])
need = lambda r: len(X_test) / r["rows_per_s"] / 3600 * 1.15
choice = next((r for r in ranked if need(r) < hours_left()), min(ranked, key=need))
print(f"submitting: {choice['setting']} (holdout AUC {choice['auc']:.4f}); "
      f"expected {need(choice):.1f} h, {hours_left():.1f} h left")

t = time.time()
p_test = make_from(choice)().fit(X_all, y_all).predict_proba(X_test)[:, 1]
sub = pd.DataFrame({"id": test["id"], TARGET: p_test})
sub.to_csv("submission.csv", index=False)
print(f"wrote submission.csv: {len(sub):,} rows in {time.time() - t:.0f} s")
sub.head()
""")

md(r"""
## 12 · Takeaways

The full results table, sorted. Read it with the holdout size in mind: with about 330 fraud rows in
20,000, differences below roughly 0.01 ROC-AUC are within noise.
""")

code(r"""
summary = table().sort_values("auc", ascending=False).reset_index(drop=True)
print(f"total run time {(time.time() - T0) / 3600:.1f} h on {DEVICE}")
summary
""")

md(r"""
What this notebook shows, in general terms (the numbers are in the table above):

1. **For the in-context head, context is the main dial**, in size and in which rows are chosen. On a 1.7%
   fraud rate a random context holds very few fraud examples, so how the context is built matters as much
   as how big it is.
2. **A trained readout beats every in-context setting.** Fitting the final decision to this table, on the
   network's representation, gains more than any choice of context. It is also cheaper at prediction time.
3. **Even an untuned gradient-boosted tree remains the stronger single model here**, as FinTFM's own
   README says for real credit panels. It learns from every training row; FinTFM's network sees a thousand
   at a time.
4. **The useful question is complementarity**: section 10 measures whether FinTFM's different view of the
   data adds anything to a tree model.

**Where to go from here:** FinTFM's development ended with 1.0.0, so these are things to try with the
released package: the readout as one member of a stacked ensemble, with out-of-fold predictions on the
same folds as the trees; the readout with more ensemble members on a GPU session; and
`context_positive_floor` to fix the number of fraud rows in an in-context run.

*FinTFM's code and measurement log are at [github.com/kabartay/fintfm](https://github.com/kabartay/fintfm);
the log records every result, including the negative ones. Questions and comments welcome.*
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, HERE / "fintfm-starter-s6e11.ipynb")
print(f"wrote {len(cells)} cells")
