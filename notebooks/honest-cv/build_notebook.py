"""Generate s6e11-own-models-honest-cv.ipynb from the cell sources below and the modules in src/.

Run with any Python that has nbformat:  python build_notebook.py
The notebook trains six of our own models on Kaggle CPU (about 2.5 h), stacks them with a nested logistic
regression, checks the zero-fraud zones inside the folds and writes submission.csv. A model whose
runs/<name>/oof.npy already exists is not retrained, so a local check can point S6E11_RUNS at the repo's
runs/ and S6E11_DATA at data/.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"
REPO = "https://github.com/kabartay/playground-series-s6e11"
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


md(rf"""
# S6E11 | Own Models Only | Honest 5-Fold CV

*Playground Series, Season 6 Episode 11: Predicting Credit Card Fraud (ROC-AUC)*

This notebook trains **six models we wrote ourselves**, stacks them, and checks every step on the same
five folds. It uses no public submission files and no public out-of-fold predictions, and nothing is
chosen on the public leaderboard. It is not the top of the board, and it does not try to be: it is a
pipeline whose cross-validation you can trust, built from parts you can reuse.

**Why bother.** The public leaderboard scores 20% of the test rows, about 60,000 rows with about a
thousand frauds. Section 7 measures how much an AUC on that many rows moves by chance; at the top of
the board the steps are smaller than that. The private leaderboard (the other 80%) decides the ranking,
so here the five-fold CV decides every choice.

**What is in it.**

| Section | What it shows |
| --- | --- |
| [1 · Setup](#s1) | the library (the same code as the [GitHub repo]({REPO}), MIT) |
| [2 · Data and metric](#s2) | class balance, the two scores that carry most of the signal |
| [3 · Folds, and the one rule](#s3) | everything learned from labels stays inside the training fold |
| [4 · Three GLMs](#s4) | target-encoded value keys, generator artefacts, GPT-2 token keys |
| [5 · Two shallow boosters](#s5) | 4-leaf LightGBM, and LightGBM started from the GLM ("GLM boost") |
| [6 · RealMLP](#s6) | a network on value-bucket embeddings |
| [7 · The stack](#s7) | nested logistic regression, and how noisy the public board is |
| [8 · Zero-fraud zones](#s8) | four regions with no fraud, checked with their edges refitted in each fold |
| [9 · What did not help](#s9) | measured in the repo, with numbers |
| [10 · Submission](#s10) | |

Run time on Kaggle CPU: about 2.5 hours, most of it the boosters and the RealMLP.
""")

md(r"""
## 1 · Setup

The models come from one training script, `train_gbdt.py`, with its feature builders in `features.py`
and the data loading and folds in `common.py`. They are written to disk below exactly as they are in
the repo, so every model in this notebook is one command line. `tiktoken` (MIT) gives GPT-2's tokeniser
for one feature family; `pytabkit` (Apache-2.0) provides RealMLP.
""")

code(r"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken", "pytabkit"], check=True)
import os
os.makedirs("src", exist_ok=True)
os.environ["S6E11_THREADS"] = "4"
""")

for module in ("common.py", "features.py", "train_gbdt.py"):
    code(f"%%writefile src/{module}\n" + (SRC / module).read_text())

code(r"""
import json, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
from common import RUNS, TARGET, folds, load

tr, te, og = load()
y = tr[TARGET].to_numpy()
FOLDS = folds(y)
print(f"train {tr.shape}, test {te.shape}, original {og.shape}")
""")

md(r"""
## 2 · Data and metric

700k synthetic training rows, generated from the 20,000-row
[Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026)
dataset (CC0). Fraud is rare, so ROC-AUC is decided by how the 1.7% of frauds rank against the rest.
""")

code(r"""
print(f"fraud rate: train {y.mean():.4f} ({y.sum():,} frauds), original {og[TARGET].mean():.4f}")
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
for a, c in zip(ax, ["merchant_risk_score", "velocity_score"]):
    r = tr.groupby(pd.qcut(tr[c], 10, duplicates="drop"), observed=True)[TARGET].mean()
    a.bar(range(len(r)), 100 * r.to_numpy(), color="#3b6ea8")
    a.set_xticks(range(len(r)), [f"{iv.right:.0f}" for iv in r.index], fontsize=8)
    a.set_title(f"fraud rate (%) by decile of {c}"); a.set_xlabel("decile upper edge")
plt.tight_layout(); plt.show()
flags = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
         "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
print(pd.DataFrame({f: tr.groupby(f)[TARGET].mean() for f in flags}).T.rename(columns={0: "flag 0", 1: "flag 1"}).round(4))
""")

md(r"""
Fraud rises smoothly with both scores and with every flag. No single column is decisive, so the work is
in how each column's values are encoded. Two facts about the synthetic data shape the features below:

- **Exact values repeat.** The generator copies values from the original rows, so a value's own fraud
  rate (a target encoding of the exact value, its integer part, its decimals) carries signal.
- **Pooled artefacts.** Whether a value occurs in the original data at all, and how often it occurs in
  train + test, also carries signal; per-value encodings cannot see it, because each rare value is
  smoothed towards the base rate on its own. These are unsupervised features (`--fe art`).
""")

md(r"""
## 3 · Folds, and the one rule

`StratifiedKFold(5, shuffle=True, random_state=42)` on `is_fraud`, the split most public out-of-fold
files also use. **Everything learned from labels is learned inside the training fold**:

- target encodings are cross-fitted inside the training fold (5 inner folds), never on the scored fold;
- boosters pick their number of rounds on a 10% holdout of the training fold, then refit on all of it
  (`--es inner`). Early-stopping on the scored fold, which many notebooks do, inflated our boosters' CV
  by 0.00004 to 0.00009 each (measured in the repo);
- the stacker in section 7 is itself cross-validated (nested).

`run()` trains one model with `train_gbdt.py` and prints its fold scores.
""")

code(r"""
RESULTS = {}

def run(name, args):
    # train one model unless its predictions already exist, then report its fold AUCs
    d = RUNS / name
    if not (d / "oof.npy").exists():
        t0 = time.time()
        cmd = [sys.executable, "train_gbdt.py", "--name", name, *args.split()]
        p = subprocess.run(cmd, cwd="src", capture_output=True, text=True)
        if p.returncode:
            print(p.stdout[-3000:], p.stderr[-3000:]); raise RuntimeError(name)
        print(f"{name}: trained in {(time.time() - t0) / 60:.0f} min")
    m = json.loads((d / "meta.json").read_text())
    RESULTS[name] = dict(cv=m["cv"], folds=m["folds"])
    print(f"{name}: CV {m['cv']:.5f}  folds {np.round(m['folds'], 5).tolist()}")
""")

md(r"""
## 4 · Three GLMs

A logistic regression is a strong model here once each value is encoded well. The design: cubic splines
(8 knots) on every numeric column, one-hot categories, and in-fold target encodings of many **keys**:
each column's exact value, its integer part, its value rounded to tens, its decimal ending, and (with
`--keys full`) its leading digits, its value at 0.1 resolution and the length of its printed value.

**4a. Raw columns and the value keys** (`--fe raw --keys full`, one smoothing weight).
""")

code(r"""run("glm_keys", '--model glm --fe raw --te single --keys full --te-smooth 20 --orig 0 --params {"splines":8}')""")

md(r"""
**4b. Add the pooled artefacts, three smoothing weights and joint keys.** Each key is encoded with
m-estimate weights 5, 20 and 100 (rare values trusted less or more), and eight column pairs whose effect
is not additive become joint keys (the pairs measured and published in megayak's notebook
*S6E11 | Per-Value + Digit TE | Shallow-Tree Stack*, Apache-2.0).
""")

code(r"""run("glm_art", '--model glm --fe art --te single --keys full --te-smooth 5,20,100 --orig 0 --joint glm --params {"splines":8}')""")

md(r"""
**4c. Add GPT-2 token keys.** The language-model notebooks at the top of this competition read each row
as text. Part of what they see is how GPT-2's tokeniser splits a number: `1371` becomes `" 13"` and
`"71"`. Here the first two tokens of each float column's integer part, and its fixed 3-digit chunks,
become keys for the same target encoding (idea after aphoristicartist's and busyaprime's notebooks;
written independently). The GLM gets part of the language models' view at a fraction of the cost.
""")

code(r"""run("glm_tok", '--model glm --fe art --te single --keys tokens --te-smooth 5,20,100 --orig 0 --joint glm --params {"splines":8}')""")

md(r"""
## 5 · Two shallow boosters

Deep trees overfit the encodings; 4-leaf trees with a low learning rate do best. Both boosters see the
same encoded keys plus the log count of every key's value over train + test (`--key-counts`).

**5a. LightGBM, 4 leaves.**
""")

code(r"""
LGB = '{"num_leaves":4,"learning_rate":0.03,"feature_fraction":0.5,"min_child_samples":100,"lambda_l2":1.0,"n_estimators":20000}'
run("lgb_l4", f"--model lgb --fe art --te single --keys full --te-smooth 5,20,100 --orig 0 --joint glm --es inner --key-counts --params {LGB}")
""")

md(r"""
**5b. GLM boost.** The same trees, but started from an in-fold GLM's logit (`--offset glm`): the GLM
carries the smooth part, the trees only learn the interactions it misses. This is our best single model.
""")

code(r"""run("glm_boost", f"--model lgb --fe art --te single --keys full --te-smooth 5,20,100 --orig 0 --joint glm --es inner --key-counts --offset glm --params {LGB}")""")

md(r"""
## 6 · RealMLP on value-bucket embeddings

A different view for the stack: no target encodings at all. Every numeric column's integer part and its
two decimals become categories, and RealMLP (pytabkit) learns an embedding per value bucket
(`--fe emb`). Hyperparameters start from yekenot's tuned RealMLP notebook, with 8 ensemble members
instead of 16 for CPU, 3 epochs and no early stopping (the scored fold never picks the model).
""")

code(r"""
RMLP = ('{"n_ens":8,"n_epochs":3,"batch_size":288,"use_early_stopping":false,"val_metric_name":"1-auc_ovr",'
        '"lr":0.073,"wd":0.038,"sq_mom":0.988,"lr_sched":"flat_anneal","wd_sched":"cos_log_15",'
        '"first_layer_lr_factor":0.25,"embedding_size":5,"max_one_hot_cat_size":16,"hidden_sizes":[512,256,128],'
        '"act":"silu","p_drop":0.05,"p_drop_sched":"pow5","plr_hidden_1":16,"plr_hidden_2":8,"plr_act_name":"gelu",'
        '"plr_lr_factor":0.1151,"plr_sigma":0.75,"ls_eps":0.01,"ls_eps_sched":"sqrt_cos","add_front_scale":false,'
        '"bias_init_mode":"neg-uniform-dynamic-2","tfms":["one_hot","median_center","robust_scale","smooth_clip",'
        '"embedding","l2_normalize"]}')
run("realmlp_emb", f"--model realmlp --fe emb --te none --orig 0 --params {RMLP}")
""")

code(r"""
table = pd.DataFrame({k: [v["cv"], *v["folds"]] for k, v in RESULTS.items()},
                     index=["CV", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4"]).T
table.round(5)
""")

md(r"""
## 7 · The stack

A logistic regression on the six models' logits. It is **nested**: for each outer fold it is fitted on
the other four folds' out-of-fold predictions and scored on the fifth, so its CV is as honest as the
members'. Regularisation C = 0.1.
""")

code(r"""
names = list(RESULTS)
def to_logit(oof, test):
    # the GLMs and the GLM boost save logits already (values outside [0, 1]); the others save probabilities
    if oof.min() < 0 or oof.max() > 1:
        return oof, test
    f = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return f(oof), f(test)
pairs = [to_logit(np.load(RUNS / n / "oof.npy"), np.load(RUNS / n / "test.npy")) for n in names]
Z = np.column_stack([o for o, _ in pairs])
Zt = np.column_stack([t for _, t in pairs])

stack_oof = np.zeros(len(y))
for i_tr, i_va in FOLDS:
    lr = LogisticRegression(C=0.1, max_iter=2000).fit(Z[i_tr], y[i_tr])
    stack_oof[i_va] = lr.decision_function(Z[i_va])
stack_folds = [roc_auc_score(y[i], stack_oof[i]) for _, i in FOLDS]
best = max(RESULTS, key=lambda k: RESULTS[k]["cv"])
print(f"best single ({best}): {RESULTS[best]['cv']:.5f}")
print(f"nested stack:        {roc_auc_score(y, stack_oof):.5f}  folds {np.round(stack_folds, 5).tolist()}")
lr = LogisticRegression(C=0.1, max_iter=2000).fit(Z, y)
stack_test = lr.decision_function(Zt)
print("weights:", dict(zip(names, np.round(lr.coef_[0], 3))))
""")

md(r"""
**How noisy is the public leaderboard?** Draw 60,000 rows (the size of the public part of the test set)
from the stack's out-of-fold predictions many times and look at how much the AUC moves.
""")

code(r"""
rng = np.random.default_rng(0)
single = np.load(RUNS / best / "oof.npy")
level, diff = [], []
for _ in range(300):
    s = rng.choice(len(y), 60_000, replace=False)
    a = roc_auc_score(y[s], stack_oof[s]); level.append(a)
    diff.append(a - roc_auc_score(y[s], single[s]))
full_gain = roc_auc_score(y, stack_oof) - RESULTS[best]["cv"]
print(f"stack AUC on 60,000 random rows: standard deviation {np.std(level):.5f}")
print(f"stack minus best single, all 700k rows: {full_gain:+.5f}")
print(f"stack minus best single, 60,000 rows:   mean {np.mean(diff):+.5f}, standard deviation {np.std(diff):.5f}, "
      f"below zero in {np.mean(np.array(diff) < 0):.0%} of draws")
""")

md(r"""
Two kinds of noise. The **level** of one file's score moves by the first standard deviation between the
public 20% and the private 80%, so a public score says little about the private one at the third
decimal. The **difference** between two files scored on the same rows moves much less, because both see
the same frauds; but even a real, all-folds gain like the stack's over its best member is sometimes
reversed on 60,000 rows. Steps of 0.00001 to 0.0001, as at the top of the board, are well inside that
noise. Five folds of 700,000 rows are not.
""")

md(r"""
## 8 · Zero-fraud zones

Public notebooks (daoviet, goodpjw2008, Weni Mohamed) found four regions with no fraud among the
training rows:

- zone 1: `merchant_risk_score < 13.1`
- zone 2: `velocity_score <= 4.1` and `merchant_risk_score <= 20`
- zone 3: all eight risk flags clear (`cvv_retry_count` and `prior_disputes` 0, the six binary flags 0),
  `velocity_score <= 4.1` and `merchant_risk_score <= 33.8`
- zone 4: all eight flags clear, `velocity_score <= 6.0` and `merchant_risk_score <= 25`

Sending these rows to the bottom of the ranking helps only if they hold no fraud on unseen rows too.
They were found on all training rows, so scoring them on the same rows would flatter them. Here each
merchant-risk edge is refitted on the four training folds (just below the lowest fraud inside the zone,
never above the published edge) and applied to the held-out fold. The velocity edges stay as published,
a small optimism this check does not remove.
""")

code(r"""
def clean(d):
    ok = (d["cvv_retry_count"].to_numpy() == 0) & (d["prior_disputes"].to_numpy() == 0)
    for f in flags:
        ok &= d[f].to_numpy().astype(int) == 0
    return ok

ZONES = [  # (condition before the merchant-risk edge, published edge, strict <)
    (lambda d: np.ones(len(d), bool), 13.1, True),
    (lambda d: d["velocity_score"].to_numpy() <= 4.1, 20.0, False),
    (lambda d: clean(d) & (d["velocity_score"].to_numpy() <= 4.1), 33.8, False),
    (lambda d: clean(d) & (d["velocity_score"].to_numpy() <= 6.0), 25.0, False),
]

def zone_rows(d_fit, y_fit, d_app):
    # rows of d_app inside any zone, with each merchant-risk edge refitted on (d_fit, y_fit)
    z = np.zeros(len(d_app), bool)
    m_fit, m_app = d_fit["merchant_risk_score"].to_numpy(), d_app["merchant_risk_score"].to_numpy()
    for cond, pub, strict in ZONES:
        fraud = cond(d_fit) & (y_fit == 1)
        low = m_fit[fraud].min() if fraud.any() else np.inf
        edge, s = (low, True) if low <= pub else (pub, strict)
        z |= cond(d_app) & ((m_app < edge) if s else (m_app <= edge))
    return z

zoned = stack_oof.copy(); n_rows = n_fraud = 0
for i_tr, i_va in FOLDS:
    z = zone_rows(tr.iloc[i_tr], y[i_tr], tr.iloc[i_va])
    n_rows += z.sum(); n_fraud += y[i_va][z].sum()
    zoned[i_va[z]] = stack_oof.min() - 1
zoned_folds = [roc_auc_score(y[i], zoned[i]) for _, i in FOLDS]
print(f"held-out rows in a zone: {n_rows:,}, frauds among them: {n_fraud}")
print(f"stack:         {roc_auc_score(y, stack_oof):.5f}  folds {np.round(stack_folds, 5).tolist()}")
print(f"stack + zones: {roc_auc_score(y, zoned):.5f}  folds {np.round(zoned_folds, 5).tolist()}")
print(f"folds up: {sum(a > b for a, b in zip(zoned_folds, stack_folds))} of 5")
""")

md(r"""
A small gain, but on every fold, and no held-out fraud falls into a refitted zone. The stack already
ranks these rows low; the zones only finish the job.
""")

md(r"""
## 9 · What did not help

Measured in the repo on our full 24-model own stack (nested CV 0.95138), with the rule that a change is
kept only if the stack improves on at least 4 of the 5 folds. Details and every other run:
[docs/DESIGN.md](https://github.com/kabartay/playground-series-s6e11/blob/main/docs/DESIGN.md).

| Idea | Result |
| --- | --- |
| Our own target-first GPT-2 (rows as text), 1 epoch | 0.94396 alone; stack weight about 0, stack 0.95134 |
| The same, 3 epochs on 5 GPUs | 0.94717 alone (+0.003 on every fold); stack 0.95131, 2 of 5 folds up |
| Seed bags of the RealMLPs (3 seeds) | each member +0.0001; stack 0.95139, 1 of 5 folds up |
| RealMLPs trained on all rows instead of a 90% split | members +0.0004; stack 0.95137 |
| A residual LightGBM on raw features, started from the stack | stops after 1 to 7 rounds; 0.95137 |
| The 20,000 original rows added to training | LightGBM -0.0001 (0.94680 against 0.94690) |
| Boosters early-stopped on the scored fold | CV inflated by 0.00004 to 0.00009 per booster |

The pattern: once the stack has a GLM, a GLM boost and a network, a new member has to see something none of
them sees. More of the same, or a stronger version of the same, does not move it.
""")

md(r"""
## 10 · Submission

The stack's test logit, with the zone rows (edges refitted on all training rows) sent to the bottom.
""")

code(r"""
z_test = zone_rows(tr, y, te)
p = 1 / (1 + np.exp(-stack_test))
p[z_test] *= 1e-6
sub = pd.DataFrame({"id": te["id"], TARGET: p})
sub.to_csv("submission.csv", index=False)
print(f"test rows in a zone: {z_test.sum():,} of {len(te):,}")
sub.head()
""")

md(rf"""
## Credits

- Data: the competition, and [Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026) (CC0).
- Joint pairs (section 4b): megayak, *S6E11 | Per-Value + Digit TE | Shallow-Tree Stack* (Apache-2.0).
- GPT-2 token keys (4c): the idea after aphoristicartist's and busyaprime's notebooks.
- RealMLP settings and value-bucket embeddings (6): yekenot's RealMLP notebook.
- Zero-fraud zones (8): daoviet, goodpjw2008, Weni Mohamed.

All code here is ours, from [{REPO.removeprefix("https://")}]({REPO}) (MIT), which also logs every run, including the
ones that failed. If something here helps you, an upvote is appreciated.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = HERE / "s6e11-own-models-honest-cv.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
