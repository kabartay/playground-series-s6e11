# Predicting Credit Card Fraud: Kaggle Playground S6E11

[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![LightGBM 4.7](https://img.shields.io/badge/LightGBM-4.7-blue.svg)
![XGBoost 3.4](https://img.shields.io/badge/XGBoost-3.4-blue.svg)
![CatBoost 1.2](https://img.shields.io/badge/CatBoost-1.2-blue.svg)
![Validation: 5-fold CV](https://img.shields.io/badge/validation-5--fold%20CV-informational.svg)

A personal entry to the Kaggle competition
[Playground Series, Season 6 Episode 11: Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11),
developed in the open while the competition runs.

Given 24 attributes of a card transaction (amount, timing, velocity, distance from home, merchant risk,
authentication method, device, VPN use and other flags), predict the probability that it is fraudulent.
About 1.7% of the 699,960 training transactions are fraud. Submissions are scored by ROC-AUC on 299,984
test transactions; the competition closes on 2026-11-30.

> **Status (2026-11-16):** best stack 5-fold CV ROC-AUC **0.95079** (nested CV); best single model 0.95062
> (GLM boost: 4-leaf LightGBM started from an in-fold logistic model on target encodings of exact values
> and digit keys, 3 seeds). Last submission: CV 0.95079, public 0.95113. Round six (CatBoost GLM boost
> seeds and depth, a slower MLP) is running.

## Contents

- [Approach](#approach)
- [Results](#results)
- [What worked and what did not](#what-worked-and-what-did-not)
- [FinTFM](#fintfm)
- [Quick start](#quick-start)
- [Commands](#commands)
- [Validation](#validation)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Licences and data](#licences-and-data)
- [Acknowledgements](#acknowledgements)

## Approach

```
train.csv (699,960 rows) ──► 5 fixed stratified folds ──► for each fold:
                                                            in-fold target encoding of exact values
                                                            and digit keys (fitted on training rows only)
                                                            │
                                    ┌───────────────────────┼───────────────────────┐
                                    ▼                       ▼                       ▼
                             GLM (logistic)        4-leaf LightGBM         GLM boost (trees that
                                    │                       │              start from the GLM logit)
                                    └───────────── out-of-fold + test predictions ──┘
                                                            │
                                                            ▼
                                              stack scored by nested CV ──► submission
```

1. **One validation scheme for everything.** Five stratified folds over the training rows, fixed seed.
   Every model writes out-of-fold (OOF) predictions on these folds, so models can be stacked without
   leakage, and every decision is made on CV, not on the public leaderboard.
2. **Encode what the generator left behind.** The data is synthetic. Its generator reuses the original
   dataset's values, and the exact value of a column, its cents, its integer part and its leading digits
   all carry fraud signal that a tree cannot express with range splits. Each becomes a category,
   target-encoded inside the fold.
3. **Simple models on rich encodings.** Very shallow trees (4 leaves) and a logistic regression do as
   well as or better than deep trees on these features; trees that start from the logistic model's
   prediction combine both.
4. **Stack, honestly.** A logistic stacker over OOF predictions, its own score measured by nested CV.

## Results

5-fold CV ROC-AUC on the training rows. Full table, including every run that did not help, in
[docs/DESIGN.md](docs/DESIGN.md#results-at-a-glance).

| Model | Features | CV ROC-AUC |
| :--- | :--- | :--- |
| LightGBM, 31 leaves | raw columns | 0.94690 |
| Logistic stack of 8 round-one models | | 0.94728 |
| LightGBM, 31 leaves | + decimal digits | 0.94712 |
| LightGBM, 31 leaves | + in-fold target encoding | 0.94854 |
| LightGBM, 4 leaves | same features + pairs | 0.94923 |
| XGBoost, depth 2 | same features + pairs | 0.94926 |
| Logistic stack of 7 round-one and round-two models | | 0.94952 |
| GLM (L2 logistic) | m-estimate encoding + full digit keys | 0.94987 |
| LightGBM, 4 leaves | m-estimate encoding + full digit keys | 0.95023 |
| GLM boost: 4-leaf LightGBM from the GLM's logit | m-estimate encoding + full digit keys | 0.95056 |
| Logistic stack of round-three and round-two models | | 0.95063 |
| GLM + cubic splines on the numeric columns | m-estimate encoding + full digit keys | 0.95015 |
| GLM boost, 3 seeds averaged | m-estimate encoding + full digit keys | 0.95062 |
| Logistic stack: GLM boost (3 seeds), 4-leaf LightGBM, spline GLM, raw 4-leaf LightGBM, depth-2 XGBoost | | 0.95066 |
| XGBoost depth 2 / CatBoost depth 4 started from the GLM's logit | m-estimate encoding + full digit keys | 0.95049 / 0.95057 |
| The same stack plus XGBoost and CatBoost GLM boost | | **0.95071** |
| MLP (64-32) on the spline GLM design | logit encodings + splines | 0.94715 |
| Logistic stack of 9: the above plus MLP and spline-GLM boost | | **0.95079** |
| Round six: CatBoost GLM boost seeds and depth, slower MLP | same | running |

Leaderboard submissions:

| Date | Submission | Validation | Public LB |
| :--- | :--- | :--- | :--- |
| 2026-11-03 | LightGBM, default settings | 0.9427 (20% holdout) | 0.94578 |
| 2026-11-03 | FinTFM zero-shot, 2,000-row context | 0.8759 (20% holdout) | 0.88790 |
| 2026-11-16 | Logistic stack of 7 round-one and round-two models | 0.94952 (5-fold CV) | 0.94982 |
| 2026-11-16 | Logistic stack of 9 (GLM boost ×3 families, 4-leaf LightGBM, spline GLM, MLP) | 0.95079 (5-fold CV) | 0.95113 |

## What worked and what did not

| Idea | Effect on CV | Verdict |
| :--- | :--- | :--- |
| In-fold target encoding of exact values, rounded keys and cents | +0.0016, 5 of 5 folds | adopted |
| Very shallow trees (4 leaves or depth 2) instead of 31 leaves | +0.0006 to +0.0008, 5 of 5 folds | adopted |
| GLM boost with m-estimate encoding and full digit keys, against the best round-two tree | +0.0013, 5 of 5 folds | adopted |
| Stacker choice (rank average, logistic at C 0.01-1, hill climbing) | within 0.00002 of each other | members matter, not the stacker |
| Stacking round-three models together | +0.00007 over GLM boost alone | members too alike; need diversity |
| GLM: cubic splines (8 knots) on quantile-transformed numeric columns | +0.00028, 5 of 5 folds | adopted |
| GLM boost: 3 seeds averaged | +0.00006, 4 of 5 folds | adopted |
| GLM: C 0.003 or 0.03 instead of 0.01 | -0.00012 to 0.00000 | dropped |
| GLM: 10-way instead of 5-way inner cross-fit of the encoder | -0.00002 | dropped |
| GLM boost: learning rate 0.015 with feature fraction 0.3 | -0.00001 | dropped |
| GLM boost: 8 leaves instead of 4 | -0.00006 | dropped |
| XGBoost and CatBoost GLM boost as extra stack members | stack +0.00005 | adopted for diversity |
| GLM boost started from the spline GLM | -0.00002 alone; stack +0.00006 | stack member |
| 4-leaf LightGBM, 3 seeds | +0.0001 | adopted |
| GLM splines with 16 knots instead of 8 | -0.00009 | dropped |
| MLP on the spline GLM design | 0.94715 alone; stack +0.00003 | stack member |
| First and second decimal digits as plain features | +0.0002 | small |
| 45 pair encodings among the ten strongest columns | -0.0001 | dropped |
| Original dataset appended to training folds | -0.0001 | dropped |
| Fraud rates per value from the original dataset | -0.0001 to -0.0005 | dropped |
| A model trained on the original only, as a feature | -0.0012 | dropped |
| FinTFM (balanced context) blended with LightGBM | -0.0005 to +0.0001 | not used in the stack |

## FinTFM

[FinTFM](https://github.com/kabartay/fintfm) is a small tabular foundation model by the same author: it
predicts a row from labelled example rows in one forward pass, with no training on the competition data.
Here it was probed as a stack member. On one fold its best context strategy (`balanced`) scores 0.90679,
against 0.93666 for LightGBM on the same rows, and it does not improve the trees in a blend
([docs/DESIGN.md](docs/DESIGN.md#d8-fintfm-probe-balanced-context-is-best-but-it-does-not-lift-the-trees)).
The notebook's first full run on a 20,000-row holdout: best setting (balanced 2,000-row context, 4
contexts ensembled) 0.9089 against 0.9356 for default LightGBM; measured table in
[notebooks/fintfm-starter/README.md](notebooks/fintfm-starter/README.md).

[`notebooks/fintfm-starter/`](notebooks/fintfm-starter/) holds an educational Kaggle notebook built
around it: how in-context learning works on a table, how context size and the choice of context rows
change the result, partitioned contexts, ensembling, and a measured comparison with LightGBM.

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and a configured
[Kaggle API](https://github.com/Kaggle/kaggle-api) token.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# download the data as described in docs/DATA.md, then:
cd src
../.venv/bin/python train_gbdt.py --model glm --te single --keys full --te-smooth 20 --orig 0 --name glm
../.venv/bin/python stack.py glm lgb_digits_te1 --out ../submissions/stack.csv
```

## Commands

| Command | What it does |
| :--- | :--- |
| `src/train_gbdt.py --model {lgb,xgb,cat,glm} --name N` | 5-fold CV; writes `runs/N/{oof,test}.npy` and `meta.json` |
| `  --fe {raw,digits,ogte,combo,count,ogmodel}` | feature set (`src/features.py`) |
| `  --te {none,single,pairs} --keys {basic,full} --te-smooth {auto,20}` | in-fold target encoding: which keys, which smoothing |
| `  --offset glm` | GLM boost: the trees (LightGBM, XGBoost or CatBoost) start from the in-fold GLM's logit |
| `  --seeds K --te-cv 10` | average K seeds per fold; inner cross-fitting folds of the encoder |
| `  --rows 20000 --folds 1` | smoke test on a stratified subsample; saves nothing |
| `  --orig W` | weight of the original rows in each training fold (0 drops them) |
| `  --params '{"num_leaves": 4}' --folds 1` | model overrides; a one-fold check that saves nothing |
| `src/stack.py RUN [RUN ...] [--method {logistic,hill}] [--out FILE]` | rank average, logistic stacks and hill climbing, each scored by nested CV; optional submission |
| `src/train_fintfm.py --strategy balanced --folds 1 --rows 20000 --name N` | FinTFM zero-shot on the shared folds |
| `scripts/queue_kernel.sh K [--internet]` | run `jobs/K.txt` in a private Kaggle CPU kernel when a slot frees, then fetch `runs/` |
| `scripts/kaggle_kernel.py {push,status,fetch} --kernel K` | the same steps one at a time |

Environment variables: `S6E11_THREADS` (default 4), `S6E11_DATA`, `S6E11_RUNS`.

## Validation

- Five-fold `StratifiedKFold(shuffle=True, random_state=42)` over the competition rows only.
- Anything fitted on labels (target encoders, the GLM behind GLM boost) is fitted on the fold's training
  rows only; encodings of those rows are themselves cross-fitted.
- A change is adopted when OOF ROC-AUC improves and at least four of five folds improve.
- The public leaderboard holds roughly a thousand fraud rows, so public differences of a few thousandths
  are within noise; it is a sanity check, not a selector.

## Repository layout

```
src/
  common.py           data loading, the fixed folds, run saving
  features.py         feature sets and the target-encoding keys (exact values, digit family, pairs)
  train_gbdt.py       5-fold CV for LightGBM / XGBoost / CatBoost / GLM, with in-fold encoding
  train_fintfm.py     FinTFM zero-shot predictions on the same folds
  stack.py            rank average and nested-CV logistic stack; writes a submission
scripts/
  kaggle_kernel.py    bundles src/ and a job list into a private Kaggle kernel; push, status, fetch
  queue_kernel.sh     push when a CPU slot frees, then watch and fetch
  watch_kernels.sh    fetch each kernel's runs when it completes
jobs/                 job lists, one per kernel (each line is one train_*.py call)
notebooks/
  fintfm-starter/     educational FinTFM notebook (generated by build_notebook.py)
docs/                 competition, data and design notes
runs/                 per-run OOF and test predictions (not committed)
data/                 competition and original data (not committed)
```

## Documentation

| File | Contents |
| :--- | :--- |
| [docs/DESIGN.md](docs/DESIGN.md) | results table, every decision and finding, what the public notebooks measured |
| [docs/DATA.md](docs/DATA.md) | files, columns, measured facts, download commands |
| [docs/COMPETITION.md](docs/COMPETITION.md) | metric, deadline, how to read the leaderboard, our submissions |

## Licences and data

- Code: [MIT Licence](LICENSE).
- Competition data: not included; subject to the
  [competition rules](https://www.kaggle.com/competitions/playground-series-s6e11/rules).
- Original dataset: not included;
  [CC0-1.0](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026).

## Acknowledgements

Thanks to Kaggle for the Playground Series and to the author of the original Credit Card Fraud Detection
2026 dataset. The round-three recipe follows measurements published in public notebooks by
[Georgy Mamarin](https://www.kaggle.com/code/georgymamarin/s6e11-fraud-glm-offset-trees-review-budgets),
[Alex Chilton](https://www.kaggle.com/code/alexchilton/credit-card-fraud-lb-0-9542-shallow-tree-stack) and
[evgendvorkin](https://www.kaggle.com/code/evgendvorkin/s6e11-deep-eda-artifacts-xgb-oof-0-950285); the
code here is written independently. The FinTFM notebook takes its shape from
[Psi's TabPFN-3.5 starter](https://www.kaggle.com/code/philippsinger/tabpfn-3-5-starter-playground-series-s6e11).
