# Predicting Credit Card Fraud: Kaggle Playground S6E11

[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![LightGBM 4.7](https://img.shields.io/badge/LightGBM-4.7-blue.svg)
![XGBoost 3.4](https://img.shields.io/badge/XGBoost-3.4-blue.svg)
![CatBoost 1.2](https://img.shields.io/badge/CatBoost-1.2-blue.svg)

A personal entry to the Kaggle competition
[Playground Series, Season 6 Episode 11: Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11).

Given 24 attributes of a card transaction (amount, timing, velocity, distance from home, merchant risk,
authentication method, device, VPN use and other flags), predict the probability that it is fraudulent.
About 1.7% of the 699,960 training transactions are fraud. Submissions are scored by ROC-AUC on 299,984
test transactions; the competition closes on 2026-11-30.

> **Status:** in progress. Results below are updated as runs finish.

## Contents

- [Approach](#approach)
- [Status and results](#status-and-results)
- [Quick start](#quick-start)
- [Commands](#commands)
- [Validation](#validation)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Licences and data](#licences-and-data)
- [Acknowledgements](#acknowledgements)

## Approach

```
train.csv (699,960) ──┐
                      ├─► 5 fixed stratified folds ──► LightGBM / XGBoost / CatBoost
original.csv (20,000) ┘      (original rows only on      │  out-of-fold + test predictions
                              the training side)          ▼  runs/<name>/
                                                     stack over OOF ──► submission
```

1. **One validation scheme for everything.** Five stratified folds over the competition training rows,
   fixed seed. Every model writes out-of-fold (OOF) predictions on these folds, so members can be stacked
   without leakage.
2. **Use the original dataset.** The competition data is synthetic, generated from the 20,000-row
   [Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026)
   dataset (CC0). It is appended to the training side of each fold, and its per-value fraud rates and
   counts become features. It never enters validation, and the competition target is never used to build
   features.
3. **Gradient-boosted trees.** LightGBM, XGBoost and CatBoost with native categorical handling and early
   stopping on the validation fold.
4. **Stacking.** A simple model over the members' OOF predictions sets the blend.
5. **FinTFM.** A tabular foundation model by the same author, tested as an extra stack member for
   diversity rather than as a standalone model.

## Status and results

| Step | State |
| :--- | :--- |
| Data, original dataset, shared folds, CV harness | done |
| GBDT sweep: effect of original rows and original-derived features | queued on Kaggle CPU kernels |
| Stacking | next |
| FinTFM as a stack member | after stacking |

Submissions so far:

| Submission | Validation ROC-AUC | Public LB |
| :--- | :--- | :--- |
| LightGBM, default settings (2026-11-03) | 0.9427 (single holdout) | 0.94578 |
| FinTFM zero-shot, 2,000-row context (2026-11-03) | 0.8759 (single holdout) | 0.88790 |

Every run, including those that did not help, is logged in [docs/DESIGN.md](docs/DESIGN.md).

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and a configured
[Kaggle API](https://github.com/Kaggle/kaggle-api) token.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# download the data as described in docs/DATA.md, then:
cd src
../.venv/bin/python train_gbdt.py --model lgb --fe ogte --orig 1 --name lgb_ogte_o1
```

## Commands

| Command | What it does |
| :--- | :--- |
| `src/train_gbdt.py --model {lgb,xgb,cat} --fe {raw,ogte,combo,count} --orig W --name N` | 5-fold CV; original rows weighted `W` (0 drops them); writes `runs/N/` |
| `src/train_gbdt.py ... --folds 1 --params '{"n_estimators": 30}'` | quick one-fold check; nothing saved |
| `scripts/kaggle_kernel.py push --kernel K --jobs jobs/K.txt` | run a job list in a private Kaggle CPU kernel |
| `scripts/kaggle_kernel.py status --kernel K` | kernel state |
| `scripts/kaggle_kernel.py fetch --kernel K` | download the kernel's `runs/` and print CV scores |

Environment variables: `S6E11_THREADS` (default 4), `S6E11_DATA`, `S6E11_RUNS`.

## Validation

- Five-fold `StratifiedKFold(shuffle=True, random_state=42)` over competition rows only.
- A change is adopted when OOF ROC-AUC improves and at least four of five folds improve.
- The public leaderboard holds roughly a thousand fraud rows, so public differences of a few thousandths
  are within noise; it is a sanity check, not a selector.

## Repository layout

```
src/
  common.py         data loading, fixed folds, run saving
  features.py       feature sets: raw, original-data rates (ogte), ratios (combo), frequencies (count)
  train_gbdt.py     5-fold CV for LightGBM / XGBoost / CatBoost
scripts/
  kaggle_kernel.py  bundles src/ into a private Kaggle kernel; push, status, fetch
jobs/               job lists for Kaggle kernels
docs/               competition, data and design notes
runs/               per-run OOF and test predictions (not committed)
data/               competition and original data (not committed)
```

## Documentation

| File | Contents |
| :--- | :--- |
| [docs/COMPETITION.md](docs/COMPETITION.md) | metric, deadline, leaderboard notes, our submissions |
| [docs/DATA.md](docs/DATA.md) | files, columns, measured facts, download commands |
| [docs/DESIGN.md](docs/DESIGN.md) | decision log and the results table for every run |

## Licences and data

- Code: [MIT Licence](LICENSE).
- Competition data: not included; subject to the
  [competition rules](https://www.kaggle.com/competitions/playground-series-s6e11/rules).
- Original dataset: not included;
  [CC0-1.0](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026).

## Acknowledgements

Thanks to Kaggle for the Playground Series and to the author of the original Credit Card Fraud Detection
2026 dataset.
