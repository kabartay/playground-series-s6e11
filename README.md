# Predicting Credit Card Fraud: Kaggle Playground S6E11

Solution code for [Kaggle Playground Series, Season 6 Episode 11](https://www.kaggle.com/competitions/playground-series-s6e11):
predict the probability that a card transaction is fraudulent, scored by ROC-AUC.

> **Status:** in progress. The competition closes on 2026-11-30. Results below are updated as runs finish.

## Results

| Submission | CV ROC-AUC (5-fold) | Public LB | Private LB |
| --- | --- | --- | --- |
| LightGBM, default settings (2026-11-03) | 0.9427 (single holdout) | 0.94578 | pending |
| FinTFM zero-shot (2026-11-03) | 0.8759 (single holdout) | 0.88790 | pending |

Every run, including the ones that did not help, is logged in [docs/DESIGN.md](docs/DESIGN.md).

## Approach

1. **One validation scheme for everything.** Five stratified folds over the 699,960 training rows, fixed
   seed. All models write out-of-fold predictions on these folds so they can be stacked without leakage.
2. **Use the original dataset.** The competition data is synthetic, generated from the 20,000-row
   [Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026)
   dataset (CC0). It is added to the training side of each fold and used to build per-value fraud-rate
   features. It never enters validation.
3. **Gradient-boosted trees.** LightGBM, XGBoost and CatBoost with native categorical handling and
   early stopping on the validation fold.
4. **Stacking.** A simple model over the OOF predictions of all members picks the blend.
5. **FinTFM.** A tabular foundation model by the same author, tested as an extra stack member.

## Repository layout

```
src/
  common.py       data loading, fixed folds, run saving
  features.py     feature sets (raw, original-data target rates, ratios, frequencies)
  train_gbdt.py   5-fold CV for LightGBM / XGBoost / CatBoost
docs/
  COMPETITION.md  metric, deadline, leaderboard notes, our submissions
  DATA.md         files, columns, measured facts, how to download
  DESIGN.md       decision log and results table
runs/             per-run OOF and test predictions (not committed)
data/             competition and original data (not committed)
```

## Reproduce

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and a configured
[Kaggle API](https://github.com/Kaggle/kaggle-api) token.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# download data as described in docs/DATA.md, then:
cd src
../.venv/bin/python train_gbdt.py --model lgb --fe ogte --orig 1 --name lgb_ogte
```

Each run writes `runs/<name>/oof.npy`, `test.npy` and `meta.json` (CV score, per-fold scores, best
iterations, run time).

## Data

The data is not included. Competition files are subject to the
[competition rules](https://www.kaggle.com/competitions/playground-series-s6e11/rules); the original dataset
is CC0-1.0. See [docs/DATA.md](docs/DATA.md).

## Licence

Code is released under the [MIT Licence](LICENSE).
