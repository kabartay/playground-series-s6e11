# Competition

| | |
| --- | --- |
| Name | Kaggle Playground Series, Season 6 Episode 11: *Predicting Credit Card Fraud* |
| URL | https://www.kaggle.com/competitions/playground-series-s6e11 |
| Task | Binary classification: the probability that a card transaction is fraudulent (`is_fraud`) |
| Metric | ROC-AUC on the predicted probability (only the ranking matters) |
| Deadline | 2026-11-30 23:59 UTC |
| Prize | Kaggle swag for the top places (no money) |
| Submission | `id,is_fraud`, one row per test id (299,984 rows) |

## Reading the leaderboard

- The public leaderboard scores about 20% of the test set, roughly 60,000 rows with about a thousand
  fraud rows. Public differences of a few thousandths of AUC are within noise; the private leaderboard
  (the other 80%) decides the ranking.
- Snapshot 2026-11-16: first place 0.95931, then a dense group of about twenty teams at 0.9542-0.9544.
  Much of that group submits blends of each other's public files, tuned on the public board
  (`DESIGN.md`, D7). Notebooks that report their own cross-validation reach about 0.9503-0.9513 OOF.
- This entry chooses models by 5-fold cross-validation only.

## Our submissions

| Date | File | Public | Validation | Notes |
| --- | --- | --- | --- | --- |
| 2026-11-03 | `submission_lightgbm.csv` | 0.94578 | 0.9427 (single 20% holdout) | LightGBM 500 rounds, lr 0.05, defaults otherwise |
| 2026-11-03 | `submission_fintfm.csv` | 0.88790 | 0.8759 (single 20% holdout) | FinTFM zero-shot, published checkpoint, 2,000-row uniform context |
