# Competition

| | |
| --- | --- |
| Name | Kaggle Playground Series, Season 6 Episode 11: *Predicting Credit Card Fraud* |
| URL | https://www.kaggle.com/competitions/playground-series-s6e11 |
| Task | Binary classification: predict the probability that a card transaction is fraudulent (`is_fraud`) |
| Metric | ROC-AUC on the predicted probability |
| Deadline | 2026-11-30 23:59 UTC |
| Prize | Kaggle swag for the top 3 (no money) |
| Submission | `id,is_fraud`, one row per test id (299,984 rows) |

## Leaderboard notes

- The public leaderboard scores a fraction of the test set; the private leaderboard decides the final
  ranking. With about 1.7% positives, the public subset holds roughly a thousand fraud rows, so public
  differences of a few thousandths of AUC are within noise. Model choice follows cross-validation,
  not the public score.
- Snapshot 2026-11-16: first place 0.95931, a dense group of about twenty teams at 0.9542-0.9544.

## Our submissions

| Date | File | Public | CV | Notes |
| --- | --- | --- | --- | --- |
| 2026-11-03 | `submission_fintfm.csv` | 0.88790 | 0.8759 (holdout) | FinTFM zero-shot, published synthetic-only checkpoint, 2,000-row context |
| 2026-11-03 | `submission_lightgbm.csv` | 0.94578 | 0.9427 (holdout) | LightGBM 500 rounds, lr 0.05, defaults otherwise |
