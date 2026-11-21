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

Final selection (marked on Kaggle 2026-11-20): `stack_pubwz.csv` and `stack_pub5z.csv`, the two best files by CV
(0.95263 and 0.95254). Earlier selection (2026-11-19): `stack_pub5z.csv` and `stack_r12z.csv`.

| Date | File | Public | Validation | Notes |
| --- | --- | --- | --- | --- |
| 2026-11-21 | `stack_pub4wz.csv` | 0.95325 | 0.95273 (nested 5-fold CV) | fit-free mean of four nested stacks (`stack_pub5`, wide LM, DavutKursun, dhananjayk2609; logit-ranks) + zones (D31); 5 of 5 folds above `stack_pubwz`; best public score |
| 2026-11-20 | `stack_pubwz.csv` | 0.95293 | 0.95263 (nested 5-fold CV) | 0.5 `stack_pub5` + 0.5 wide LM stack (logit-ranks) + zero-fraud zones (D29); 4 of 5 folds above `stack_pub5z`; final pick |
| 2026-11-20 | honest-CV notebook, Kaggle version 1 | 0.95155 | 0.95135 (nested 5-fold CV, local check of the same six runs) | public notebook: six own models, nested stack + zones; submitted from the notebook, not a final pick |
| 2026-11-20 | FinTFM starter, Kaggle version 3 | 0.94105 | 0.9296 (single 20,000-row holdout) | fintfm 1.0.0 linear readout fitted on the whole pool; up from 0.91516 for Kaggle version 2 (in-context, partitioned + balanced) |
| 2026-11-19 | `stack_pub5z.csv` | 0.95288 | 0.95254 (nested 5-fold CV) | `stack_pub5` + zero-fraud zones (D27); best public score, final pick |
| 2026-11-19 | `stack_r12z.csv` | 0.95164 | 0.95143 (nested 5-fold CV) | `stack_r12` + zero-fraud zones (D27); best own-model public score, final pick |
| 2026-11-19 | `stack_r16.csv` | 0.95163 | 0.95136 (nested 5-fold CV) | `stack_r12` with the all-rows RealMLPs (D23); public check only, not a final pick |
| 2026-11-18 | `stack_r12.csv` | 0.95160 | 0.95138 (nested 5-fold CV) | 24 own models: honest boosters + two RealMLPs (D23); best own-model public score |
| 2026-11-17 | `stack_pub5.csv` | 0.95284 | 0.95249 (nested 5-fold CV) | our 21 models + goodpjw2008's nested wide LM stack (D21); best public score |
| 2026-11-17 | `stack_r10.csv` | 0.95134 | 0.95132 (nested 5-fold CV) | logistic stack of 22 own models, round ten |
| 2026-11-17 | `stack_pub4.csv` | 0.95219 | 0.95225 (nested 5-fold CV) | our 21 models + 152 public OOF members (D20) |
| 2026-11-17 | `stack_pub3.csv` | 0.95190 | 0.95204 (nested 5-fold CV) | our 21 models + 52 public OOF members (D19); rank 596 of 2,098 |
| 2026-11-17 | `stack_pub2.csv` | 0.95164 | 0.95183 (nested 5-fold CV) | our 21 round-eight models + 38 public OOF members (D17); rank 624 of 2,065 at the time |
| 2026-11-17 | `stack_r8.csv` | 0.95123 | 0.95114 (nested 5-fold CV) | logistic stack of 21 own models, round eight (D17) |
| 2026-11-17 | `stack_pub1.csv` | 0.95165 | 0.95175 (nested 5-fold CV) | our 9 models + 38 fold-checked public OOF members (D15); rank 606 of 2,026 |
| 2026-11-16 | `stack_r5.csv` | 0.95113 | 0.95079 (nested 5-fold CV) | logistic stack of 9: GLM boost (LightGBM, XGBoost, CatBoost), 4-leaf LightGBM, spline GLM, MLP (D13); rank 766 of 2,024 |
| 2026-11-16 | `stack_r2.csv` | 0.94982 | 0.94952 (nested 5-fold CV) | logistic stack of 7 round-one and round-two models (D10) |
| 2026-11-03 | `submission_fintfm.csv` | 0.88790 | 0.8759 (single 20% holdout) | FinTFM zero-shot, published checkpoint, 2,000-row uniform context |
| 2026-11-03 | `submission_lightgbm.csv` | 0.94578 | 0.9427 (single 20% holdout) | LightGBM 500 rounds, lr 0.05, defaults otherwise |
