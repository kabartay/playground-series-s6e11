# Design log

Decisions, results and failures, newest last. Every CV figure is 5-fold stratified ROC-AUC on the
699,960 competition train rows (seed 42, `src/common.py:folds`), unless stated otherwise. Original rows
are only ever added to the training side of a fold.

## D1. Validation (2026-11-16)

- 5-fold `StratifiedKFold(shuffle=True, random_state=42)` over competition rows only. All models share
  these folds so out-of-fold (OOF) predictions can be stacked without leakage.
- A change is adopted when OOF AUC improves and at least four of five folds improve. The public
  leaderboard is a sanity check, not a selector (see `COMPETITION.md`).

## D2. The original dataset (2026-11-16)

`uditjain13/credit-card-fraud-detection-2026` (20,000 rows, CC0) has the same 25 columns and a 1.69%
fraud rate against 1.67% in train. Two uses are tested:

1. append it to every training fold (`--orig <weight>`);
2. derive per-value fraud rates and counts from it as features (`--fe ogte`). These never read the
   competition target, so they are fold-safe.

## D3. Models (2026-11-16)

GBDTs first (LightGBM, XGBoost, CatBoost through `src/train_gbdt.py`), then a stack over their OOF
predictions. FinTFM, the owner's tabular foundation model, is tested as an extra stack member: its
zero-shot submission on 2026-11-03 scored 0.88790 public against 0.94578 for a default LightGBM, so it
is expected to help only as a diverse input, not alone.

## D4. Compute (2026-11-16)

The Mac is shared and was at load average 280-410 on 2026-11-16, so a local 5-fold LightGBM run got
about a third of a core. CV runs go to private Kaggle CPU kernels (`scripts/kaggle_kernel.py`, 4 threads),
which share the account's five batch CPU slots with fintfm kernels; pushes queue until a slot frees.

## D5. FinTFM cost and probe (2026-11-16)

The 2026-11-03 FinTFM kernel took 6,944 s for 140k holdout rows and 15,106 s for 300k test rows on Kaggle
CPU, about 20 rows/s, with uniform 2,000-row contexts that hold roughly 33 fraud rows. Full OOF plus test
would cost about 14 kernel-hours. So the first step is a probe (`jobs/tfm1.txt`): one fold, 20,000
stratified validation rows, four context strategies (uniform, balanced, retrieval, hybrid). A full run
follows only if the best strategy lifts the GBDT stack on those same rows.

## Results

| Run | Model | Features | Original | CV AUC | Fold SD | Notes |
| --- | --- | --- | --- | --- | --- | --- |
