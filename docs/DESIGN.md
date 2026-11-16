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

Measured 2026-11-16: no train or test row equals an original row on all 24 columns, yet 85-100% of
train values per continuous column occur in the original (amount 97.4%, balance 84.8%, risk and
velocity 100%), and only 0.02% of train rows share (amount, balance) with an original row. The
generator reuses the original's values but recombines them, so per-value statistics carry over and
row lookup does not. A third use follows: `--fe ogmodel` adds the prediction of a LightGBM fitted on the
original alone (5-fold OOF for the original's own rows).

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
| lgb_combo_o1 | lgb | combo | 1 | 0.94659 | 0.00343 | Kaggle CPU, 460 s |
| lgb_count_o1 | lgb | count | 1 | 0.94647 | 0.00356 | Kaggle CPU, 571 s |
| lgb_ogmodel_o1 | lgb | ogmodel | 1 | 0.94564 | 0.00366 | Kaggle CPU, 603 s |
| lgb_ogte_o1 | lgb | ogte | 1 | 0.94633 | 0.00342 | Kaggle CPU, 510 s |
| lgb_raw_o0 | lgb | raw | 0 | 0.94690 | 0.00310 | Kaggle CPU, 353 s |
| lgb_raw_o1 | lgb | raw | 1 | 0.94680 | 0.00316 | Kaggle CPU, 332 s |
| xgb_ogte_o1 | xgb | ogte | 1 | 0.94676 | 0.00360 | Kaggle CPU, 1074 s |
| xgb_raw_o1 | xgb | raw | 1 | 0.94681 | 0.00331 | Kaggle CPU, 355 s |
| stack of the 8 above | logistic | | | 0.94728 | | nested CV; rank average 0.94738 |

## D6. Round one: the original data does not help; the gap is elsewhere (2026-11-16)

All eight runs above land at CV 0.9456-0.9469. Adding the original rows changes LightGBM by -0.0001
(0.94690 without, 0.94680 with); original-derived features cost 0.0002-0.0013. A logistic stack adds
+0.0004. The public cluster at 0.9543 implies a CV near 0.951, so the gap is about 0.004.

Public notebooks (read as text, nothing run) point at generator artefacts: decimal endings of the
continuous columns carry signal that no range split can express; every column's exact value works as a
category when target-encoded inside the fold; pairs among the strongest columns help; trees with 4 leaves
do as well as or better than deep ones. Round two (`jobs/lgb2a.txt`, `jobs/lgb2b.txt`) tests each of
these in turn. Original rows are dropped from round two.
