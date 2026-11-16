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

## D7. What the public notebooks measured (2026-11-16)

Seven public notebooks were read as text (nothing run). Numbers below are as printed in them, 5-fold
OOF ROC-AUC unless stated; "public" is the public leaderboard.

- **Most of the 0.9542 cluster is public-file blending.** "Steer, Then Mix", "GPT-2 and GLM Fusion" and
  the "LM ladder" train nothing: they mix other people's submission CSVs in logit-rank space and steer by
  the public board. The ladder notebook's own CV-backed file is OOF 0.95266 (public 0.95379); its
  public-tuned file is about 0.0028 worse in OOF. "GLM" and "LM" are linear models, not language models.
- **Honest single-model ceiling about 0.9503-0.9513:**
  - In-fold target encoding of every column's exact value plus digit keys of the six float columns
    (cents, integer part, tens, first decimal, leading digits, digit length), m-estimate smoothing 20,
    5-way inner cross-fit. Raw 0.9463874 to 0.9492956 with all groups (Mamarin); per group +0.00089
    (values), +0.00088 (integer keys), +0.00052 (cents), each up on 5/5 folds.
  - 4-leaf LightGBM (lr 0.03, colsample 0.5, min_child_samples 100, reg_lambda 1, ~1,600 trees):
    0.9502590 (Mamarin H10); 0.95036-0.95052 on the same recipe (alexchilton); 2 or 3 leaves no better.
  - Logistic regression (L2, C=0.01) on one-hot plus logit of the encodings: 0.9499797; fixed 80/20
    with the 4-leaf trees: 0.9504790.
  - "GLM boost": 4-leaf trees with `init_score` set to the GLM's logit: 0.9506947-0.9513073
    (Mamarin, adapted), 0.95159 ten-fold (sameershaikh27, quoted).
- **Measured as not helping:** the 45 top-10 pair encodings (0.94936 without, 0.94951 with); the
  original dataset as a teacher probability (-0.0036715); frequency encoding (+0.0001); XGBoost in the
  blend (-0.00001); a GPT-2-small member (-0.0029 on the blend); RealMLP and CatBoost got blend weight 0.
  Removing `is_ai_generated_scam_attempt` costs 0.0013.

Consequence for us: round two (`lgb2a`, `lgb2b`) uses sklearn's `smooth="auto"` and only end digits,
so expect it to land short of 0.950. Round three switches to the m-estimate encoder and the full digit
key set, then adds the GLM and GLM-boost members.
