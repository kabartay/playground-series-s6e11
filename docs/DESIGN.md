# Design log

Decisions, results and failures for this entry. Every CV figure is 5-fold stratified ROC-AUC on the
699,960 competition training rows (seed 42, `src/common.py:folds`) unless stated otherwise. Negative
results stay in, so the same idea is not tried twice.

## Results at a glance

All 5-fold runs, best first. "TE" is in-fold target encoding; "orig" is the weight of the 20,000 original
rows added to each training fold (0 = not used).

| Run | Round | Model | Features | Orig | CV AUC | Fold SD |
| --- | --- | --- | --- | --- | --- | --- |
| stack of round five | 5 | logistic (C=0.1) on 9 OOF | | | **0.95079** | |
| stack of round four + r4c | 4 | logistic (C=0.1) on 7 OOF | | | 0.95071 | |
| r5_gb_spl8_s3 | 5 | GLM boost from the spline GLM, 3 seeds | as r3_glmboost | 0 | 0.95060 | 0.0033 |
| r5_lgb_l4_s3 | 5 | LightGBM, 4 leaves, 3 seeds | as r3_lgb_l4 | 0 | 0.95033 | 0.0034 |
| r5_glm_spl16 | 5 | GLM + splines (16 knots) | logit TE + one-hot | 0 | 0.95006 | 0.0033 |
| r5_mlp | 5 | MLP 64-32 on the spline GLM design, 3 seeds | logit TE + one-hot + splines | 0 | 0.94715 | 0.0035 |
| r5_mlp_a1e2 | 5 | MLP 32, alpha 0.01, 3 seeds | as r5_mlp | 0 | 0.94578 | 0.0037 |
| stack of round four | 4 | logistic (C=0.01) on 5 OOF | | | 0.95066 | |
| r4_gb_s3 | 4 | GLM boost, 3 seeds averaged | as r3_glmboost | 0 | 0.95062 | 0.0034 |
| stack of round three | 3 | logistic (C=0.01) on 6 OOF | | | 0.95063 | |
| r3_glmboost | 3 | 4-leaf LightGBM started from the in-fold GLM's logit | raw + TE (m-estimate 20, full digit keys) | 0 | 0.95056 | 0.0034 |
| r4_gbc_d4 | 4 | CatBoost depth 4 started from the GLM's logit | as r3_glmboost | 0 | 0.95057 | 0.0034 |
| r4_gbx_d2 | 4 | XGBoost depth 2 started from the GLM's logit | as r3_glmboost | 0 | 0.95049 | 0.0034 |
| r4_gb_lr015 | 4 | GLM boost, lr 0.015, feature fraction 0.3 | as r3_glmboost | 0 | 0.95055 | 0.0034 |
| r4_gb_l8 | 4 | GLM boost, 8 leaves | as r3_glmboost | 0 | 0.95050 | 0.0034 |
| r3_lgb_l4 | 3 | LightGBM, 4 leaves | raw + TE (m-estimate 20, full digit keys) | 0 | 0.95023 | 0.0035 |
| r4_glm_spl8 | 4 | GLM + cubic splines (8 knots) on numeric columns | logit TE + one-hot | 0 | 0.95015 | 0.0033 |
| r3_glm | 3 | L2 logistic, C=0.01 | logit TE + one-hot | 0 | 0.94987 | 0.0034 |
| r4_glm_c03 | 4 | GLM, C=0.03 | logit TE + one-hot | 0 | 0.94987 | 0.0034 |
| r4_glm_cv10 | 4 | GLM, 10-way inner cross-fit | logit TE + one-hot | 0 | 0.94985 | 0.0034 |
| r4_glm_c003 | 4 | GLM, C=0.003 | logit TE + one-hot | 0 | 0.94975 | 0.0034 |
| stack of round two | 2 | logistic on 7 OOF | | | 0.94952 | |
| xgb_digits_te2_d2 | 2 | XGBoost, depth 2 | raw + digits + TE + pairs | 0 | 0.94926 | 0.0034 |
| lgb_digits_te2_l4 | 2 | LightGBM, 4 leaves | raw + digits + TE + pairs | 0 | 0.94923 | 0.0033 |
| lgb_digits_te1 | 2 | LightGBM, 31 leaves | raw + digits + TE (exact values, rounded keys, cents) | 0 | 0.94854 | 0.0035 |
| lgb_digits_te2 | 2 | LightGBM, 31 leaves | as above + 45 top-10 pairs | 0 | 0.94845 | 0.0034 |
| stack of round one | 1 | logistic on 8 OOF | | | 0.94728 | |
| lgb_raw_l4 | 2 | LightGBM, 4 leaves | raw | 0 | 0.94752 | 0.0030 |
| lgb_digits | 2 | LightGBM, 31 leaves | raw + first two decimal digits | 0 | 0.94712 | 0.0032 |
| lgb_raw_o0 | 1 | LightGBM, 31 leaves | raw | 0 | 0.94690 | 0.0031 |
| xgb_raw_o1 | 1 | XGBoost, depth 6 | raw | 1 | 0.94681 | 0.0033 |
| lgb_raw_o1 | 1 | LightGBM, 31 leaves | raw | 1 | 0.94680 | 0.0032 |
| xgb_ogte_o1 | 1 | XGBoost, depth 6 | raw + original per-value rates | 1 | 0.94676 | 0.0036 |
| cat_ogte_o1 | 1 | CatBoost, depth 6 | raw + original per-value rates | 1 | 0.94654 | 0.0035 |
| cat_raw_o1 | 1 | CatBoost, depth 6 | raw | 1 | 0.94610 | 0.0031 |
| lgb_combo_o1 | 1 | LightGBM, 31 leaves | + ratios, flag count | 1 | 0.94659 | 0.0034 |
| lgb_count_o1 | 1 | LightGBM, 31 leaves | + value frequencies | 1 | 0.94647 | 0.0036 |
| lgb_ogte_o1 | 1 | LightGBM, 31 leaves | raw + original per-value rates | 1 | 0.94633 | 0.0034 |
| lgb_ogmodel_o1 | 1 | LightGBM, 31 leaves | + original-only model prediction | 1 | 0.94564 | 0.0037 |

Round four (D11) is running.

## D1. Validation

- 5-fold `StratifiedKFold(shuffle=True, random_state=42)` over competition rows only. All models share
  these folds, so out-of-fold (OOF) predictions can be stacked without leakage.
- A change is adopted when OOF AUC improves and at least four of five folds improve.
- The public leaderboard scores about 20% of the test set (roughly a thousand fraud rows); differences of
  a few thousandths there are noise. It is a sanity check, not a selector.

## D2. The original dataset

[Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026)
(20,000 rows, CC0) has the same columns and a 1.69% fraud rate against 1.67% in train.

How the generator used it (measured): no train or test row equals an original row on all 24 columns, yet
85-100% of train values per continuous column occur in the original (amount 97.4%, balance 84.8%,
merchant risk and velocity 100%), and only 0.02% of train rows share (amount, balance) with an original
row. The generator reuses the original's values but recombines them into new rows.

Three uses were tested in round one: appending the rows to each training fold (`--orig`), per-value fraud
rates and counts from the original as features (`--fe ogte`), and the prediction of a LightGBM trained on
the original alone (`--fe ogmodel`). None helped (D6).

## D3. Models

Gradient-boosted trees first (LightGBM, XGBoost, CatBoost through `src/train_gbdt.py`), then a stack over
their OOF predictions (`src/stack.py`). Round three adds a GLM (L2 logistic regression) and GLM boost.
FinTFM, a tabular foundation model by the same author, is tested as a candidate member (D8).

## D4. Compute

The development Mac is shared with other long-running jobs (load average 280-410 on 2026-11-16), so CV
runs go to private Kaggle CPU kernels: `scripts/kaggle_kernel.py` embeds `src/` into one script with a
job list, `scripts/queue_kernel.sh` pushes it when one of the account's five CPU slots frees and fetches
`runs/` when it finishes.

## D5. FinTFM cost

A FinTFM kernel on 2026-11-03 (fintfm 4ad16a8) took 6,944 s for 140k holdout rows and 15,106 s for 300k
test rows on Kaggle CPU, about 20 rows/s, with uniform 2,000-row contexts holding roughly 33 fraud rows.
Full OOF plus test would have cost about 14 kernel-hours, so a one-fold probe came first (D8).

## D6. Round one: the original data does not help

All eight round-one runs land at CV 0.9456-0.9469. The original rows change LightGBM by -0.0001 (0.94690
without, 0.94680 with); original-derived features cost 0.0002-0.0013. A logistic stack of all eight adds
+0.0004 (0.94728; rank average 0.94738). The public leaderboard cluster at 0.9542-0.9544 implies a CV of
about 0.951, so the gap was about 0.004 and had to come from somewhere else.

## D7. What the public notebooks measured

Seven public notebooks were read as text (none was run). Numbers are as printed in them, 5-fold OOF
ROC-AUC unless stated; "public" means public leaderboard.

- **Most of the 0.954 cluster is blending of public files.**
  [Steer, Then Mix](https://www.kaggle.com/code/areebbb/s6e11-steer-then-mix-lb-0-95423),
  [GPT-2 and GLM Fusion](https://www.kaggle.com/code/malyshevdanil/fraud-signal-lab-gpt-2-and-glm-fusion) and
  [LM Ladder](https://www.kaggle.com/code/goodpjw2008/s6e11-lm-ladder-ladder-meets-stack-lb-0-95427) train
  nothing: they mix other people's submission files in logit-rank space and steer by the public board.
  The ladder notebook's own CV-backed file has OOF 0.95266 (public 0.95379); its public-tuned file is
  about 0.0028 worse in OOF. "GLM" and "LM" mean linear models, not language models.
- **The honest single-model level is about 0.9503-0.9513**, from
  [GLM-Offset Trees](https://www.kaggle.com/code/georgymamarin/s6e11-fraud-glm-offset-trees-review-budgets)
  (Georgy Mamarin) and the
  [Shallow-Tree Stack](https://www.kaggle.com/code/alexchilton/credit-card-fraud-lb-0-9542-shallow-tree-stack)
  (Alex Chilton):
  - In-fold target encoding of every column's exact value plus digit keys of the float columns (cents,
    integer part, tens, first decimal, leading digits, printed length), m-estimate smoothing 20, 5-way
    inner cross-fit: raw 0.9463874 to 0.9492956. Per group: values +0.00089, integer keys +0.00088,
    cents +0.00052, each up on 5 of 5 folds.
  - 4-leaf LightGBM (learning rate 0.03, colsample 0.5, min_child_samples 100, reg_lambda 1, about
    1,600 trees): 0.9502590; 0.95036-0.95052 in the second notebook. 2 or 3 leaves are no better.
  - Logistic regression (L2, C=0.01) on one-hot plus logit encodings: 0.9499797; an 80/20 blend with
    the 4-leaf trees: 0.9504790.
  - GLM boost (4-leaf trees with `init_score` set to the GLM's logit): 0.9506947-0.9513073.
- **The generator's artefacts** are described in
  [Deep EDA & Artifacts](https://www.kaggle.com/code/evgendvorkin/s6e11-deep-eda-artifacts-xgb-oof-0-950285)
  (single XGBoost OOF 0.950285): signal in decimal endings, flag effects stronger in the synthetic data
  than in the original, and a truncated distance tail.
- **Measured as not helping:** 45 pair encodings (0.94936 without, 0.94951 with); the original dataset as
  a teacher probability (-0.0036715); frequency encoding (+0.0001); XGBoost in the blend (-0.00001); a
  GPT-2-small member (-0.0029 on the blend); RealMLP and CatBoost got blend weight 0. Removing
  `is_ai_generated_scam_attempt` costs 0.0013.
- [EDA | blend](https://www.kaggle.com/code/saamhm/eda-blend-lb-0-95239-credit-card-fraud) trains its own
  trees (best XGBoost depth 3, 0.94751) and takes 80% of its final file from public submissions.

## D8. FinTFM probe: balanced context is best, but it does not lift the trees

`jobs/tfm1.txt`, fintfm 8a25b4c, published checkpoint, 2,000-row context drawn from fold 0's training
rows plus the original rows, scored on 20,000 stratified rows of fold 0 (about 334 fraud), Kaggle CPU:

| Context strategy | AUC | Seconds |
| --- | --- | --- |
| uniform | 0.89958 | 295 |
| balanced | 0.90679 | 285 |
| hybrid | 0.89648 | 281 |
| retrieval | 0.78609 | 673 |

Throughput was about 70 rows/s, against about 20 rows/s for the 2026-11-03 kernel.

On the same rows its rank correlation with `lgb_raw_o0` is 0.864. A blend weight picked on one random half
and scored on the other changes AUC by -0.00043 and +0.00014 (LightGBM), -0.00047 and 0.00000 (XGBoost).
No gain, so FinTFM is not run in full for the stack. It is the subject of the starter notebook in
`notebooks/fintfm-starter/`.

## D9. Round two, and the round-three set-up

Round two (`jobs/lgb2a.txt`), 31-leaf LightGBM, no original rows, sklearn `TargetEncoder(smooth="auto")`:

| Run | Features | CV AUC | Folds |
| --- | --- | --- | --- |
| lgb_digits | raw + first and second decimal digits | 0.94712 | 0.94561 0.94934 0.95168 0.94725 0.94198 |
| lgb_digits_te1 | + TE of exact values, rounded keys, cents | 0.94854 | 0.94747 0.94991 0.95370 0.94910 0.94282 |
| lgb_digits_te2 | + 45 top-10 pairs | 0.94845 | 0.94754 0.95026 0.95338 0.94882 0.94290 |

Target encoding adds +0.0016 over raw (0.94690) and lifts all five folds; pairs add nothing, as D7 found.

Round three (`jobs/r3a.txt`, `jobs/r3b.txt`) switches to m-estimate smoothing 20, adds the digit-key
family (`--keys full`: integer part, tens, 0.1 resolution, first one and two significant digits, printed
length), and tests a GLM (L2 logistic, C=0.01, on logit encodings plus one-hot) and GLM boost (4-leaf
LightGBM with `init_score` from the in-fold GLM). One-fold check on fold 0: GLM 0.94887; GLM boost with
only 30 trees 0.94901, against 0.94747 for lgb_digits_te1 on the same fold.

## D10. Shallow trees (lgb2b) and the round-two stack

Same in-fold encoding as `lgb_digits_te2` (D9), trees made very shallow (learning rate 0.05,
`min_child_samples` 200 for LightGBM):

| Run | CV AUC | Folds | Best iterations |
| --- | --- | --- | --- |
| lgb_raw_l4 (4 leaves, raw columns) | 0.94752 | 0.94602 0.94941 0.95188 0.94780 0.94275 | 1,695-2,296 |
| lgb_digits_te2_l4 (4 leaves) | 0.94923 | 0.94809 0.95061 0.95416 0.94962 0.94393 | 1,178-1,734 |
| xgb_digits_te2_d2 (depth 2) | 0.94926 | 0.94792 0.95038 0.95432 0.95005 0.94397 | 1,231-1,674 |

Four leaves beat 31 by +0.0006 on raw columns and +0.0008 on the encoded set, on all five folds.

A logistic stack of seven round-one and round-two runs (`stack.py lgb_digits_te2_l4 xgb_digits_te2_d2
lgb_digits_te1 lgb_digits_te2 lgb_raw_l4 lgb_raw_o0 xgb_raw_o1`) scores 0.94952 by nested CV (folds 0.94836
0.95099 0.95440 0.95006 0.94438), +0.0003 over the best single run; rank average 0.94935. The two shallow
encoded models carry most of the weight (0.32 and 0.38).

## D11. GLM boost, stacker choice, and round four

`r3_glmboost` (`jobs/r3b.txt`): m-estimate (20) encoding of 76 keys (exact values plus the full digit
family), an in-fold GLM (C=0.01), then 4-leaf LightGBM from the GLM's logit. CV **0.95056**, folds 0.94996
0.95109 0.95548 0.95161 0.94490, best iterations 558-1,139, 550 s on a Kaggle CPU kernel. That is +0.0013
over the best round-two model and higher on all five folds; it sits inside the 0.9507-0.9513 range the
public notebooks report for the same recipe (D7).

The stacker is not the bottleneck. On the seven round-two members, every stacker scored by nested CV
lands within 0.00002:

| Stacker | Nested CV |
| --- | --- |
| rank average | 0.94935 |
| logistic C=0.01 | 0.94954 |
| logistic C=0.1 | 0.94953 |
| logistic C=1 | 0.94952 |
| greedy hill climbing on standardised logits | 0.94952 |

Gains therefore have to come from the members.

Round three's other runs (`jobs/r3a.txt`) reproduce the published levels: GLM 0.94987 (folds 0.94887
0.95045 0.95511 0.95059 0.94456, 25 iterations, 148 s) and 4-leaf LightGBM 0.95023 (folds 0.94981 0.95106
0.95515 0.95139 0.94426, 1,365-2,505 trees, 828 s).

A bug found on the way: GLM and GLM-boost runs save logits, and `stack.py` clipped every input to
[1e-6, 1-1e-6] before taking the logit, which flattened those runs to near-constants. It now detects
logit-scale runs, and hill climbing is recalibrated in each fold (it scored 0.94688 pooled while every fold
was near 0.951). Round-two results are unaffected (all their members are probabilities).

Stack of `r3_glmboost r3_lgb_l4 r3_glm xgb_digits_te2_d2 lgb_digits_te2_l4 lgb_raw_l4`, nested CV:

| Stacker | Nested CV | Folds |
| --- | --- | --- |
| rank average | 0.95049 | |
| logistic C=0.01 | **0.95063** | 0.94998 0.95149 0.95554 0.95160 0.94512 |
| logistic C=1 | 0.95061 | |
| hill climbing | 0.95057 | |

Only +0.00007 over `r3_glmboost` alone (weights 0.68 GLM boost, 0.32 4-leaf trees, -0.13 GLM): members built
on the same encodings agree too closely. The next gains need members that differ, not a better stacker.

Round one's CatBoost runs (`jobs/cat1.txt`, depth 6, about 75 min each): `cat_raw_o1` 0.94610 and
`cat_ogte_o1` 0.94654. Adding `cat_ogte_o1` to the round-three stack gives 0.95060 (weight 0.03), no gain. Round four changes one thing per run against round three:

- `jobs/r4a.txt`, GLM: C 0.003 and 0.03; cubic splines (8 knots) on quantile-transformed numeric
  columns; 10-way instead of 5-way inner cross-fitting of the encoder.
- `jobs/r4b.txt`, LightGBM GLM boost: 3 seeds averaged; learning rate 0.015 with feature fraction 0.3;
  8 leaves.
- `jobs/r4c.txt`, GLM boost with other tree families for diversity: XGBoost depth 2 (`base_margin`) and
  CatBoost depth 4 (`baseline`).

## D12. Round four results

One change per run against round three (D11):

| Run | Change | CV AUC | Folds up | Verdict |
| --- | --- | --- | --- | --- |
| r4_glm_spl8 | GLM + cubic splines (8 knots) on quantile-transformed numeric columns | 0.95015 (+0.00028) | 5 of 5 | adopted |
| r4_glm_c03 | C 0.03 | 0.94987 (0.00000) | | no change |
| r4_glm_cv10 | 10-way inner cross-fit of the encoder | 0.94985 (-0.00002) | | dropped |
| r4_glm_c003 | C 0.003 | 0.94975 (-0.00012) | | dropped |
| r4_gb_s3 | GLM boost, 3 seeds averaged | 0.95062 (+0.00006) | 4 of 5 | adopted |
| r4_gb_lr015 | learning rate 0.015, feature fraction 0.3 | 0.95055 (-0.00001) | | dropped |
| r4_gb_l8 | 8 leaves, min_child_samples 200 | 0.95050 (-0.00006) | | dropped |

The splines give the GLM a smooth shape on each numeric column that the exact-value encodings miss.
Stack `r4_gb_s3 r3_lgb_l4 r4_glm_spl8 lgb_raw_l4 xgb_digits_te2_d2`, logistic C=0.01: nested CV
**0.95066** (folds 0.94997 0.95149 0.95566 0.95157 0.94521); rank average 0.95057.

GLM boost with other tree families (`jobs/r4c.txt`): XGBoost depth 2 0.95049 (folds 0.94976 0.95105
0.95551 0.95150 0.94489), CatBoost depth 4 0.95057 (folds 0.94972 0.95107 0.95562 0.95142 0.94520). Neither
beats LightGBM alone, but they rank rows differently: added to the stack above, nested CV rises to
**0.95071** (logistic C=0.1; rank average 0.95072; hill climbing 0.95065), and CatBoost takes the largest
weight (0.26, against 0.21 for LightGBM GLM boost and 0.16 for XGBoost GLM boost).

Round five: `jobs/r5a.txt` puts both winners together (GLM boost from the spline GLM, 3 seeds);
`jobs/r5b.txt` tries GLM splines with 16 knots, 4-leaf trees with 3 seeds, and a small MLP on the spline
GLM design as a different kind of member.

## D13. Round five

| Run | Change | CV AUC | Against | Verdict |
| --- | --- | --- | --- | --- |
| r5_gb_spl8_s3 | GLM boost from the spline GLM, 3 seeds | 0.95060 | r4_gb_s3 0.95062, 3 of 5 folds up | no single gain; kept as a stack member |
| r5_lgb_l4_s3 | 4-leaf LightGBM, 3 seeds | 0.95033 | r3_lgb_l4 0.95023, 3 up, 1 tie, 1 down | replaces r3_lgb_l4 |
| r5_glm_spl16 | 16 spline knots | 0.95006 | r4_glm_spl8 0.95015 | dropped |
| r5_mlp | MLP 64-32 on the spline GLM design | 0.94715 | | weak alone; stack member |
| r5_mlp_a1e2 | MLP 32, alpha 0.01 | 0.94578 | | dropped |

A spline GLM is a better model than the plain GLM (D12) but not a better starting point for the trees:
the trees already recover the numeric shapes. The MLP stops after 8-9 epochs and is weak alone, yet it
gets stack weight 0.08 because it errs differently.

Stacks, logistic C=0.1, nested CV (base: `r4_gb_s3 r5_lgb_l4_s3 r4_glm_spl8 lgb_raw_l4 xgb_digits_te2_d2
r4_gbx_d2 r4_gbc_d4`):

| Members | Nested CV | Rank average |
| --- | --- | --- |
| base (7) | 0.95070 | 0.95072 |
| + r5_mlp | 0.95073 | 0.95076 |
| + r5_mlp, r5_gb_spl8_s3 | **0.95079** | 0.95078 |
| + both MLPs, r5_gb_spl8_s3 | 0.95078 | 0.95071 |

The 9-member stack (folds 0.95006 0.95156 0.95575 0.95178 0.94538) is up on all five folds against the
7-member round-four stack, and is written to `submissions/stack_r5.csv`. Four member sets were compared on
the same folds, so a selection effect of the order of 0.00002 is possible.

Round six: CatBoost GLM boost with 3 seeds (`jobs/r6a.txt`), depth 6 and a slower, wider MLP
(`jobs/r6b.txt`).
