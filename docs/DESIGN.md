# Design log

Decisions, results and failures for this entry. Every CV figure is 5-fold stratified ROC-AUC on the
699,960 competition training rows (seed 42, `src/common.py:folds`) unless stated otherwise. Negative
results stay in, so the same idea is not tried twice.

## Results at a glance

All 5-fold runs, best first. "TE" is in-fold target encoding; "orig" is the weight of the 20,000 original
rows added to each training fold (0 = not used).

| Run | Round | Model | Features | Orig | CV AUC | Fold SD |
| --- | --- | --- | --- | --- | --- | --- |
| stack_pub4wz: mean of 4 nested stacks (stack_pub5, wide LM, DavutKursun wall, dhananjayk2609 wide), + zones 1-4 (D31) | 22 | fit-free logit-rank average, zone rows sent to the bottom; up on 5 of 5 folds over stack_pubwz | | | **0.95273** | |
| stack_pubwz + 0.05-0.1 megayak 4-epoch RealMLP, weight chosen inside the folds (D31) | 22 | fit-free average + one member; up on 4 of 5 folds, within selection noise, superseded | | | 0.95265 | |
| stack_pubwz: 0.5 stack_pub5 + 0.5 wide LM stack, + zones 1-4 (D29) | 20 | fit-free logit-rank average, zone rows sent to the bottom | | | 0.95263 | |
| wide LM stack + zones 1-4 (D29; 2 of 5 folds against stack_pub5z) | 20 | public stack, zone rows sent to the bottom | | | 0.95261 | |
| stack: wide LM stack + 2 RealMLP (D21) | 9 | logistic (C=0.1) on 3 OOF | | | 0.95255 | |
| stack_pub5 + zero-fraud zones 1-4 (D27) | 19 | logistic (C=0.01) on 22 OOF, zone rows sent to the bottom | | | 0.95254 | |
| stack_pub5 + yekenot TabM (D27; not adopted, 3 of 5 folds) | 19 | logistic (C=0.01) on 23 OOF | | | 0.95251 | |
| stack_pub5 + 5 new public members (D27; not adopted, 3 of 5 folds) | 19 | logistic (C=0.01) on 27 OOF | | | 0.95250 | |
| stack: own 21 + wide LM stack (D21) | 9 | logistic (C=0.01) on 22 OOF | | | 0.95249 | |
| stack: own 21 honest boosters + wide LM stack (D22) | 11 | logistic (C=0.01) on 22 OOF | | | 0.95247 | |
| stack_pub5 + r21_gbx_rank_p1 (D30) | 21 | logistic (C=0.01) on 23 OOF; up on 1 of 5 folds, not adopted (rank-normalised p1n 0.95235, p8n 0.95230) | | | 0.95246 | |
| stack_pub5 + pub_mg_realmlp_4ep (D31) | 22 | logistic (C=0.01) on 23 OOF; up on 3 of 5 folds, not adopted | | | 0.95247 | |
| stack_pub5 + r20_fintfm_ro (D29) | 20 | logistic (C=0.01) on 23 OOF; up on 1 of 5 folds, not adopted | | | 0.95245 | |
| stack_pub5 + both Storyteller GPT-2 members (D30) | 21 | logistic (C=0.01) on 24 OOF; up on 2 of 5 folds, not adopted | | | 0.95240 | |
| stack: own 21 + public 156 (D21) | 9 | logistic (C=0.01) on 177 OOF | | | 0.95233 | |
| stack: own 21 + public 152 (D20) | 9 | logistic (C=0.01) on 173 OOF | | | 0.95225 | |
| stack: D20 without busyaprime's 27 boosters (D21) | 9 | logistic (C=0.01) on 146 OOF | | | 0.95219 | |
| stack: own 21 + public 52 (D19) | 9 | logistic (C=0.1) on 73 OOF | | | 0.95204 | |
| stack_r12 + zero-fraud zones 1-4 (own models, D27) | 19 | logistic (C=0.1) on 24 OOF, zone rows sent to the bottom | | | 0.95143 | |
| stack_r12 with r20_glm_tok_enc3 in place of r10_glm_tok (own models, D29) | 20 | logistic (C=0.1) on 24 OOF | | | 0.95140 | |
| stack of round twelve (own models, D23) | 12 | logistic (C=0.1) on 24 OOF: round eleven + r12_rmlp_emb, r12_rmlp_emb_te | | | 0.95138 | |
| stack_r12 + r20_fintfm_ro (own models, D29) | 20 | logistic (C=0.1) on 25 OOF; up on 3 of 5 folds, not adopted | | | 0.95135 | |
| stack_r12 + r21_gbx_rank_p1n or p8n (own models, D30) | 21 | logistic (C=0.1) on 25 OOF; not adopted | | | 0.95134 | |
| stack of round eleven (own models, honest boosters, D22) | 11 | logistic (C=0.1) on 22 OOF: round ten with the five r8 boosters swapped for r11 | | | 0.95133 | |
| stack of round ten (own models) | 10 | logistic (C=0.1) on 22 OOF: round eight + r10_glm_tok | | | 0.95132 | |
| stack of round eight (own models) | 8 | logistic (C=1) on 21 OOF | | | 0.95114 | |
| r10_gb_tok | 10 | GLM boost (LightGBM), not adopted (3 of 5 folds) | as r8_gb_all_kc + GPT-2 token keys and 3-digit chunks | 0 | 0.95110 | |
| r10_gb_tok_l3 | 10 | GLM boost (LightGBM, 3 leaves), not adopted (3 of 5 folds) | as r10_gb_tok | 0 | 0.95109 | |
| stack of round seven (own models) | 7 | logistic (C=1) on 14 OOF | | | 0.95107 | |
| r10_gbx_tok | 10 | GLM boost (XGBoost depth 2), not adopted (2 of 5 folds) | as r10_gb_tok | 0 | 0.95104 | |
| r9_gb_all_kc_s3 | 9 | GLM boost (LightGBM), 3 seeds | as r8_gb_all_kc | 0 | 0.95098 | 0.0032 |
| r10_gbc_tok | 10 | GLM boost (CatBoost depth 4), not adopted (3 of 5 folds) | as r10_gb_tok | 0 | 0.95097 | |
| r8_gb_all_kc | 8 | GLM boost (LightGBM), all round-seven features + key counts | artefacts + TE m 5/20/100 + joint keys | 0 | 0.95096 | 0.0032 |
| r8_gbx_all_kc | 8 | GLM boost (XGBoost depth 2) | as r8_gb_all_kc | 0 | 0.95094 | 0.0032 |
| r9_gbx_all_kc_s3 | 9 | GLM boost (XGBoost depth 2), 3 seeds | as r8_gb_all_kc | 0 | 0.95094 | 0.0032 |
| r11_gb_all_kc | 11 | GLM boost (LightGBM), --es inner, not adopted alone (bias check, D22) | as r8_gb_all_kc | 0 | 0.95092 | |
| r9_gbc_all_kc_s3 | 9 | GLM boost (CatBoost depth 4), 3 seeds | as r8_gb_all_kc | 0 | 0.95091 | 0.0032 |
| r8_gbc_all_kc | 8 | GLM boost (CatBoost depth 4) | as r8_gb_all_kc | 0 | 0.95090 | 0.0032 |
| r8_gb_all | 8 | GLM boost (LightGBM) | artefacts + TE m 5/20/100 + joint keys | 0 | 0.95087 | 0.0033 |
| r11_gbx_all_kc | 11 | GLM boost (XGBoost depth 2), --es inner, not adopted alone (bias check, D22) | as r8_gbx_all_kc | 0 | 0.95087 | |
| r20_glm_tok_enc3 | 20 | GLM, mean logit of three encoder seeds (D29) | as r10_glm_tok | 0 | 0.95087 | |
| r10_glm_tok | 10 | spline GLM, adopted (4 of 5 folds) | as r8_glm_all + GPT-2 token keys and 3-digit chunks | 0 | 0.95083 | |
| r11_gbc_all_kc | 11 | GLM boost (CatBoost depth 4), --es inner, not adopted alone (bias check, D22) | as r8_gbc_all_kc | 0 | 0.95082 | |
| r11_glm_tok_hier | 11 | spline GLM + hierarchical encoding (m 20), not adopted (1 of 5 folds) | as r10_glm_tok + `--hier 20` | 0 | 0.95080 | |
| stack of round five | 5 | logistic (C=0.1) on 9 OOF | | | 0.95079 | |
| r9_gb_all_s3 | 9 | GLM boost (LightGBM), 3 seeds | as r8_gb_all | 0 | 0.95087 | 0.0033 |
| r11_gb_all | 11 | GLM boost (LightGBM), --es inner, not adopted alone (bias check, D22) | as r8_gb_all | 0 | 0.95078 | |
| r7_gb_joint_kc | 7 | GLM boost; GLM has joint keys, trees see key counts | raw + TE | 0 | 0.95078 | 0.0032 |
| r7_gb_art_m3 | 7 | GLM boost | artefacts + TE at m 5/20/100 | 0 | 0.95076 | 0.0033 |
| stack of round four + r4c | 4 | logistic (C=0.1) on 7 OOF | | | 0.95071 | |
| r7_gb_m3 | 7 | GLM boost | TE at m 5/20/100 | 0 | 0.95068 | 0.0034 |
| r7_gb_art | 7 | GLM boost | artefacts + TE | 0 | 0.95066 | 0.0033 |
| r21_gbx_rank_p1n | 21 | XGBoost depth 2, GLM start, `rank:pairwise` (1 pair per row, Kaggle T4); OOF logit-ranked per fold; not adopted. Raw pooled OOF 0.94604 | as r11_gbx_all_kc | 0 | 0.95066 | 0.0033 |
| r21_gbx_rank_p8n | 21 | as r21_gbx_rank_p1n, 8 pairs per row; not adopted. Raw pooled OOF 0.94295 | as r11_gbx_all_kc | 0 | 0.95062 | 0.0033 |
| stack of round four | 4 | logistic (C=0.01) on 5 OOF | | | 0.95066 | |
| r7_gb_joint | 7 | GLM boost, joint keys in the GLM | raw + TE | 0 | 0.95063 | 0.0034 |
| stack of round three | 3 | logistic (C=0.01) on 6 OOF | | | 0.95063 | |
| r4_gb_s3 | 4 | GLM boost, 3 seeds averaged | as r3_glmboost | 0 | 0.95062 | 0.0034 |
| r7_gb_kc | 7 | GLM boost, key counts | raw + TE | 0 | 0.95060 | 0.0033 |
| r5_gb_spl8_s3 | 5 | GLM boost from the spline GLM, 3 seeds | as r3_glmboost | 0 | 0.95060 | 0.0033 |
| r6_gbc_d4_s3 | 6 | CatBoost depth 4 GLM boost, 3 seeds | as r3_glmboost | 0 | 0.95058 | 0.0034 |
| r4_gbc_d4 | 4 | CatBoost depth 4 started from the GLM's logit | as r3_glmboost | 0 | 0.95057 | 0.0034 |
| r3_glmboost | 3 | 4-leaf LightGBM started from the in-fold GLM's logit | raw + TE (m-estimate 20, full digit keys) | 0 | 0.95056 | 0.0034 |
| r4_gb_lr015 | 4 | GLM boost, lr 0.015, feature fraction 0.3 | as r3_glmboost | 0 | 0.95055 | 0.0034 |
| r8_glm_all | 8 | spline GLM | artefacts + TE m 5/20/100 + joint keys | 0 | 0.95062 | 0.0031 |
| r8_glm_all_kc | 8 | spline GLM | as r8_glm_all + key counts | 0 | 0.95062 | 0.0032 |
| r7_glm_art_m3 | 7 | spline GLM | artefacts + TE at m 5/20/100 | 0 | 0.95053 | 0.0032 |
| r9_lgb_l4_all_kc_s3 | 9 | LightGBM, 4 leaves, 3 seeds | as r8_gb_all_kc | 0 | 0.95052 | 0.0034 |
| r4_gb_l8 | 4 | GLM boost, 8 leaves | as r3_glmboost | 0 | 0.95050 | 0.0034 |
| r8_lgb_l4_all_kc | 8 | LightGBM, 4 leaves | as r8_gb_all_kc | 0 | 0.95050 | 0.0034 |
| r4_gbx_d2 | 4 | XGBoost depth 2 started from the GLM's logit | as r3_glmboost | 0 | 0.95049 | 0.0034 |
| r6_gbc_d6 | 6 | CatBoost depth 6 started from the GLM's logit | as r3_glmboost | 0 | 0.95048 | 0.0034 |
| r11_lgb_l4_all_kc | 11 | LightGBM, 4 leaves, --es inner, not adopted alone (bias check, D22) | as r8_lgb_l4_all_kc | 0 | 0.95043 | |
| r7_glm_joint_kc | 7 | spline GLM | joint keys + key counts | 0 | 0.95043 | 0.0032 |
| r7_lgb_l4_art | 7 | LightGBM, 4 leaves | artefacts + TE | 0 | 0.95040 | 0.0035 |
| r7_glm_m3 | 7 | spline GLM | TE at m 5/20/100 | 0 | 0.95037 | 0.0033 |
| r5_lgb_l4_s3 | 5 | LightGBM, 4 leaves, 3 seeds | as r3_lgb_l4 | 0 | 0.95033 | 0.0034 |
| r7_glm_art | 7 | spline GLM | artefacts + TE | 0 | 0.95028 | 0.0032 |
| r7_glm_joint | 7 | spline GLM | joint keys | 0 | 0.95028 | 0.0033 |
| r7_glm_kc | 7 | spline GLM | key counts | 0 | 0.95028 | 0.0032 |
| r7_glm_pairs | 7 | spline GLM | 45 pair keys | 0 | 0.95024 | 0.0033 |
| r3_lgb_l4 | 3 | LightGBM, 4 leaves | raw + TE (m-estimate 20, full digit keys) | 0 | 0.95023 | 0.0035 |
| r4_glm_spl8 | 4 | GLM + cubic splines (8 knots) on numeric columns | logit TE + one-hot | 0 | 0.95015 | 0.0033 |
| r5_glm_spl16 | 5 | GLM + splines (16 knots) | logit TE + one-hot | 0 | 0.95006 | 0.0033 |
| r17_rmlp_emb_all_bag3 | 17 | mean of r16_rmlp_emb_all seeds 0-2; 4 of 5 folds up, 1 level; stack flat (0.95139), not adopted | as r12_rmlp_emb | 0 | 0.94996 | 0.0033 |
| r3_glm | 3 | L2 logistic, C=0.01 | logit TE + one-hot | 0 | 0.94987 | 0.0034 |
| r4_glm_c03 | 4 | GLM, C=0.03 | logit TE + one-hot | 0 | 0.94987 | 0.0034 |
| r16_rmlp_emb_all | 16 | RealMLP on all training rows (no 10% holdout), 5 of 5 folds above r12_rmlp_emb; stack flat (0.95137), not adopted | as r12_rmlp_emb | 0 | 0.94986 | 0.0034 |
| r4_glm_cv10 | 4 | GLM, 10-way inner cross-fit | logit TE + one-hot | 0 | 0.94985 | 0.0034 |
| r17_rmlp_emb_te_all_bag3 | 17 | mean of r16_rmlp_emb_te_all seeds 0-2; 4 of 5 folds up; stack flat, not adopted | as r12_rmlp_emb_te | 0 | 0.94978 | 0.0036 |
| r4_glm_c003 | 4 | GLM, C=0.003 | logit TE + one-hot | 0 | 0.94975 | 0.0034 |
| r16_rmlp_emb_te_all | 16 | RealMLP on all training rows, 5 of 5 folds above r12_rmlp_emb_te; stack flat, not adopted | as r12_rmlp_emb_te | 0 | 0.94967 | 0.0035 |
| stack of round two | 2 | logistic on 7 OOF | | | 0.94952 | |
| r12_rmlp_emb | 12 | RealMLP, adopted with r12_rmlp_emb_te (stack 4 of 5 folds) | raw + embedded integer parts and decimals (`--fe emb`) | 0 | 0.94949 | 0.0034 |
| r12_rmlp_emb16 | 12 | RealMLP, 16 networks, not adopted (stack 0.95131) | as r12_rmlp_emb | 0 | 0.94936 | 0.0034 |
| r12_rmlp_art | 12 | RealMLP (pytabkit, 8 nets x 3 epochs), not adopted (stack 0.95128) | as r8_gb_all_kc, TE m 20 only | 0 | 0.94938 | 0.0035 |
| r12_rmlp_emb_te | 12 | RealMLP, adopted with r12_rmlp_emb | `--fe emb` + basic-key TE m 20 | 0 | 0.94911 | 0.0036 |
| xgb_digits_te2_d2 | 2 | XGBoost, depth 2 | raw + digits + TE + pairs | 0 | 0.94926 | 0.0034 |
| lgb_digits_te2_l4 | 2 | LightGBM, 4 leaves | raw + digits + TE + pairs | 0 | 0.94923 | 0.0033 |
| r12_rmlp_raw | 12 | RealMLP, not adopted (stack 0.95132) | raw + basic-key TE m 20 | 0 | 0.94889 | 0.0037 |
| lgb_digits_te1 | 2 | LightGBM, 31 leaves | raw + digits + TE (exact values, rounded keys, cents) | 0 | 0.94854 | 0.0035 |
| r16_rmlp_emb_all_e5 | 16 | as r16_rmlp_emb_all with 5 epochs; overfits, not adopted | as r12_rmlp_emb | 0 | 0.94854 | 0.0035 |
| lgb_digits_te2 | 2 | LightGBM, 31 leaves | as above + 45 top-10 pairs | 0 | 0.94845 | 0.0034 |
| lgb_raw_l4 | 2 | LightGBM, 4 leaves | raw | 0 | 0.94752 | 0.0030 |
| stack of round one | 1 | logistic on 8 OOF | | | 0.94728 | |
| r6_mlp_lr3e4 | 6 | MLP 128-64, learning rate 3e-4, 3 seeds | as r5_mlp | 0 | 0.94715 | 0.0034 |
| r5_mlp | 5 | MLP 64-32 on the spline GLM design, 3 seeds | logit TE + one-hot + splines | 0 | 0.94715 | 0.0035 |
| lgb_digits | 2 | LightGBM, 31 leaves | raw + first two decimal digits | 0 | 0.94712 | 0.0032 |
| lgb_raw_o0 | 1 | LightGBM, 31 leaves | raw | 0 | 0.94690 | 0.0031 |
| xgb_raw_o1 | 1 | XGBoost, depth 6 | raw | 1 | 0.94681 | 0.0033 |
| lgb_raw_o1 | 1 | LightGBM, 31 leaves | raw | 1 | 0.94680 | 0.0032 |
| xgb_ogte_o1 | 1 | XGBoost, depth 6 | raw + original per-value rates | 1 | 0.94676 | 0.0036 |
| r8_mlp_all | 8 | MLP, 3 seeds | as r8_glm_all | 0 | 0.94666 | 0.0036 |
| lgb_combo_o1 | 1 | LightGBM, 31 leaves | + ratios, flag count | 1 | 0.94659 | 0.0034 |
| cat_ogte_o1 | 1 | CatBoost, depth 6 | raw + original per-value rates | 1 | 0.94654 | 0.0035 |
| lgb_count_o1 | 1 | LightGBM, 31 leaves | + value frequencies | 1 | 0.94647 | 0.0036 |
| lgb_ogte_o1 | 1 | LightGBM, 31 leaves | raw + original per-value rates | 1 | 0.94633 | 0.0034 |
| cat_raw_o1 | 1 | CatBoost, depth 6 | raw | 1 | 0.94610 | 0.0031 |
| r5_mlp_a1e2 | 5 | MLP 32, alpha 0.01, 3 seeds | as r5_mlp | 0 | 0.94578 | 0.0037 |
| lgb_ogmodel_o1 | 1 | LightGBM, 31 leaves | + original-only model prediction | 1 | 0.94564 | 0.0037 |
| r13_tabm_emb | 13 | TabM (pytabkit, CPU), stopped at the 12 h kernel limit after fold 0; not adopted | as r12_rmlp_emb | 0 | fold 0 only: 0.93680 | |
| r14_gpt2 | 14 | own target-first GPT-2 (RunPod A40, one epoch, 2 OOF orders); stack weight about 0, not adopted | rows as text | 0 | 0.94396 | 0.0037 |
| r20_fintfm_ro | 20 | fintfm 1.0.0 linear readout, entry defaults (three Kaggle CPU kernels, folds merged); not adopted | raw columns | 0 | 0.94054 | 0.0033 |
| lc_0.25 | 21 | r11_gb_all_kc settings on 25% of each training fold, folds 0-1 only (learning curve, D30) | as r11_gb_all_kc | 0 | 0.94691 (2 folds) | |
| lc_0.5 | 21 | r11_gb_all_kc settings on 50% of each training fold, folds 0-1 only (learning curve, D30) | as r11_gb_all_kc | 0 | 0.94896 (2 folds) | |
| lc_0.75 | 21 | r11_gb_all_kc settings on 75% of each training fold, folds 0-1 only (learning curve, D30) | as r11_gb_all_kc | 0 | 0.94993 (2 folds) | |
| r18_gpt2_e3 | 18 | own target-first GPT-2 (RunPod A40, three epochs, 4 OOF orders, 2 test orders); stack weight 0.18 but stack CV lower, not adopted | rows as text | 0 | 0.94717 | 0.0035 |

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
  about 0.0028 worse in OOF. "GLM" is a linear model; "LM" is a target-first language model (corrected in D19).
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

## D14. Round six, and the search for more artefacts

Round six (`jobs/r6b.txt`): CatBoost GLM boost at depth 6 scores 0.95048 (depth 4: 0.95057), dropped; a
slower, wider MLP scores 0.94715, the same as round five's. `jobs/r6a.txt`: CatBoost GLM boost with 3 seeds scores
0.95058 (one seed: 0.95057, 3 of 5 folds up); swapped into the round-five stack it gives 0.95078 against
0.95079. No gain; the single-seed member stays.

`src/research_artefacts.py` measures pooled effects that per-value encodings cannot see, because each rare
value is smoothed towards the base rate on its own (base fraud rate 1.670%):

- **Values the original data never contains.** 2.6% of `amount_usd` values (fraud 2.42% against 1.65%);
  the 178 `velocity_score` and 91 `merchant_risk_score` values outside the original are fraud at 5.6% and
  5.5%. 15% of `account_balance_usd` values are new, with no effect (1.71% against 1.66%).
- **Pairs copied from one original row.** 105,697 rows repeat a (`velocity_score`, `merchant_risk_score`)
  pair of one original row: fraud 0.81%. (`distance_from_home_km`, `velocity_score`): 26,196 rows, 0.56%.
  No row repeats a whole original row, and no feature row occurs twice in train + test.
- **How often a value occurs.** `velocity_score` and `merchant_risk_score` values that occur 20-100 times in
  train + test are fraud at 10.3%; rare `amount_usd` values (at most 20 occurrences) at 3.2-3.9%.

These become `--fe art` (log counts, "occurs in the original" flags and 15 copied-pair flags; no labels
used). Round seven (`jobs/r7a.txt` to `r7e.txt`) tests them one change at a time, together with encodings at
three smoothing weights (`--te-smooth 5,20,100`), pair keys for the GLM, and two ideas measured in megayak's
public notebook (Apache-2.0), written here independently: joint keys of non-additive column pairs
(`--joint`; merchant category x risk score is the strongest) and the count of every key's value
(`--key-counts`).

## D15. Public OOF members (the second track)

Several public datasets publish OOF and test predictions on the same split as ours
(`StratifiedKFold(5, shuffle=True, random_state=42)` on `train.csv` order). `src/import_public.py` imports
them as `runs/pub_*` after checking shapes, finite values, that the source's fold file equals our folds row
by row, and that our AUC matches the one the source reports. 38 members pass: goodpjw2008's 20 core members
(CC0), mathsonchain's 17 (CC BY 4.0) and masayakawamata's GPT-2 target-first classifier (CC BY 4.0).
alexchilton's 39 ship no fold file, so they are not used.

| Stack (logistic, C=0.01) | Members | Nested CV | Folds |
| --- | --- | --- | --- |
| ours (D13) | 9 | 0.95079 | 0.95006 0.95156 0.95575 0.95178 0.94538 |
| public only | 38 | 0.95165 | 0.95077 0.95230 0.95676 0.95192 0.94686 |
| ours + public | 47 | **0.95175** | 0.95081 0.95230 0.95682 0.95221 0.94696 |

The GPT-2 member, a language model reading each row as text, takes weight 0.55 of 47: it errs differently
from every tree and GLM. The public members' authors chose their recipes on these same folds, so part of the
+0.001 may be selection. Written to `submissions/stack_pub1.csv`; the two final picks will be one stack of
our own models and one with public members.

## D16. Round seven: artefacts, three smoothing weights, joint keys, key counts

One change against round four (spline GLM `r4_glm_spl8` 0.95015; GLM boost `r3_glmboost` 0.95056):

| Change | GLM | folds up | GLM boost | folds up |
| --- | --- | --- | --- | --- |
| artefact features (`--fe art`) | 0.95028 | 3 | 0.95066 | 2 |
| TE at m 5, 20 and 100 | 0.95037 | 5 | 0.95068 | 4 + 1 tie |
| both | **0.95053** | 5 | **0.95076** | 5 |
| joint keys (GLM only) | 0.95028 | 5 | 0.95063 | 3 |
| key counts | 0.95028 | 3 | 0.95060 | 3 |
| joint keys + key counts | 0.95043 | 5 | 0.95078 | 3 |
| 45 pair keys | 0.95024 | 5 | | |
| 4-leaf trees with artefacts (against r5_lgb_l4_s3 0.95033) | | | 0.95040 (plain trees) | 3 |

Each idea helps the GLM, which cannot build interactions or count effects itself; the trees gain less. The
pooled artefacts and the multi-weight encodings are complementary (+0.00038 together on the GLM, 5 of 5).
Adding five round-seven runs to the round-five stack: nested CV **0.95107** (logistic C=1; folds 0.95021
0.95173 0.95606 0.95193 0.94591), +0.00028, all five folds up.

Round eight (`jobs/r8a.txt` to `r8d.txt`) puts all of it together: artefacts, m 5/20/100, joint keys in
the GLM, with and without key counts, for the GLM, LightGBM, XGBoost and CatBoost GLM boost, plain 4-leaf
trees and the MLP.

## D17. Round eight: everything combined, every family

Shared inputs: pooled artefacts, TE at m 5, 20 and 100 on the full digit keys, joint keys in the GLM, with and
without key counts (`jobs/r8a.txt` to `r8d.txt`). Each run against the member it would replace:

| Run | CV AUC | Against | Folds up | Verdict |
| --- | --- | --- | --- | --- |
| r8_gb_all_kc | **0.95096** | r7_gb_joint_kc 0.95078 | 5 | adopted; best single model |
| r8_gbx_all_kc | 0.95094 | r4_gbx_d2 0.95049 | 5 | adopted |
| r8_gbc_all_kc | 0.95090 | r4_gbc_d4 0.95057 | 5 | adopted |
| r8_gb_all | 0.95087 | r7_gb_joint_kc 0.95078 | 5 | stack member |
| r8_glm_all | 0.95062 | r7_glm_art_m3 0.95053 | 4 | adopted |
| r8_glm_all_kc | 0.95062 | r7_glm_art_m3 0.95053 | 4 | stack member (3 of 5 against r8_glm_all) |
| r8_lgb_l4_all_kc | 0.95050 | r5_lgb_l4_s3 0.95033 | 4 | adopted |
| r8_mlp_all | 0.94666 | r5_mlp 0.94715 | | dropped |

The round-seven gains carry over to XGBoost and CatBoost GLM boost (+0.00045 and +0.00033), which had not
had them before; key counts add +0.00009 to LightGBM GLM boost (4 of 5 folds) and nothing to the GLM.
The MLP loses 0.00049 on the larger design.

Stacks (nested CV, folds in order):

| Stack | Members | CV AUC | Folds | Against |
| --- | --- | --- | --- | --- |
| own models | 21 (round-seven 14 + the seven adopted or member runs above) | **0.95114** (C=1) | 0.95022 0.95183 0.95609 0.95201 0.94594 | 0.95107, 5 of 5 up |
| own + r8_mlp_all | 22 | 0.95114 | | no gain |
| ours + public | 21 + 38 public | **0.95183** (C=0.01) | 0.95091 0.95229 0.95689 0.95231 0.94704 | stack_pub1 0.95175, 4 of 5 up |

## D18. Round nine: seed bags (step 4)

Each round-eight tree member retrained with 3 seeds averaged (`jobs/r9a.txt` to `r9d.txt`):

| Run | CV AUC | Against | Folds up |
| --- | --- | --- | --- |
| r9_gb_all_kc_s3 | 0.95098 | r8_gb_all_kc 0.95096 | 3 |
| r9_gbx_all_kc_s3 | 0.95094 | r8_gbx_all_kc 0.95094 | 2 |
| r9_gbc_all_kc_s3 | 0.95091 | r8_gbc_all_kc 0.95090 | 2 |
| r9_gb_all_s3 | 0.95087 | r8_gb_all 0.95087 | 2 |
| r9_lgb_l4_all_kc_s3 | 0.95052 | r8_lgb_l4_all_kc 0.95050 | 5 |

Seed averaging moves single models by at most +0.00002: with 4-leaf trees, low learning rates and a GLM
start, seed variance is already small. In the stacks it does not help:

| Stack | Own models | Ours + public |
| --- | --- | --- |
| round eight (D17) | **0.95114** | **0.95183** |
| seed bags in place of the round-eight runs | 0.95112 | 0.95179 |
| seed bags added alongside | 0.95113 | 0.95182 |

Dropped. The final candidates stay `stack_r8.csv` (own models, 0.95114) and `stack_pub2.csv` (with
public members, 0.95183).

## D19. More public members: the language models behind the 0.954 notebooks

The best CV-backed public file we know of, `submission_cv.csv` of goodpjw2008's
[LM Ladder](https://www.kaggle.com/code/goodpjw2008/s6e11-lm-ladder-ladder-meets-stack-lb-0-95432)
(read as text, not run), reports OOF 0.95266. That figure is not a nested stack score: its base is Aman Atar's
public 0.95370 file, a blend tuned on the public board, "replayed" on OOF rows with weights fitted on test rows,
plus steps whose sizes were partly chosen on all five folds. Its members are public, and "LM" there is a
target-first language model (GPT-2 small reading each row as text, label token first), not a linear model as
D7 said. Three CC0 datasets carry them:

- [amanatar/s6e11-rank1-q17-meta-bundle](https://www.kaggle.com/datasets/amanatar/s6e11-rank1-q17-meta-bundle):
  16 members' OOF and test predictions as clipped logit ranks, with labels and a fold array; both equal ours
  row by row. 12 imported (`pub_q17_*`): megayak's compact LM, two TabPFN, RealMLP, three GLMs, a VAE, and
  four GLM-boost and shallow-tree members. Skipped: the two members already imported, and two LMs whose bundle
  test predictions come from the fold-0 model only.
- goodpjw2008's [compact LM, seed 7](https://www.kaggle.com/datasets/goodpjw2008/s6e11-compact-lm-seed7)
  (0.94945) and [GPT-2, seed 2026](https://www.kaggle.com/datasets/goodpjw2008/s6e11-target-first-gpt2-seed-2026)
  (0.95014), with test predictions averaged over the five fold models (`pub_gw_*`). Their per-fold AUCs equal
  those of the bundle's copies, which carry the bundle's (our) folds.

Single members range from 0.94649 to 0.95159; four score above our best single model (0.95096). Stacks on
the same members as D17 plus these 14 (nested CV, folds in order):

| Stacker | CV AUC | Folds |
| --- | --- | --- |
| rank average | 0.95130 | |
| logistic C=0.01 | 0.95199 | 0.95101 0.95231 0.95708 0.95236 0.94749 |
| logistic C=0.1 | **0.95204** | 0.95102 0.95235 0.95723 0.95232 0.94751 |
| logistic C=1 | 0.95199 | 0.95120 0.95221 0.95711 0.95231 0.94734 |
| hill climbing | 0.95202 | 0.95086 0.95244 0.95711 0.95249 0.94761 |

+0.00021 over D17's 0.95183, all five folds up. The three GPT-2-family members share the language-model
weight (0.30, 0.25, and 0.11 for megayak's LM). Written to `submissions/stack_pub3.csv`. The same caveat as
D15 applies, more strongly: many of these authors chose recipes and blends on these folds and on the public
board, so part of the gain may be selection that the private rows will not repeat.

## D20. Three more public sources

All CC0, imported by `src/import_public.py` with every member's AUC equal to its source's manifest within
0.0001:

- [busyaprime/s6e11-33-own-members](https://www.kaggle.com/datasets/busyaprime/s6e11-33-own-members): 52
  members (GLM-start boosters, shallow trees and GLMs on digit and GPT-2 token keys, EBMs, TabM, RealMLP),
  0.94835-0.95109. No fold file; the description states the shared split, which by construction equals ours
  ("folds stated").
- dariushafshar's [9-model library](https://www.kaggle.com/datasets/dariushafshar/s6e11-golem-oof-library)
  and [3 GPU members](https://www.kaggle.com/datasets/dariushafshar/s6e11-gpu-oof-members) (digit Transformer,
  target-first digit LM, CatBoost), fold file equal to ours. Members a, h and i are left out: the manifest
  says they early-stop on the validation fold, so their OOF is optimistic. 0.93117-0.94659.
- [alexchilton/s6e11-oof-predictions](https://www.kaggle.com/datasets/alexchilton/s6e11-oof-predictions):
  39 members, 0.89930-0.95066; the description now states the shared split.

Stacks on D19's 73 members plus (nested CV; best C per row):

| Added | Members | Best C | CV AUC | Folds |
| --- | --- | --- | --- | --- |
| busyaprime | 125 | 0.1 | 0.95215 | 0.95132 0.95239 0.95728 0.95259 0.94755 |
| busyaprime + dariushafshar | 134 | 0.01 | 0.95211 | 0.95117 0.95243 0.95724 0.95259 0.94753 |
| busyaprime + alexchilton | 164 | 0.003 | 0.95218 | 0.95127 0.95242 0.95706 0.95256 0.94784 |
| all three | 173 | 0.01 | **0.95225** | 0.95138 0.95242 0.95718 0.95265 0.94782 |

busyaprime adds +0.00011 (5 of 5 folds); after that the variants differ by less than the spread across C,
so the public-member stack has levelled off near 0.9522. Written: `submissions/stack_pub4.csv` (all three,
C=0.01).

Own-model side (round ten, `jobs/r10a.txt` to `r10d.txt`): GPT-2 token keys and 3-digit chunks of each
float column's integer part (`--keys tokens`; idea after aphoristicartist and busyaprime, written here
independently) on the GLM and the three GLM-boost families.

Round ten results (fold AUCs against round eight):

| Run | CV AUC | vs | Change | Folds up |
| --- | --- | --- | --- | --- |
| r10_glm_tok | 0.95083 | r8_glm_all 0.95062 | +0.00022 | 4 of 5 (adopted) |
| r10_gb_tok | 0.95110 | r8_gb_all_kc 0.95096 | +0.00013 | 3 of 5 (folds 4 and 5 carry it) |
| r10_gb_tok_l3 | 0.95109 | r8_gb_all_kc 0.95096 | +0.00012 | 3 of 5 |
| r10_gbx_tok | 0.95104 | r8_gbx_all_kc 0.95094 | +0.00010 | 2 of 5 |
| r10_gbc_tok | 0.95097 | r8_gbc_all_kc 0.95090 | +0.00007 | 3 of 5 |

Own-model stack with r10_glm_tok added: **0.95132** (C=0.1; round eight 0.95114), up on 4 folds, level on
fold 2. Adding the boosters as well gives 0.95127 (two LightGBM) and 0.95128 (all four). Written: `submissions/stack_r10.csv`. With the wide
LM stack on top (D21) the three round-ten members give 0.95241, below the wide stack alone. In every booster family the token
keys lose on folds 1 and 2 and gain about 0.0004 on folds 4 and 5.

## D21. The nested wide stack, RealMLP v15, and an early-stopping check

Read as text, 2026-11-17: asteriospanoras's *Top Public, Fully Explained*, kagankoral's *Final-Pick Check*,
busyaprime's *124 Voices* and goodpjw2008's *Board Model + RealMLP*. All four agree the 0.954 files are
public-board steering that nested CV does not support, and point to the same CV reference: goodpjw2008's
nested wide LM stack.

New members (`src/import_public.py`, folds stated as the shared split):

- [goodpjw2008/s6e11-fraud-series-artifacts](https://www.kaggle.com/datasets/goodpjw2008/s6e11-fraud-series-artifacts)
  (CC0): the nested OOF of the wide LM stack (about 200 public members, **0.95257**, better than our D20
  stack on all five folds) and of the wide stack without LMs (0.95250). The single models in the dataset
  duplicate the OOF bank and are not imported.
- yekenot's RealMLP notebook outputs: PyTabKit v15 (0.95031; 3 epochs, no early stopping, every fold
  improves to the last epoch, so the validation fold does not pick the model) and the earlier PyTorch run
  (0.95002).

Stacks (nested CV, best C per row):

| Members | Best C | CV AUC | Folds |
| --- | --- | --- | --- |
| wide LM stack alone (no fit) | | **0.95257** | 0.95177 0.95253 0.95737 0.95270 0.94845 |
| wide LM stack + 2 RealMLP | 0.1 | 0.95255 | 0.95178 0.95263 0.95744 0.95269 0.94836 |
| own 21 + 2 wide stacks + 2 RealMLP | 0.01 | 0.95251 | 0.95185 0.95268 0.95728 0.95288 0.94813 |
| own 21 + wide LM stack | 0.01 | 0.95249 | 0.95180 0.95260 0.95727 0.95294 0.94801 |
| D20's 173 + 2 wide stacks + 2 RealMLP | 0.01 | 0.95233 | 0.95164 0.95251 0.95723 0.95260 0.94789 |
| D20's 173 (D20) | 0.01 | 0.95225 | 0.95138 0.95242 0.95718 0.95265 0.94782 |
| D20 without busyaprime's 27 boosters | 0.01 | 0.95219 | 0.95138 0.95232 0.95711 0.95251 0.94785 |

Fit-free steps from the wide LM stack towards one direction, `W + w (X - W)` on logit ranks, w chosen on
four folds and scored on the fifth: towards the mean of our 21 models -2.5e-5 (2 of 5 folds up), our best
single model -0.6e-5 (3/5), our best GLM -1.1e-5 (2/5), RealMLP v15 +0.5e-5 (4/5), the wide stack without
LMs -1.0e-5 (0/5).

Reading:

- Nothing beats the wide LM stack on its own. Refitting weights on top of it costs more than the
  members add, as kagankoral measured for the bundle's 16 models; our own 21 models do not add to it
  either, by stack or by step.
- busyaprime's notebook says its boosters early-stop on the fold they are scored on. Without those 27
  members D20 drops 0.00006, so `stack_pub4` (public 0.95219) carries at most that much optimism.
- The public-member track is now limited by the wide stack, not by our members. Written:
  `submissions/stack_pub5.csv` (own 21 + wide LM stack, C=0.01), the best file that still contains our
  models.

## D22. Our boosters early-stopped on the validation fold

Reading `train_gbdt.py` after D21: every LightGBM, XGBoost and CatBoost run so far early-stopped on the
fold it was then scored on (`eval_set` = the validation fold, patience 200), the same flaw for which D20
and D21 left out public members. Each OOF is optimistic by the gain of picking the round count on its own
labels, and the stacker trusts those members a little too much.

Fix, `--es inner`: the rounds are picked on a stratified 10% holdout of the training fold, then the model
is refit on the whole training fold for that count / 0.9 rounds; the validation fold is used only for
scoring. `--es val` (the default) keeps the old behaviour so earlier runs stay reproducible. Smoke test on
40,000 rows (LightGBM GLM boost): 0.93434 with `val`, 0.93388 with `inner`.

Round eleven (`jobs/r11a.txt` to `r11d.txt`) reruns the five round-eight boosters with `--es inner` to
measure the bias on full folds.

Hierarchical encoding (`--hier m`, idea after busyaprime's "hier" members): each float column's keys form
a chain, tens -> integer part -> 0.1 steps -> exact value, and each level's fraud rate is shrunk towards
its parent's rate instead of the overall rate (cross-fitted inside the training fold). On the token-key GLM
at m 20: 0.95080 against 0.95083, folds -10 -1 -1 +3 -6 (x 1e-5). Not adopted: the three flat m-estimate
encodings of every level already carry what the chain adds.

Round-eleven result: every booster loses a little when its rounds are picked away from the scored fold.

| Run (`--es inner`) | CV | Round eight (`--es val`) | Change | Folds lower |
| --- | --- | --- | --- | --- |
| r11_gb_all_kc | 0.95092 | 0.95096 | -0.00004 | 5 of 5 |
| r11_gbx_all_kc | 0.95087 | 0.95094 | -0.00007 | 5 of 5 |
| r11_gb_all | 0.95078 | 0.95087 | -0.00009 | 5 of 5 |
| r11_gbc_all_kc | 0.95082 | 0.95090 | -0.00007 | 4 of 5 |
| r11_lgb_l4_all_kc | 0.95043 | 0.95050 | -0.00007 | 5 of 5 |

So our booster CVs were inflated by about 0.00004 to 0.00009, the same size as busyaprime's (D21). The
stacks do not move: the own-model stack with the five honest boosters in place of the round-eight ones
scores 0.95133 (C=0.1; folds 0.95031 0.95183 0.95615 0.95235 0.94634) against 0.95132, and own 21 honest
+ wide LM stack scores 0.95247 against 0.95249. The stacker only uses where members disagree, and the
inflation shifts each booster's OOF about evenly, so the stack CVs reported so far stand. The honest
own-model stack is written as `submissions/stack_r11.csv` (Spearman 0.99998 with `stack_r10.csv`); it is
the cleaner own-model final at the same CV. The older r4/r5/r7 boosters are not rerun: the measured effect
on the stack is nil.

## D23. Our own RealMLP

Adding yekenot's two public RealMLP outputs to the 22-model own stack (D22, 0.95133) gives 0.95147, up on
all five folds (C=0.1), so a RealMLP of our own looked worth building. `--model realmlp` in
`train_gbdt.py` wraps pytabkit's RealMLP_TD_Classifier on the same fold frames as every other model. pytabkit
picks its best epoch on a validation set; ours is a stratified 10% holdout of the training rows, never the
scored fold (the public notebook passes the scored fold, which can flatter its OOF the way early stopping
did in D22). Hyperparameters start from that notebook's tuned values (read as text), with 8 networks
instead of 16 to fit a Kaggle CPU kernel: about 20 minutes a fold with 127 inputs.

| Run | Inputs | CV | Folds | Stack with it (C=0.1) |
| --- | --- | --- | --- | --- |
| r12_rmlp_art | artefacts + digit-key TE (127 columns) | 0.94938 | 0.94827 0.94987 0.95496 0.95036 0.94422 | 0.95125 |
| r12_rmlp_raw | raw + basic-key TE | 0.94889 | 0.94769 0.95040 0.95443 0.94959 0.94304 | 0.95132 |
| both | | | | 0.95130 |

Neither helps the stack. Fed target encodings, the network repeats what the GLM and boosters already see
(rank correlation with r11_gb_all_kc 0.976); the public RealMLP correlates only 0.973 to 0.977 with ours,
because its inputs differ: the integer part of every numeric column and the decimals as categories that the
network embeds. `--fe emb` builds that input (our code), and r12c (no TE) and r12d (+ basic-key TE) test it.

| Run | Inputs | CV | Folds | Rank corr. with public RealMLP | Stack with it (C=0.1) |
| --- | --- | --- | --- | --- | --- |
| r12_rmlp_emb | `--fe emb` | 0.94949 | 0.94892 0.95047 0.95453 0.95000 0.94399 | 0.990 | 0.95135 |
| r12_rmlp_emb_te | `--fe emb` + basic-key TE | 0.94911 | 0.94854 0.95032 0.95478 0.94964 0.94342 | 0.985 | 0.95125 |
| both | | | | | **0.95138** |

With both, the own stack reaches 0.95138 (folds 0.95050 0.95186 0.95620 0.95240 0.94628), up on 4 of 5
folds against 0.95133 (fold 5 down 0.00006): adopted by the rule, written as `submissions/stack_r12.csv`
(Spearman 0.99946 with `stack_r11.csv`). The embedded input reproduces the public network (0.990) but only
a third of its stack gain (+0.00005 against +0.00014). The rest is 16 networks against 8, training on all
rows against 90%, or the public notebook picking its epoch on the scored fold; these outputs cannot tell
which.

Ensemble size is not it: r12_rmlp_emb16 (16 networks, `jobs/r12e.txt`) scores 0.94936 (folds 0.94881
0.95056 0.95469 0.94931 0.94409) against 0.94949 with 8, and in the stack 0.95131 alone or 0.95134 with
r12_rmlp_emb_te (C=0.1), below `stack_r12`. Not adopted. What remains of the gap is the 10% holdout and
the public notebook's epoch choice on the scored fold; we keep the honest version.

Public score of `stack_r12.csv`: 0.95160 against 0.95134 for `stack_r10.csv`, +0.00026 for +0.00006 of CV.
The RealMLPs help the test set more than the folds show: neural members are the direction to extend.

TabM (`jobs/r13a.txt`, `TabM_D_Classifier`, batch 1024, 12 epochs, same `--fe emb` input and 10% holdout)
does not fit on a CPU kernel: fold 0 took 6.7 hours and scored 0.93680 against 0.94892 for r12_rmlp_emb on
the same fold, and Kaggle stopped the kernel at 12 hours during fold 1. Dropped; a GPU run would be needed
to tune it, and the GPU allowance goes to the GPT-2 member (D24).

**Round sixteen: all training rows.** With early stopping off and three fixed epochs, the 10% holdout only
chose among three epochs, so `--params {"holdout":0,...}` now trains on all training rows and gives
pytabkit a 5% subset of those same rows as its validation set (still never the scored fold):

| Run | CV | Folds | Against round twelve |
| --- | --- | --- | --- |
| r16_rmlp_emb_all | 0.94986 | 0.94931 0.95077 0.95485 0.95026 0.94436 | +0.00037, 5 of 5 folds |
| r16_rmlp_emb_te_all | 0.94967 | 0.94892 0.95079 0.95519 0.95023 0.94418 | +0.00056, 5 of 5 folds |
| r16_rmlp_emb_all_e5 (5 epochs) | 0.94854 | 0.94790 0.94969 0.95365 0.94922 0.94268 | -0.00095; overfits |

Better members, but no stack gain: swapping the pair into `stack_r12` gives 0.95137 (C=1; 0.95136 at
C=0.1) against 0.95138, adding them beside the round-twelve pair 0.95132, and adding them to `stack_pub5`
0.95243 against 0.95249. Not adopted by the CV rule. The swapped stack is a candidate for a public check
only, since the round-twelve RealMLPs helped the public score more than CV showed.

Public check (2026-11-19, owner's yes): `submissions/stack_r16.csv` (the swapped stack at C=0.1, CV 0.95136)
scores 0.95163 public against 0.95160 for `stack_r12`. A difference of 0.00003 on about 60,000 public rows
is noise, and CV is 0.00002 lower, so `stack_r12` stays the own-model final pick.


## D24. Own target-first GPT-2 member (not adopted)

The strongest single public members are GPT-2 models that read each row as text (about 0.9500 alone), and
the RealMLPs showed that members unlike our boosters help the public score more than CV shows (D23).
`src/train_lm.py` is our own implementation of the recipe in Chris Deotte's S6E10 first-place write-up:
GPT-2 is fine-tuned on rows written as "fraud yes, amount 61.16, hours 9.77, ..." with the label first and
the 24 feature fragments in a new random order each time; a held-out row is scored with "yes" and with "no",
and the difference of the two sequence log-likelihoods is its log-odds, averaged over fixed orders.
Training runs for a fixed time (90 minutes a fold) or one epoch, never stopped on the scored fold.

On a T4 x2 kernel the folds run two at a time, one process per GPU. The owner allowed one run of about 10
GPU hours (2026-11-18), after the weekly quota resets on 2026-11-21. Smoke test (CPU, tiny random GPT-2,
3,000 rows, two workers): runs end to end, AUC 0.73 and 0.78 after one minute of training.

**Moved to RunPod (2026-11-18).** The owner offered RunPod instead of waiting for the Kaggle reset and
approved this run: five A40 pods (Secure Cloud, $0.59/h each), one fold per pod, all in parallel, using the
fintfm RunPod key this once. `scripts/runpod_lm.py` creates a pod, waits for sshd and the torch install,
streams `src/` and the two competition CSVs in over SSH (the pod is private and deleted afterwards; nothing
is uploaded anywhere else), starts `train_lm.py --folds <k>`, polls `/work/EXIT`, streams `/work/runs` back
and deletes the pod in a `finally`, also on errors, Ctrl-C and a 3.5-hour cap. The pod refuses to train if
torch cannot see the GPU. `train_lm.py --merge-only` then combines the five fold files locally.

Two pods failed and were deleted by the launcher: the first fold-0 pod because the start command kept the
SSH session open (fixed by detaching it with `setsid ... < /dev/null`; SSH timeouts are now retried), and
the first fold-4 pod because its host never finished installing packages in 30 minutes. Both cost about
$0.35 together.

Measured on the A40: tokenizing 1.0M rows takes about 7 minutes, training about 24,000 tokens/s, so one
epoch (17,499 steps of 32 rows) takes 41 minutes and ends before the 90-minute budget. Scoring 140k held-out
rows with 2 orders takes 11-12 minutes, the 300k test rows with 1 order about 13 minutes. A pod costs about
$0.75; the run about $5 including the failed pods.

| fold | GPT-2 val AUC | r12_rmlp_emb, same fold |
|---|---|---|
| 2 | 0.94974 | 0.95453 |
| 1 | 0.94567 | 0.95047 |
| 3 | 0.94362 | 0.95000 |
| 0 | 0.94188 | 0.94892 |
| 4 | 0.93877 | 0.94399 |

So far 0.005-0.007 below our RealMLP on each fold, and below the public GPT-2 members (about 0.9500 OOF);
one epoch and two scoring orders are less than their recipe. Whether it helps is decided in the stacks
(`stack_r12` 0.95138, `stack_pub5` 0.95249), with the usual 4-of-5-folds rule.

**Paused (owner, 2026-11-19, to save money).** Prepared but not launched: a longer GPT-2 run (`--epochs 3
--train-min 150 --oof-orders 4 --test-orders 2`, about 3.1 h and $1.8 per A40 pod, about $9 for five) and a
FinTFM 0.12.0 probe on RunPod (the D8 rows of fold 0, fine-tuned and partitioned contexts against 0.907;
all folds only if it closes much of the gap; about $1-2). The RunPod balance was $16.35, shared with
fintfm's pods. The Kaggle GPU launcher for 2026-11-21 was cancelled: it would have rerun this same job.

**Result.** r14_gpt2 CV 0.94396 (folds above), 0.005-0.007 below RealMLP on every fold. In the stacks it gets
a weight of about 0 and costs a little: `stack_r12` + r14_gpt2 0.95134 against 0.95138 (C=0.1), `stack_pub5`
+ r14_gpt2 0.95243 against 0.95249 (C=0.01). Not adopted. The public stack member `pub_fa_wide_lm_stack`
already carries language-model signal, and ours is weaker than the public GPT-2 members.

A third launcher fault lost fold 1 once: folds 1 and 3 finished in the same second and shared one archive
name, so one copy deleted the other's archive before it was unpacked. Archive and log names are now per
pod; the fold-1 rerun reproduced 0.94567 exactly (same seed). Total RunPod cost about $7.


## D25. Round seventeen: RealMLP seed bags (not adopted)

Seeds 1 and 2 of both all-rows RealMLPs (D23) on four Kaggle CPU kernels (`jobs/r17a1.txt` to
`jobs/r17b2.txt`, new `--seed0` option in `train_gbdt.py`; the seed also moves the 5% validation subset).
Seed 0 is the round-sixteen run. Each bag is the plain mean of three probability vectors.

| Run | CV | Folds | Against seed 0 |
| --- | --- | --- | --- |
| r17_rmlp_emb_all_s1 | 0.94983 | 0.94924 0.95064 0.95503 0.94991 0.94459 | -0.00003 |
| r17_rmlp_emb_all_s2 | 0.94989 | 0.94934 0.95084 0.95484 0.95034 0.94443 | +0.00003 |
| r17_rmlp_emb_all_bag3 | 0.94996 | 0.94940 0.95086 0.95499 0.95026 0.94456 | +0.00010, 4 folds up, 1 level |
| r17_rmlp_emb_te_all_s1 | 0.94969 | 0.94884 0.95072 0.95509 0.95026 0.94405 | +0.00002 |
| r17_rmlp_emb_te_all_s2 | 0.94967 | 0.94887 0.95077 0.95509 0.95014 0.94398 | 0.00000 |
| r17_rmlp_emb_te_all_bag3 | 0.94978 | 0.94897 0.95085 0.95521 0.95031 0.94416 | +0.00011, 4 of 5 folds |

Stacks (nested logistic):

| Stack | CV | Folds | Against |
| --- | --- | --- | --- |
| `stack_r12` with both bags in place of the round-twelve pair, C=0.1 | 0.95139 | 0.95048 0.95186 0.95628 0.95237 0.94627 | 0.95138, 1 of 5 folds up |
| `stack_r12` with the emb_all bag and r16_rmlp_emb_te_all, C=0.1 | 0.95140 | 0.95057 0.95183 0.95627 0.95237 0.94626 | 0.95138, 2 of 5 folds up |
| `stack_r12` plus both bags beside the round-twelve pair, C=0.1 | 0.95133 | 0.95057 0.95188 0.95616 0.95238 0.94603 | 0.95138, lower |
| `stack_pub5` with both bags in place of its RealMLPs, C=0.01 | 0.95243 | 0.95176 0.95266 0.95725 0.95285 0.94790 | 0.95249, 1 of 5 folds up |

Seed bagging lifts each RealMLP by about 0.0001 but the stack does not move: the logistic stacker already
averages 24 members, so the seed noise a bag removes is noise the stack had removed. Not adopted; the
finals stay `stack_r12` and `stack_pub5`. Cost: Kaggle CPU only, about 50 min (emb_all) and 85 min
(emb_te_all) per kernel.

## D26. Round eighteen: three-epoch GPT-2 (not adopted)

The longer run prepared in D24, approved by the owner on 2026-11-19: five A40 pods (one fold each,
`train_lm.py --epochs 3 --train-min 150 --oof-orders 4 --test-orders 2`), the fintfm RunPod key again.
Each fold trained 52,497 steps (1,679,904 rows, three epochs) in 122-126 minutes, scored its held-out rows
with 4 orders in about 25 minutes and the test rows with 2 orders in about 26 minutes; 183-192 minutes per
pod, no failures. Cost: 15.6 pod hours at $0.59/h, about $9.20. No pod was left afterwards. The free Kaggle
job for 2026-11-21 (`jobs/r18a.txt`, about one epoch on a T4) was cancelled before it was pushed: it would
have been a weaker copy of this run, and the GPU hours stay with fintfm.

| fold | r18_gpt2_e3 | r14_gpt2 (one epoch) | r12_rmlp_emb |
|---|---|---|---|
| 0 | 0.94570 | 0.94188 | 0.94892 |
| 1 | 0.94832 | 0.94567 | 0.95047 |
| 2 | 0.95270 | 0.94974 | 0.95453 |
| 3 | 0.94727 | 0.94362 | 0.95000 |
| 4 | 0.94210 | 0.93877 | 0.94399 |

CV 0.94717 against 0.94396: three epochs and more orders add 0.0027-0.0038 on every fold, but the member
stays 0.0018-0.0032 below our RealMLP and below the public GPT-2 members (about 0.9500).

| Stack | CV | Folds | Against |
| --- | --- | --- | --- |
| `stack_r12` + r18_gpt2_e3, C=0.1 | 0.95131 | 0.95043 0.95188 0.95628 0.95212 0.94624 | 0.95138, 2 of 5 folds up |
| `stack_r12` + r18_gpt2_e3, C=1 | 0.95135 | 0.95044 0.95187 0.95625 0.95235 0.94626 | 0.95138, 2 of 5 folds up |
| `stack_pub5` + r18_gpt2_e3, C=0.01 | 0.95243 | 0.95168 0.95262 0.95721 0.95286 0.94800 | 0.95249, 1 of 5 folds up |

Unlike r14 the member gets a real weight in the own stack (0.18), but it takes it from members that already
carry the same signal: the GPT-2-token GLM `r10_glm_tok` falls from 0.56 to 0.29 and `r12_rmlp_emb` from
0.32 to 0.18, and the stack CV drops. Not adopted; the finals stay `stack_r12` and `stack_pub5`. Closing the
gap to the public GPT-2 members would need more epochs or a larger model, at several times this cost.

## D27. Round nineteen: new public packs and the zero-fraud zones

Why: the public top moved to about 0.955 (2026-11-19). Reading its notebooks as text, the gap is not better
models: the top files are blends of public files, each step kept if the public score rose. Their own
checks agree: a 0.95388 public file scores 0.95246 on labelled rows (alexchilton), and the 0.95508 step
loses on all five folds (goodpjw2008). Two of their ideas can be checked on our folds, so we tried those.

**New public members (not adopted).** Imported by `src/import_public.py`, which now also checks a
source's per-fold AUCs against ours when it states them (all eight match within 0.0001):

- [wenimohamed/s6e11-diverse-bpe-realmlp-tabm-oof-pack](https://www.kaggle.com/datasets/wenimohamed/s6e11-diverse-bpe-realmlp-tabm-oof-pack)
  (CC0): seven members on BPE-digit features, 0.94368-0.95093 (LightGBM 0.95093, CatBoost 0.95068 and
  0.95067, GLM 0.95027, RealMLP v2 0.95001, TabM 0.94842, RealMLP v1 0.94368).
- yekenot's [TabM (pytabkit)](https://www.kaggle.com/code/yekenot/ps-s6-e11-tabm-pytabkit) notebook output,
  0.94900. One epoch, so no early stopping on the scored fold.

busyaprime's 52 members were already in (D20).

| Stack, C=0.01 | CV | Folds | Against `stack_pub5` 0.95249 |
| --- | --- | --- | --- |
| + all 8 | 0.95247 | 0.95182 0.95264 0.95724 0.95285 0.94801 | lower |
| + 5 (BPE LightGBM, CatBoost d5, GLM, RealMLP v2, yekenot TabM) | 0.95250 | 0.95181 0.95272 0.95720 0.95294 0.94803 | 3 up, 1 level, 1 down |
| + BPE LightGBM | 0.95249 | 0.95178 0.95264 0.95727 0.95295 0.94800 | 2 up |
| + yekenot TabM | 0.95251 | 0.95184 0.95270 0.95728 0.95290 0.94795 | 3 up |
| + RealMLP v2 and BPE TabM | 0.95248 | 0.95185 0.95265 0.95724 0.95288 0.94806 | 3 up |

`pub_fa_wide_lm_stack` takes a weight of about 1 in `stack_pub5` and already holds about 90 public members,
so new members add little to it. None passes the 4-of-5 rule.

**Zero-fraud zones (adopted).** Public notebooks (daoviet, goodpjw2008, Weni Mohamed) found four regions
with no fraud among the training rows: merchant_risk_score < 13.1 (zone 1); velocity_score <= 4.1 and
merchant_risk_score <= 20 (zone 2); all eight risk flags clear (cvv_retry_count and prior_disputes 0, the
six binary flags 0) with velocity_score <= 4.1 and merchant_risk_score <= 33.8 (zone 3), or velocity_score
<= 6.0 and merchant_risk_score <= 25 (zone 4). Sending these rows to the bottom of the ranking helps if
they hold no fraud on unseen rows too. They were found on all training rows, so `src/zones.py` refits each
merchant-risk edge on the four training folds (just below the lowest fraud, capped at the published edge)
and scores the held-out fold; the velocity edges stay as published, a small optimism this test does not
remove. The stack's nested out-of-fold scores come from the new `stack.py --oof-out`.

| Stack | CV | Folds | Up |
| --- | --- | --- | --- |
| `stack_r12` | 0.95138 | 0.95050 0.95186 0.95620 0.95240 0.94628 | |
| `stack_r12` + zones 1-4 | 0.95143 | 0.95056 0.95191 0.95625 0.95244 0.94637 | 5 of 5 |
| `stack_pub5` | 0.95249 | 0.95180 0.95260 0.95727 0.95294 0.94801 | |
| `stack_pub5` + zones 1-4 | 0.95254 | 0.95185 0.95265 0.95732 0.95297 0.94809 | 5 of 5 |

Zone 1 adds 0.00003 in both stacks and zones 2-4 another 0.00002. 39,149 held-out rows fall in
the refitted zones and none is a fraud; no refitted edge moved below the published one. On the test set
16,730 of 299,984 rows are in a zone. Written as `submissions/stack_r12z.csv` and
`submissions/stack_pub5z.csv` (the zone rows' probabilities multiplied by 1e-6, so they keep their order
among themselves). Adopted by the rule; both replace their unzoned file as final pick candidates.

Public scores (2026-11-19): `stack_r12z` 0.95164 against 0.95160, `stack_pub5z` 0.95288 against 0.95284;
+0.00004 each, in line with CV (+0.00005). They are the two final picks.

## D28. Two nulls: a residual booster and the original data's rules

**Residual booster (`src/residual.py`, not adopted).** LightGBM on the raw features, started from the
stack's nested out-of-fold logit (init_score), early-stopped on a 10% split of the training folds, scored
on the held-out fold. On `stack_r12` it stops after 1-7 rounds: 0.95137 with 4 leaves (1 of 5 folds up)
and 0.95137 with 8 leaves (0 of 5), against 0.95138. The stack is already well calibrated in log-loss; the
zones' gain (D27) is a ranking effect at the low end that a log-loss booster does not see.

**Rules in the original data (none found).** If the original's label followed simple rules, decoding them
would beat any model. It does not look so: LightGBM reaches 0.92782 5-fold AUC on the 20,000 original
rows, fraud rises smoothly with merchant_risk_score (0.4% in the lowest decile, 5.1% in the highest) and
velocity_score (0.1% to 4.8%), and zone 1 (merchant_risk_score < 13.1) holds 3 frauds among 1,112 original
rows. The competition's zero-fraud zones are a property of the synthetic rows, not of a generator rule.

## D29. Round twenty: stacker ideas, new members, and averaging with the wide LM stack

Why: our two picks sit at the honest ceiling that other entrants report too (zohrab's WideLM alone 0.95257,
yashteotia's 278-model pool 0.95261, both on our folds). We brainstormed what could still add signal, tried
the cheap ideas on both final stacks, and read the public discussions and Chris Deotte's past Playground
solutions (GitHub, read as text) for more. Every test is nested on the shared folds and judged by the
4-of-5-folds rule against the stack it would replace: `stack_r12` (own models, C=0.1, 0.95138) and
`stack_pub5` (with public members, C=0.01, 0.95249).

**Averaging with the wide LM stack (adopted).** `pub_fa_wide_lm_stack` alone scores 0.95257, more than
`stack_pub5` (which holds it as a member) without zones. A fit-free average of logit-ranks, then zones
1-4 (`zones.py`, D27):

| File | CV | Folds | Against `stack_pub5z` 0.95254 |
| --- | --- | --- | --- |
| wide LM stack + zones 1-4 | 0.95261 | 0.95182 0.95258 0.95742 0.95273 0.94852 | 2 of 5 up |
| 0.25 pub5 + 0.75 wide, + zones | 0.95263 | 0.95186 0.95265 0.95742 0.95282 0.94845 | 3 up, 1 level |
| **0.5 pub5 + 0.5 wide, + zones (`stack_pubwz`)** | **0.95263** | 0.95188 0.95269 0.95740 0.95290 0.94835 | **4 of 5 up** |
| 0.75 pub5 + 0.25 wide, + zones | 0.95260 | 0.95188 0.95268 0.95737 0.95295 0.94823 | 4 of 5 up |

The zones alone lift the wide stack on all five folds (0.95257 to 0.95261). Weight 0.5 is the untuned
default; 0.75 also passes, so the result does not hinge on the weight. The two inputs' test logit-ranks
correlate at 0.9990. Written as `submissions/stack_pubwz.csv`; it replaces `stack_pub5z` as the CV pick. Public score
(2026-11-20): 0.95293 against 0.95288 for `stack_pub5z`, +0.00005 (CV +0.00009). Marked as final picks on 2026-11-20: `stack_pubwz` and `stack_pub5z`.

**Averaging more stacks (not adopted).** Equal-weight logit-rank averages of every subset of `stack_pub5`,
the wide LM stack, goodpjw2008's wide stack (no LM, 0.95250) and `stack_r12`, each with zones 1-4: none
beats `stack_pubwz` on more than 2 of 5 folds. The best are pub5 + wide LM + wide stack (0.95263, 2 of 5)
and wide LM + wide stack (0.95260, 2 of 5). Any average that includes `stack_r12` loses 0.0001-0.0005.
yashteotia's 278-model wide stack would be a third independently built stack, but its notebook saves only
test files for it, not out-of-fold scores, so it cannot be checked on our folds and is not used.

**Stacker ideas (`src/stack_ideas.py`, none adopted).**

| Idea | `stack_r12` (0.95138) | `stack_pub5` (0.95249) |
| --- | --- | --- |
| Pairwise (RankNet-style) logistic on member-logit differences, 300k pairs, C 0.01 / 0.1 / 1 | 0.95127 / 0.95126 / 0.95125; 2, 2, 1 of 5 up | 0.95010 / 0.94996 / 0.94991; 3, 2, 2 of 5 up |
| Segment stacker: member logit x merchant-risk quartile (C 0.01 / 0.1) | 0.95107 / 0.95106; 0 of 5 | 0.95232 / 0.95240; 1, 0 of 5 |
| Segment stacker: x velocity quartile | 0.95115 / 0.95115; 0 of 5 | 0.95240 / 0.95249; 0, 2 of 5 |
| Segment stacker: x 3 x 3 grid of both | 0.95100 / 0.95091; 0 of 5 | 0.95226 / 0.95232; 1, 0 of 5 |
| Tree-found zero-fraud leaves on top of zones 1-4 (min leaf 500-5,000) | 0.93214-0.95015; 0 of 5 | 0.93286-0.95124; 0 of 5 |
| Members rank-gaussianised over all rows | 0.95101; 1 of 5 | 0.95209; 0 of 5 |
| Members rank-gaussianised within each fold (Deotte, s6e3) | 0.95111; 1 of 5 | 0.95212; 0 of 5 |
| Members + disagreement columns (mean, SD, range), standardised | 0.95118; 0 of 5 | 0.95234; 1 of 5 |
| Members standardised (control for the row above) | 0.95110; 0 of 5 | 0.95224; 1 of 5 |

Leaves with no fraud in the four training folds still hold 43-615 frauds on the held-out folds, so
tree-found zones do not generalise the way the hand zones do. Standardising member logits loses on its own
(the control), so at a fixed C the raw-logit stacker regularises better here.

**Best-matching original row (`src/research_ogmatch.py`, not adopted).** For each row, the original rows
sharing the most exact values over the six high-cardinality columns: ogm_max (1 for 513,594 rows, fraud
1.87%; 2 for 186,070 rows, 1.12%), the count of such rows, and their fraud rate (AUC alone 0.60878).
Added to `stack_r12`'s stacker: 0.95134 against 0.95138. It repeats D14's copied-pair signal.

**New public members (not adopted).** Imported by `src/import_public.py` (fold files checked against ours
where shipped): megayak's GPT-2 target-first LM (CC0, 0.95043; its OOF is log-odds and its test file
rank fractions, so both are ranked before the logit), megayak's 8-seed RealMLP (CC0, 0.95017), and
DavutKursun's XGBoost-on-GLM bag (Apache-2.0, 0.95069). Against `stack_pub5` (C=0.01): + LM 0.95244 (2 of
5 up), + RealMLP 0.95240 (1 of 5), + XGB bag 0.95216 (0 of 5), all three 0.95198. The wide LM stack
already holds these families.

**Error inspection (`stack_r12`, not a model).** Segment AUCs are flat (0.93-0.96) and calibration by
segment is close (No Authentication 6.41% predicted against 6.45% actual). 433,210 rows with every risk
flag clear hold 13% of frauds but 47% of the fraud-side AUC loss (AUC 0.8988 within them); every member
scores 0.888-0.898 there. A LightGBM on raw columns started from the stack's logit loses 0.00044 (0 of 5),
as in D28. Member disagreement does not mark errors. No targeted fix is indicated.

**From the discussions.** Public-board noise, measured by yashteotia by rescoring fixed out-of-fold
predictions on 300 random 60k-row samples: one file moves by about 0.003 (1 SD), the difference of two close
models by about 0.0003. Probing the public board, simulated on out-of-fold rows (yashteotia): +0.00027
on 60k "public" rows, -0.00010 on 240k "private" rows, which supports picking by CV. A logistic
regression fitted on the 20k original rows alone scores 0.934 on train (yashteotia), consistent with the
GLM-first design (D16-D17). Seven more public banks (TabPFN-3.5 pairs, a digit CNN, TabPack and others)
are not imported: yashteotia's pool includes them and gains 0.00004 over the wide LM stack, and their
Kaggle metadata states no licence.

**Encoder-seed bag (adopted in the own-models track).** `train_gbdt.py --enc-seed` (new) changes the seed
of the encoders' inner cross-fitting split, which was always 0. `r10_glm_tok` with seeds 1 and 2 scores
0.95086 and 0.95080 (seed 0: 0.95083); the mean of the three logits, `r20_glm_tok_enc3`, 0.95087. In
`stack_r12` in place of `r10_glm_tok`: 0.95140 against 0.95138, folds 0.95041 0.95194 0.95628 0.95242
0.94630, 4 of 5 up (fold 0 down). Kept beside it instead: 0.95129. The gain is +0.00002 and the own-models
stack is no longer a final pick, so no file was submitted.

**GLM convergence (null).** lbfgs at sklearn's default tolerance (1e-4) stops after 39-48 iterations,
and a public notebook reported a higher GLM with a tighter solver. On fold 0 of `r10_glm_tok` (new
`--params` `solver` and `tol`): lbfgs at 1e-7 takes 235 iterations, newton-cholesky at 1e-7 takes 10
iterations (54 s against 123 s), and both score 0.94957, the same as the default run (47 iterations). Our
GLM was already converged for ranking; newton-cholesky is only faster.

**MinMax tail selection (null).** Owner's idea: where all members agree a row is in the top (bottom) tail of
rank fractions, take their maximum (minimum); other rows keep the stack_pubwz score; zones 1-4 as usual. No fit,
so the OOF score is honest. Against stack_pubwz (0.95263), over thresholds 0.90/0.10 to 0.99/0.01: with
stack_pub5 and the wide LM stack voting, -0.1 to -1.1 points, at most 2 of 5 folds up; adding stack_r12, -0.5
to -11.6, 0 of 5; the 22 stack_pub5 members, -1.5 to -49.8, 0 of 5. The more members vote, the more the most
confident one overrides the stack's weighted order. Script: `src/minmax_tails.py`.

**Board Model + RealMLP re-read (2026-11-20, version 16, public 0.95533).** goodpjw2008's notebook (read
as text) now climbs from 0.95504 to 0.95533 with steps chosen by a model fitted on 167 public scores. By the
notebook's own replay, its last three steps lose OOF AUC (step M -0.00011 against step L; step N -0.00015
against step M, 0 of 5 folds up). It names step I as its CV-supported file, and the LM Ladder `submission_cv.csv`
(OOF 0.95266, D19) as the safer private pick. Every dataset it reads is already imported here (D15-D21, D29),
and its zones are inside our zones 1-4. Nothing new to test on our folds.

**FinTFM readout member (null).** fintfm 1.0.0 `LinearReadoutModel` with entry defaults
(`src/train_readout.py`), run as three Kaggle CPU kernels (folds 0-1, 2-3, 4 and test) and merged:
`r20_fintfm_ro` CV 0.94054, folds [0.93932, 0.94210, 0.94552, 0.94061, 0.93547]. In the own-models stack
(C=0.1) it gives 0.95135 against 0.95138, up on 3 of 5 folds (folds 0 and 4 down), weight -0.08. In the
public-members stack (C=0.01) it gives 0.95245 against 0.95249, up on 1 of 5 folds, weight -0.02. Not
adopted; with stack_pub5 not improved, the stack_pubwz average was not retested.

## D30. Round twenty-one: Storyteller GPT-2 members, a learning curve, adversarial validation, a ranking objective

**Storyteller GPT-2 members (null).** Two target-first GPT-2 runs trained on a Kaggle TPU, found through
DavutKursun's *The 0.9526 Wall* notebook (read as text): mojjeed's *GPT-2 Storyteller on a TPU v5e-8* (5 epochs,
16 field orders per OOF row) and DavutKursun's seed-2027 rerun (8 epochs, 32 orders). Their kernel outputs ship
per-fold log-likelihoods of the two label tokens, shape (rows, orders, 2), plus validation indices.
`src/import_public.py` (new `storyteller_arrays`) scores each row by the mean log-likelihood ratio over orders,
averages the five fold models on test, and checks that the val_idx split equals ours and that every fold's AUC
matches the author's summary: `pub_st_mojjeed` 0.95003, `pub_st_davutkursun` 0.95020. Rank correlation with
the GPT-2 members already imported is 0.976-0.981 (masaya, goodpjw2008 seed 2026, megayak), and 0.97 with the
wide LM stack.

| Added to stack_pub5 (C=0.01, 0.95249) | CV AUC | Folds | Folds up | Weight |
| --- | --- | --- | --- | --- |
| both | 0.95240 | 0.95158 0.95248 0.95737 0.95275 0.94808 | 2 | 0.16, 0.17 |
| DavutKursun's only | 0.95242 | 0.95162 0.95246 0.95737 0.95268 0.94826 | 2 | 0.31 |
| mojjeed's only | 0.95241 | 0.95150 0.95251 0.95747 0.95284 0.94794 | 2 | 0.30 |

In the stack_pubwz average with fixed weights (no fit; zones 1-4): a third equal share loses 11.6 to 12.1 points
(2 of 5 folds up); 0.9 pubwz + 0.1 Storyteller gains 2.4 points but on 3 of 5 folds only; 0.2 loses 0.4. Not
adopted: the stacker gives them weight, but the language-model signal is already in the wide LM stack.

**Learning curve of the best booster (pseudo-labelling bound).** New `train_gbdt.py --train-frac`: each fold
trains on a stratified fraction of its training rows (encoders and GLM start included) and is scored on the full
validation fold. `r11_gb_all_kc` settings, folds 0 and 1:

| Training rows per fold | 140k (25%) | 280k (50%) | 420k (75%) | 560k (all) |
| --- | --- | --- | --- | --- |
| mean fold AUC | 0.94691 | 0.94896 | 0.94993 | 0.95066 |

The curve still rises: the last 140k labelled rows add 0.0007, and doubling from 280k adds 0.0017. That bounds
pseudo-labelling from above: the 300k test rows with their true labels might be worth about +0.001 to one
booster, and pseudo-labels carry no label information beyond the model that made them. Not run (owner's choice
of next steps, 2026-11-21). Inner early stopping is noisy at small fractions (2 to 3 rounds chosen on some folds).

**Four more public notebooks, read as text (nothing to test).** mzishran's *Ranking Credit Card Fraud with ML*
(3 folds, seed 1106, no CV figure; its encodings are our D9-D16 recipe); Weni Mohamed's *RealMLP + TabM + BPE
Diversity Ensemble* (the D27 pack, Nelder-Mead weights per fold, not nested); rishabhbafnaiiitd's *Gram-Schmidt
Optimization* (public-board extrapolation, score = L(leader) + 2 (L(leader) - L(probe)), simulated figures);
DavutKursun's *Deeper Trees Lose* (XGBoost lossguide on our folds: encoded features best at 2 leaves, 0.95001
mean fold AUC; 64 leaves 0.0014 lower on 5 of 5 folds; consistent with D10, D11).

**Adversarial validation (null: no shift).** New `src/adv_val.py`: LightGBM (63 leaves, 400 rounds), 5-fold
CV over train + test rows, predicting "row is from test.csv". Adversarial AUC **0.50013** on the 24 raw columns
and **0.50071** on the 51 `art` columns (raw plus value counts): train and test are drawn alike. Importance
weights p/(1-p) range 0.41 to 2.27 with an effective sample size of 696k of 700k; weighted OOF AUC moves by
0.00003 to 0.00008, the same for every model (stack_pub5 0.95249 -> 0.95246, wide LM stack 0.95257 -> 0.95254,
r11_gb_all_kc 0.95092 -> 0.95089 on raw), so model order is unchanged. CV is a fair stand-in for the private
board, and neither row reweighting nor pseudo-labelling has a shift to correct.

**Pairwise ranking objective (null).** XGBoost `rank:pairwise` (`lambdarank_pair_method` "mean", 1 or 8
pairs per row) in place of logloss on the depth-2 GLM-start booster, with the whole fold as one query group, so
the loss is the AUC's own pairwise form. Run on a Kaggle T4 (kernel `r21rankg`, `jobs/r21_rank_gpu.txt`,
about 1.5 GPU hours for both; the CPU queue was full, owner's OK). Fold AUCs average 0.95066 (1 pair) and 0.95062
(8 pairs), level with the logloss booster (r11_gbx_all_kc 0.95092 pooled). A ranking margin has no common scale
across folds, so the raw pooled OOF falls to 0.94604 and 0.94295; per-fold logit-ranks of the OOF (no labels
used) give the `p1n` and `p8n` runs. In stack_pub5 (C=0.01, 0.95249): raw p1 0.95246 (up on 1 of 5 folds), p1n
0.95235, p8n 0.95230. In stack_r12 (C=0.1, 0.95138): 0.95134 for either. Not adopted, so the pubwz test was not
run. Ranking pairs directly gives the same ordering as logloss here; the boosters' errors are not in how the
loss weighs pairs.

**Ideas considered in round twenty-one.** Offered to the owner with expected value (2026-11-21); the owner chose
adversarial validation and the ranking objective.

| Idea | Status |
| --- | --- |
| Adversarial validation | run: no shift (above) |
| XGBoost pairwise ranking objective on the depth-2 GLM-start booster | run: null (above; Kaggle GPU kernel `r21rankg`) |
| Factorization machine (field-aware embeddings, explicit pairwise interactions) | not chosen; new code and 2-3 kernel hours for a model form the additive reading (FINDINGS 7.1) gives little room |
| Pseudo-labelling with fold-clean pseudo-labels | not chosen; bounded above by the learning curve and with no shift to correct (above) |
| MinMax tail selection (owner's idea) | run: null (D29) |
| XGBoost stumps or one-column-per-tree boosters (DavutKursun) | not run; stumps reach 0.95001, below our boosters |
| RealMLP minus TabM logit difference as a stack column (Gram-Schmidt notebook) | not run; the notebook's evidence is simulated |

## D31. Round twenty-two: public-notebook sweep, a mean of four independent stacks

**Sweep (2026-11-21).** Fifteen notebooks updated on 2026-11-19 to 2026-11-21 were read as text (three reading
agents, nothing executed), and the competition's dataset list was checked for prediction banks we had not
seen. None shows an honest route above about 0.9526 by itself. Points we used or checked:

- dhananjayk2609 (*Part 3: Final Blend*) and DavutKursun (*The 0.9526 Wall*) each build a nested logistic
  stack of 256 and 95 public members on our folds: 0.95260 and 0.95261. Both ship OOF and test scores as
  notebook outputs (Kaggle default Apache-2.0). mathsonchain reports that a mean of four such stacks reaches
  0.95262 against 0.95257 to 0.95261 for each alone.
- dhananjayk2609 says lbfgs stops early on hundreds of collinear members. Our stacker on stack_pub5's 22
  members is converged: lbfgs at tol 1e-10 (202 to 268 iterations) and newton-cholesky (9 iterations) give
  0.95248 against 0.95249 at the default (59 to 75 iterations).
- megayak added a 4-epoch RealMLP to `megayak/s6e11-realmlp-member` (CC0, our split, 2026-11-20): 0.94463
  alone, overfit, but it errs differently. Imported as `pub_mg_realmlp_4ep`.
- Not imported: mayankraaz's predictions (3 folds on an 80% subset, seed 2026), hosen42's and malyshevdanil's
  files (public-board ladder steps, test only), sk20220623 and sometimessubodh (boosters on raw columns, no fold
  file, below our members), sadamtorres (GLM-start boosters and TabPFN; yashteotia's pool already holds them,
  D29). evgendvorkin's group-mean and deviation features (+0.0001 on one XGBoost in a 2-fold screen) and
  georgymamarin's neighbouring-value gaps (+0.00002, 4 of 5 folds, nothing after GPT-2) are not run.
- Red flags noted in the text: zohrab's retrain cell and yekenot's TabM stop on the scored fold; the
  0.9552-0.9554 public files are board-steered ladders (asteriospanoras replays one step at -0.00059 on OOF).

**4-epoch RealMLP (not adopted).** In stack_pub5 (C=0.01): 0.95247, up on 3 of 5 folds. Added to stack_pubwz
at a fixed weight: 0.95266, 0.95267, 0.95266 at 0.03, 0.05, 0.1, each up on 4 of 5 folds; with the weight
chosen on the four other folds each time (0.05 to 0.1): 0.95265, up on 4 of 5. The gain, 0.00002, is the size
of the selection effect estimated in D13, and the mean of four stacks below supersedes it.

**Mean of four nested stacks (adopted, `stack_pub4wz`).** `src/stack_average.py` (new) averages, with equal
weights on logit-ranks and nothing fitted, stack_pub5, goodpjw2008's wide LM stack, DavutKursun's stack
(`stack_all`; its fold column equals ours) and dhananjayk2609's wide stack, then `zones.py` sends zones 1-4
to the bottom. It checks ids, folds and each public stack's per-fold AUCs against the author's log. The main
test was fixed before any result was seen.

| Blend (+ zones 1-4, refitted per fold) | Nested CV | Folds | Folds up vs stack_pubwz |
| --- | --- | --- | --- |
| stack_pubwz (D29) | 0.95263 | 0.95188 0.95269 0.95740 0.95290 0.94835 | |
| **mean of the four stacks (main test)** | **0.95273** | 0.95193 0.95276 0.95755 0.95293 0.94851 | 5 of 5 |
| any 2 of the 4, other than pub5 + wide LM (secondary) | 0.95266 to 0.95276 | | 2 to 5 of 5 |
| any 3 of the 4 (secondary) | 0.95267 to 0.95276 | | 4 or 5 of 5 |

OOF rank correlations between the four stacks are 0.9957 to 0.9984; on test 0.9915 to 0.9990, so the stacks
differ more on test than OOF, and averaging them cuts that variance. The best secondary mixes (0.95276) were
picked after looking and are not used. Written as `submissions/stack_pub4wz.csv` (16,730 test rows in zones);
public 0.95325 (submitted 2026-11-21 on the owner's yes) against 0.95293 for `stack_pubwz`. Two of the four inputs are other people's stacks of public members whose recipes were chosen on
these folds, which the threats-to-validity note in FINDINGS 8 covers.

**No fifth stack qualifies.** Rule fixed before looking at any average: add every stack that is nested on our
folds, publishes OOF and test scores and scores at least 0.9525 alone, and average all of them. Checked
2026-11-21: zohrab's `wide_lm_stack_nested.npy` is a copy of goodpjw2008's wide LM stack (already in);
alexchilton's wide stacks publish test files only; yashteotia's `oof_predictions.csv` holds own models and a
blend of them, below the bar; megayak's notebook publishes members, not its stack. The four-stack mean stands.

**dennis6886's four-family rank correction (null).** *Fraud Signal Rank Fusion* (two notebooks, read as text)
nudges malyshevdanil's board-selected ranking by 0.025 x the mean over four families (GPT, linear, tree,
neural) of `confidence * clip(family median rank - base, +-0.12)`, with `confidence = 1/(1 + 6 * median absolute
deviation)`; constants picked on public-board history, no OOF check, only test files published. Nothing is
fitted, so we applied it with those constants to `stack_pub4wz`'s OOF, families from our imported members by
the same name rule (2 GPT, 41 linear, 69 tree, 19 neural), zones after: 0.95273 at 0.025 (2 of 5 folds up),
0.95272 at 0.05, 0.95270 at 0.1; each family alone 0.95272 to 0.95273, at most 3 of 5.

**More notebooks read as text (nothing to test).** hittmg's four Optuna notebooks (HGB, CatBoost, XGBoost,
LightGBM; sklearn `TargetEncoder`, 100 trials each) early-stop on the validation fold, tune on the same CV and
publish no OOF. goutamsharma12's spline logistic regression (8 knots, cubic, categorical interaction strings)
repeats `r4_glm_spl8` (D12-D13). skull237's and olegpan's teaching baselines on our split: logistic regression
0.9419, depth-2 HGB 0.9466.

