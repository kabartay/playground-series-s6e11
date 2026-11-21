# Findings: Kaggle Playground S6E11, synthetic credit-card fraud

Status 2026-11-21. The competition closes 2026-11-30 23:59 UTC. Rounds twenty to twenty-two (D29 to D31) are complete. This file is a wrap-up for a technical reader who has not seen the repo. It draws every figure from `docs/DESIGN.md`, `docs/COMPETITION.md` and `docs/DATA.md`. The decision ID (Dn) beside a figure points to the section of `docs/DESIGN.md` that holds it.

Conventions used throughout:

- "CV" is 5-fold stratified ROC-AUC on the 699,960 training rows, seed 42, on the folds fixed in `src/common.py` (D1). Stack CVs are nested (section 3).
- "Public" is the Kaggle public leaderboard, which scores about 20% of the 299,984 test rows (about 60,000 rows, about a thousand fraud rows).
- "Folds up" counts the folds, out of five, on which a change scores higher than what it replaces.
- A difference such as "+0.00005" is the difference of two sourced AUCs, computed here.

## 1. Summary

The generator recombines values from a 20,000-row original dataset, so exact values, digits and counts carry fraud signal that a plain tree cannot read. We encode those keys inside each fold, fit a logistic regression (GLM), start 4-leaf boosted trees from its logit, and stack many members with a logistic regression.

| Item | CV (nested, 5-fold) | Public | Source |
| --- | --- | --- | --- |
| Highest single model, `r9_gb_all_kc_s3` (GLM boost, LightGBM, 3 seeds) | 0.95098 | not submitted alone | D18 |
| Own models only: `stack_r12z` (24 models + zero-fraud zones; a final pick until 2026-11-20) | 0.95143 | 0.95164 | D27 |
| Final pick, with public members: `stack_pub5z` | 0.95254 | 0.95288 | D27 |
| Final pick: `stack_pubwz` (fit-free average of `stack_pub5` and the wide LM stack, + zones) | 0.95263 | 0.95293 | D29, COMPETITION.md |
| Highest CV: `stack_pub4wz` (fit-free mean of four nested stacks, + zones; 2026-11-21) | 0.95273 | 0.95325 | D31, COMPETITION.md |

The final selection marked on Kaggle on 2026-11-20 is `stack_pubwz` and `stack_pub5z`, the two best files by CV. The earlier selection (2026-11-19) was `stack_pub5z` and `stack_r12z` (COMPETITION.md).

Main lessons:

1. Members that differ help; more of the same do not (seed bags, wider stacks, section 7).
2. Honest validation costs little: early stopping on the scored fold inflated booster CVs by 0.00004 to 0.00009 (D22).
3. The public files above about 0.954 that we read were tuned on the board, not on labelled rows (D7, D27).
4. Most of what we tried did not help. Section 6 lists it.
5. CV is a fair stand-in for the private board: adversarial validation cannot tell train rows from test rows (AUC 0.500, D30).
6. The best booster is still data-hungry (+0.0007 from its last 140k labelled rows, D30), but without a train/test shift pseudo-labels have nothing new to teach it.

## 2. Task and data

### 2.1 Generator and original dataset

Kaggle Playground S6E11 asks for the probability that a card transaction is fraud (`is_fraud`), scored by ROC-AUC (COMPETITION.md). The competition files were generated from the 20,000-row public dataset [Credit Card Fraud Detection 2026](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026) (CC0-1.0). The column set is identical (DATA.md).

| File | Rows | Fraud rate | Note |
| --- | --- | --- | --- |
| `train.csv` | 699,960 | 1.67% (11,689 positives) | synthetic |
| `test.csv` | 299,984 | not available | synthetic |
| `original.csv` | 20,000 | 1.69% | CC0-1.0 |

There are 24 features: 6 float columns, 7 integer columns, 5 categorical columns and 6 boolean flags. No file has missing values (DATA.md).

### 2.2 Measured data facts

All figures are measured on the training rows unless stated.

| Fact | Figure | Source |
| --- | --- | --- |
| Rows equal to an original row on all 24 columns | none (train or test) | D2 |
| Train values per continuous column that also occur in the original | 85% to 100%: `amount_usd` 97.4%, `account_balance_usd` 84.8%, `merchant_risk_score` and `velocity_score` 100% | D2 |
| Train rows sharing (`amount_usd`, `account_balance_usd`) with an original row | 0.02% | D2 |
| `amount_usd` values absent from the original | 2.6% of values; fraud 2.42% against 1.65% | D14 |
| `velocity_score` and `merchant_risk_score` values absent from the original | 178 and 91 values; fraud 5.6% and 5.5% | D14 |
| `account_balance_usd` values absent from the original | 15% of values; fraud 1.71% against 1.66%, no effect | D14 |
| Rows repeating the (`velocity_score`, `merchant_risk_score`) pair of one original row | 105,697 rows; fraud 0.81% | D14 |
| Rows repeating the (`distance_from_home_km`, `velocity_score`) pair of one original row | 26,196 rows; fraud 0.56% | D14 |
| Feature rows occurring twice in train + test | none | D14 |
| `velocity_score` and `merchant_risk_score` values occurring 20 to 100 times in train + test | fraud 10.3% | D14 |
| Rare `amount_usd` values (at most 20 occurrences) | fraud 3.2% to 3.9% | D14 |
| Rows whose best original match shares exactly 1 of the 6 high-cardinality columns' values | 513,594 rows; fraud 1.87% | D29 |
| Same, 2 of the 6 | 186,070 rows; fraud 1.12% | D29 |
| Base fraud rate in these comparisons | 1.670% | D14 |

Interpretation: the generator reuses the original's values and recombines them into new rows (D2). A row's exact value, its count and whether it was copied from one original row all say something about the label.

Digit and rounding artefacts. Per-group gains from target encoding, as printed in the public notebooks of Georgy Mamarin and Alex Chilton (D7; not run by us in that form):

| Key group | Gain in OOF AUC | Folds up |
| --- | --- | --- |
| Exact values | +0.00089 | 5 of 5 |
| Integer-part keys | +0.00088 | 5 of 5 |
| Cents | +0.00052 | 5 of 5 |

The notebook of evgendvorkin reports signal in decimal endings, flag effects stronger in the synthetic data than in the original, and a truncated tail of `distance_from_home_km` (D7). We did not re-measure the truncated tail. In our own run, adding only the first and second decimal digit to a 31-leaf LightGBM raises CV from 0.94690 to 0.94712 (D9).

Zero-fraud regions. Four regions of the training rows hold no fraud (D27): `merchant_risk_score` below 13.1 (zone 1); `velocity_score` at most 4.1 and `merchant_risk_score` at most 20 (zone 2); and, with all eight risk flags clear (`cvv_retry_count` and `prior_disputes` 0, the six binary flags 0), either `velocity_score` at most 4.1 with `merchant_risk_score` at most 33.8 (zone 3) or `velocity_score` at most 6.0 with `merchant_risk_score` at most 25 (zone 4). Public notebooks by daoviet, goodpjw2008 and Weni Mohamed found them. On the test set 16,730 of 299,984 rows fall in a zone (D27). They are a property of the synthetic rows: the original has 3 frauds among 1,112 rows below the zone-1 edge, and fraud there rises smoothly with the two scores (D28).

## 3. Validation protocol

### 3.1 Folds and target encoding

- Five folds, `StratifiedKFold(shuffle=True, random_state=42)`, over the competition rows only. Every model writes OOF and test predictions on these folds, so OOF predictions can be stacked without leakage (D1). The original rows never enter validation (D2).
- Anything fitted on labels, meaning the target encoders and the GLM behind GLM boost, is fitted on the fold's training rows. The encodings of those training rows are themselves cross-fitted by a 5-way inner split (README, D9). The inner split uses seed 0 in every run so far (D29).
- Smoothing is the m-estimate with weight 20, later at three weights (5, 20 and 100) in parallel (D11, D16).

### 3.2 Stacking and the adoption rule

- Each stacker is fitted on 4 of the 5 folds of the members' OOF predictions and scored on the fifth. The stack CV is therefore nested and honest about the stacker's own fitting (`src/stack.py`).
- A change is adopted when OOF AUC improves and at least four of five folds improve (D1).
- The member sets were compared on the same folds many times, so a selection effect is possible. D13 puts it at the order of 0.00002 for four member sets.

### 3.3 Honest early stopping (D22)

Until round ten, every LightGBM, XGBoost and CatBoost run early-stopped on the fold it was then scored on (patience 200). The fix `--es inner` picks the round count on a stratified 10% holdout of the training fold, refits on the whole training fold for that count divided by 0.9, and uses the scored fold for scoring only.

| Run (`--es inner`) | CV | Same run, old stopping (round eight) | Change | Folds lower |
| --- | --- | --- | --- | --- |
| `r11_gb_all_kc` | 0.95092 | 0.95096 | -0.00004 | 5 of 5 |
| `r11_gbx_all_kc` | 0.95087 | 0.95094 | -0.00007 | 5 of 5 |
| `r11_gb_all` | 0.95078 | 0.95087 | -0.00009 | 5 of 5 |
| `r11_gbc_all_kc` | 0.95082 | 0.95090 | -0.00007 | 4 of 5 |
| `r11_lgb_l4_all_kc` | 0.95043 | 0.95050 | -0.00007 | 5 of 5 |

The stacks barely move: 0.95133 with honest boosters against 0.95132 (own models), and 0.95247 against 0.95249 (own models plus the wide LM stack). The older round-four, five and seven boosters were not rerun, because the measured effect on the stack was nil (D22). Our RealMLPs choose their epoch on a holdout of training rows, never on the scored fold (D23).

### 3.4 Public-board noise and why picks are made by CV

- The public board scores about 60,000 rows with about a thousand fraud rows. Differences of a few thousandths there are within noise (D1, COMPETITION.md). yashteotia rescored fixed out-of-fold predictions on 300 random 60k-row samples: one file moves by about 0.003 (1 SD), and the difference of two close models by about 0.0003 (D29).
- Across our stack submissions, public minus CV runs from -0.00019 (`stack_pub2`, 0.95164 against 0.95183) to +0.00035 (`stack_pub5`, 0.95284 against 0.95249). Section 5.2 lists them. A difference of 0.00003 is noise by that spread (D23).
- Public-board probing was simulated on out-of-fold rows by yashteotia: +0.00027 on 60,000 "public" rows and -0.00010 on 240,000 "private" rows (D29). That supports picking by CV.
- The public top files are blends of public files kept when the public score rose (D7, D27). One file scoring 0.95388 public scores 0.95246 on labelled rows (alexchilton), and goodpjw2008 found a 0.95508 step losing on all five folds (D27).
- Snapshot 2026-11-16: first place 0.95931, then a dense group of about twenty teams at 0.9542 to 0.9544 (COMPETITION.md). By 2026-11-19 the top was about 0.955 (D27).
- Adversarial validation finds no train/test shift: a LightGBM told to separate train rows from test rows scores AUC 0.50013 on raw columns and 0.50071 with value counts. Reweighting training rows by p/(1-p) moves every model's OOF AUC by 0.00003 to 0.00008, all in the same direction, so model order is unchanged (D30). The private rows are drawn like the training rows, which is what lets CV stand in for the private board.

### 3.5 Checking public out-of-fold banks

Several public datasets publish OOF and test predictions on our split. `src/import_public.py` accepts a member only if:

- it has 699,960 OOF and 299,984 test values, all finite, loaded without pickle;
- the source's fold file equals our folds row by row, where one ships; a source without one must state the same split (`StratifiedKFold(5, shuffle=True, random_state=42)`);
- the AUC we compute matches the AUC the source reports within 0.0001, and from D27 its per-fold AUCs match ours when it states them (all eight matched within 0.0001).

Members were also skipped for these reasons (D15, D19, D20):

- two bundle LMs whose bundle test predictions came from the fold-0 model only;
- dariushafshar's members a, h and i, whose manifest says they early-stop on the validation fold;
- alexchilton's first 39, which shipped no fold file (its later dataset states the shared split and was imported in D20).

A source's recipe may still have been chosen on these same folds. This is a limit of the check, discussed in section 8.

## 4. Methods

### 4.1 Feature families

| Family | What it is | Where |
| --- | --- | --- |
| Target-encoded exact values | Every column's exact value as a category, encoded in-fold | D7, D9 |
| Digit keys (`--keys full`) | Integer part, tens, 0.1 resolution, first one and two significant digits, printed length, cents | D9 |
| Token and BPE keys (`--keys tokens`) | GPT-2 token keys and 3-digit chunks of each float column's integer part. Idea after aphoristicartist and busyaprime, written independently | D20 |
| Joint keys (`--joint`) | Keys of non-additive column pairs; `merchant_category` x risk score is the strongest. Idea measured in megayak's notebook (Apache-2.0), written independently | D14, D16 |
| Key counts (`--key-counts`) | The count of every key's value | D16 |
| Artefact flags (`--fe art`) | Log counts, "occurs in the original" flags, 15 copied-pair flags. No labels used | D14 |
| Three smoothing weights | Encodings at m 5, 20 and 100 together | D16 |
| Embedded value buckets (`--fe emb`) | Integer parts and decimals of numeric columns as categories that RealMLP embeds | D23 |

Pooled artefacts and encodings at three weights are complementary: +0.00038 together on the GLM, 5 of 5 folds (D16).

### 4.2 Model families

| Family | Description | Role |
| --- | --- | --- |
| GLM | L2 logistic regression, C=0.01, on logit encodings and one-hot; later with cubic splines (8 knots) on quantile-transformed numeric columns | Base for GLM boost and a stack member (D11, D12) |
| GLM-offset boosting | 4-leaf LightGBM, XGBoost depth 2 (`base_margin`) or CatBoost depth 4 (`baseline`), started from the in-fold GLM's logit | Highest single models (D11, D17) |
| Plain shallow trees | 4-leaf LightGBM, XGBoost depth 2 | Stack members (D10) |
| RealMLP | pytabkit `RealMLP_TD_Classifier`, 8 networks, on `--fe emb` inputs with and without basic-key encodings | Different kind of member (D23) |
| MLP | 64-32 on the spline GLM design | Weak alone, small stack weight (D13) |
| Own GPT-2 language model | Target-first: the label first, 24 fragments in a fresh random order each time. A row is scored with "yes" and with "no", and the log-likelihood difference is the log-odds. After Chris Deotte's S6E10 write-up and Borisov et al. (GReaT), own code | Not adopted (D24, D26) |
| Public members | OOF banks from public datasets, including GPT-2 and compact LMs, TabPFN, RealMLP, TabM, GLMs, boosters | Second track (D15 to D21) |
| FinTFM readout | fintfm 1.0.0 linear readout, a pretrained in-context transformer with a fitted linear head (the owner's own package) | CV 0.94054; not adopted (D29) |

Why a GLM first: the trees already recover numeric shapes, but the GLM adds smooth per-column shapes and the exact-value encodings, and the trees then correct it (D13, D16). A logistic regression fitted on the 20,000 original rows alone scores 0.934 on train (yashteotia, D29), consistent with this design.

### 4.3 Stacking

- A logistic regression over member logits. The stacker choice matters little: on seven round-two members every stacker lands within 0.00002 of the others (D11).

| Stacker (seven round-two members) | Nested CV |
| --- | --- |
| rank average | 0.94935 |
| logistic C=0.01 | 0.94954 |
| logistic C=0.1 | 0.94953 |
| logistic C=1 | 0.94952 |
| hill climbing on standardised logits | 0.94952 |

- The best C varies with member count: C=1 for 14 and 21 own members, C=0.1 for 22 to 24 own members, C=0.01 for the wide-stack combinations, 0.003 to 0.1 for 125 to 164 members (D16, D17, D20, D22). C was chosen on the same folds, so it is lightly tuned.
- Runs saved as logits (GLM, GLM boost) are used as logits. An early bug clipped them to [1e-6, 1-1e-6] before the logit, which flattened them; it is fixed (D11).
- Wide public stacks: goodpjw2008's nested logistic stack of about 200 public members (D21; about 90 per D27) scores 0.95257 on our folds (D21). `stack_pub5` is a logistic stack (C=0.01) of our 21 own models plus that stack. The wide stack takes a weight of about 1 and already holds the members one might add (D27).

### 4.4 Zero-fraud zones with refitted edges

`src/zones.py` tests the four public zones on the nested OOF scores. Because the zones were found on all training rows, each merchant-risk edge is refitted on the four training folds (just below the lowest fraud in the zone's other conditions, capped at the published edge) and applied to the held-out fold. The velocity edges stay as published. Rows in a zone are sent to the bottom of the ranking; in the submission files their probabilities are multiplied by 1e-6, which keeps their order among themselves (D27). For the test set each edge is the lower of the published one and the one refitted on all training rows (`zones.py`).

### 4.5 The fit-free average

`stack_pubwz` is `0.5 stack_pub5 + 0.5 wide LM stack`, averaged on logit ranks, then zones 1 to 4. It has no fitted weights. The two inputs' test logit-ranks correlate at 0.9990 (D29).

`stack_pub4wz` extends it to four nested stacks built independently on our folds: `stack_pub5`, goodpjw2008's wide LM stack, DavutKursun's 95-member stack and dhananjayk2609's 256-member stack (both Kaggle notebook outputs). Equal weights on logit ranks, then zones 1 to 4 (`src/stack_average.py`). The four agree at rank correlation 0.9957 to 0.9984 out of fold but 0.9915 to 0.9990 on test, so averaging cuts test-side variance that a single stack carries (D31).

## 5. Results

### 5.1 The ladder

Each row's change is against the row above it in the same block. Blocks are separate tracks.

Single models (CV):

| Step | Run | CV | Change | Decision |
| --- | --- | --- | --- | --- |
| Baseline, 31-leaf LightGBM, raw columns | `lgb_raw_o0` | 0.94690 | | D6 |
| + first and second decimal digits | `lgb_digits` | 0.94712 | +0.00022 | D9 |
| + target encoding of exact values, rounded keys, cents | `lgb_digits_te1` | 0.94854 | +0.00142, 5 of 5 folds | D9 |
| 4-leaf trees and pair keys (XGBoost depth 2 is 0.94926) | `lgb_digits_te2_l4` | 0.94923 | +0.00069 | D10 |
| GLM boost, m-estimate 20, full digit keys, 76 keys | `r3_glmboost` | 0.95056 | +0.00130 against XGBoost 0.94926, 5 of 5 folds | D11 |
| 3 seeds averaged | `r4_gb_s3` | 0.95062 | +0.00006, 4 of 5 | D12 |
| Artefacts + encodings at m 5, 20, 100 | `r7_gb_art_m3` | 0.95076 | +0.00014 | D16 |
| + joint keys in the GLM, key counts for the trees | `r7_gb_joint_kc` | 0.95078 | +0.00002 | D16 |
| All of it, in the GLM boost | `r8_gb_all_kc` | 0.95096 | +0.00018, 5 of 5 | D17 |
| 3 seeds (highest single model; stack effect nil) | `r9_gb_all_kc_s3` | 0.95098 | +0.00002, 3 of 5 | D18 |

Own-model stacks (nested CV):

| Round | Members | CV | Change | Decision |
| --- | --- | --- | --- | --- |
| One | 8 | 0.94728 | | D6 |
| Two | 7 | 0.94952 | +0.00224 | D10 |
| Three | 6 | 0.95063 | +0.00111 | D11 |
| Four | 5 | 0.95066 | +0.00003 | D12 |
| Four + CatBoost and XGBoost GLM boost | 7 | 0.95071 | +0.00005 | D12 |
| Five | 9 | 0.95079 | +0.00008 | D13 |
| Seven | 14 | 0.95107 | +0.00028, 5 of 5 | D16 |
| Eight | 21 | 0.95114 | +0.00007 | D17 |
| Ten (+ `r10_glm_tok`) | 22 | 0.95132 | +0.00018, 4 of 5 | D20 |
| Eleven (honest boosters) | 22 | 0.95133 | +0.00001 | D22 |
| Twelve (+ two RealMLPs) | 24 | 0.95138 | +0.00005, 4 of 5 | D23 |
| Twelve + zones 1-4 (`stack_r12z`) | 24 | 0.95143 | +0.00005, 5 of 5 | D27 |
| Twelve with the encoder-seed bag in place of `r10_glm_tok` (no zones) | 24 | 0.95140 | +0.00002 over twelve, 4 of 5 | D29 |

Stacks with public members (nested CV):

| Stack | Members | CV | Change | Decision |
| --- | --- | --- | --- | --- |
| Own 9 + 38 public (`stack_pub1`) | 47 | 0.95175 | +0.00096 over own 9 (0.95079) | D15 |
| Own 21 + 38 public (`stack_pub2`) | 59 | 0.95183 | +0.00008, 4 of 5 | D17 |
| + 14 public language-model and other members (`stack_pub3`) | 73 | 0.95204 | +0.00021, 5 of 5 | D19 |
| + busyaprime, dariushafshar, alexchilton (`stack_pub4`) | 173 | 0.95225 | +0.00021 | D20 |
| Own 21 + the wide LM stack (`stack_pub5`) | 22 | 0.95249 | +0.00024 | D21 |
| `stack_pub5` + zones 1-4 (`stack_pub5z`) | 22 | 0.95254 | +0.00005, 5 of 5 | D27 |
| 0.5 `stack_pub5` + 0.5 wide LM stack + zones (`stack_pubwz`) | fit-free | 0.95263 | +0.00009, 4 of 5 | D29 |
| Mean of four nested stacks + zones (`stack_pub4wz`) | fit-free | 0.95273 | +0.00010, 5 of 5 | D31 |

### 5.2 Best runs and submissions

| File | Date | CV (nested, 5-fold) | Public | Note |
| --- | --- | --- | --- | --- |
| `stack_pub4wz.csv` | 2026-11-21 | 0.95273 | 0.95325 | D31 |
| `stack_pubwz.csv` | 2026-11-20 | 0.95263 | 0.95293 | D29; final pick |
| `stack_pub5z.csv` | 2026-11-19 | 0.95254 | 0.95288 | D27; final pick |
| `stack_pub5.csv` | 2026-11-17 | 0.95249 | 0.95284 | D21 |
| `stack_pub4.csv` | 2026-11-17 | 0.95225 | 0.95219 | D20 |
| `stack_pub3.csv` | 2026-11-17 | 0.95204 | 0.95190 | D19; rank 596 of 2,098 at the time |
| `stack_pub2.csv` | 2026-11-17 | 0.95183 | 0.95164 | D17 |
| `stack_pub1.csv` | 2026-11-17 | 0.95175 | 0.95165 | D15 |
| `stack_r12z.csv` | 2026-11-19 | 0.95143 | 0.95164 | D27; final pick until 2026-11-20, highest own-model public score |
| `stack_r16.csv` | 2026-11-19 | 0.95136 | 0.95163 | D23; public check only |
| `stack_r12.csv` | 2026-11-18 | 0.95138 | 0.95160 | D23 |
| `stack_r10.csv` | 2026-11-17 | 0.95132 | 0.95134 | D20 |
| `stack_r8.csv` | 2026-11-17 | 0.95114 | 0.95123 | D17 |
| `stack_r5.csv` | 2026-11-16 | 0.95079 | 0.95113 | D13; rank 766 of 2,024 at the time |
| `stack_r2.csv` | 2026-11-16 | 0.94952 | 0.94982 | D10 |
| honest-CV notebook, version 1 | 2026-11-20 | 0.95135 (local check of the six runs) | 0.95155 | public notebook, six own models; not a final pick |
| FinTFM starter, version 3 | 2026-11-20 | 0.9296 (single 20,000-row holdout) | 0.94105 | linear readout fitted on the whole pool |
| `submission_lightgbm.csv` | 2026-11-03 | 0.9427 (single 20% holdout) | 0.94578 | LightGBM, 500 rounds, defaults otherwise |

Final leaderboard ranks of the picks: not measured, the competition has not closed.

## 6. Negative results

Every idea below was tried and not adopted. "Folds up" is against what it would replace. Where a result carries a gain that did not pass the 4-of-5 rule, the table says so.

### 6.1 The original dataset

| Idea | Result | Decision |
| --- | --- | --- |
| Append the 20,000 original rows to each training fold | LightGBM 0.94680 with, 0.94690 without (-0.0001) | D6 |
| Per-value fraud rates and counts from the original as features | costs 0.0002 to 0.0013 (runs 0.94633 to 0.94676) | D6 |
| LightGBM trained on the original alone as a feature | 0.94564, -0.00126 | D6 |
| Best-matching original row (`ogm_max`, count, fraud rate; alone AUC 0.60878) added to the stacker | 0.95134 against 0.95138 | D29 |
| Look for simple rules in the original's labels | none found: 0.92782 5-fold AUC, smooth rise of fraud in the two risk scores | D28 |
| The original as a teacher probability (third-party measure) | -0.0036715 | D7 |

### 6.2 Features and encodings

| Idea | Result | Decision |
| --- | --- | --- |
| 45 top-10 pair encodings, LightGBM | 0.94845 against 0.94854 | D9 |
| 45 pair keys, GLM | 0.95024; below joint keys and key counts (0.95028 each) | D16 |
| Frequency encoding (third-party measure) | +0.0001 | D7 |
| Hierarchical encoding (m 20) on the token-key GLM | 0.95080 against 0.95083; fold changes -10, -1, -1, +3, -6 (x 1e-5) | D22 |
| Token keys in the three booster families | 0.95110, 0.95104, 0.95097 against 0.95096, 0.95094, 0.95090; 3, 2 and 3 of 5 folds up | D20 |
| Token keys in a 3-leaf LightGBM | 0.95109; 3 of 5 | D20 |
| Artefact features alone in the GLM | 0.95028, 3 of 5 folds up | D16 |
| Artefact features alone in the GLM boost | 0.95066, 2 of 5 | D16 |
| Key counts in the GLM | 0.95028, 3 of 5; no gain over joint keys | D16, D17 |
| 10-way inner cross-fit of the encoder | 0.94985 against 0.94987 | D12 |
| Removing `is_ai_generated_scam_attempt` (third-party measure) | costs 0.0013 | D7 |

### 6.3 Model settings

| Idea | Result | Decision |
| --- | --- | --- |
| GLM with C=0.003 | 0.94975 (-0.00012) | D12 |
| GLM solved to tolerance 1e-7 (lbfgs, 235 iterations; newton-cholesky, 10 iterations) | fold 0 0.94957, same as the default run (47 iterations) | D29 |
| GLM with C=0.03 | 0.94987 (0.00000) | D12 |
| GLM boost, learning rate 0.015, feature fraction 0.3 | 0.95055 (-0.00001) | D12 |
| GLM boost, 8 leaves, `min_child_samples` 200 | 0.95050 (-0.00006) | D12 |
| GLM splines with 16 knots | 0.95006 against 0.95015 | D13 |
| GLM boost started from the spline GLM | 0.95060 against 0.95062; kept as a stack member only | D13 |
| CatBoost GLM boost at depth 6 | 0.95048 against 0.95057 at depth 4 | D14 |
| CatBoost GLM boost with 3 seeds | 0.95058; stack 0.95078 against 0.95079 | D14 |
| MLP 32 with alpha 0.01 | 0.94578 | D13 |
| Wider, slower MLP (128-64, learning rate 3e-4) | 0.94715, same as round five | D14 |
| MLP on the full round-seven design | 0.94666 against 0.94715 (-0.00049) | D17 |
| Seed bags of the five round-eight trees | single models +0.00000 to +0.00002; stack 0.95112 against 0.95114 | D18 |
| Seed bags of the RealMLPs (3 seeds) | +0.00010 and +0.00011 alone; stack 0.95139, 1 of 5 folds up | D25 |

### 6.4 Neural and language-model members

| Idea | Result | Decision |
| --- | --- | --- |
| RealMLP on artefacts + digit-key encodings (127 columns) | 0.94938; stack 0.95125 | D23 |
| RealMLP on raw + basic-key encodings | 0.94889; stack 0.95132 | D23 |
| RealMLP, 16 networks instead of 8 | 0.94936; stack 0.95131 | D23 |
| RealMLPs on all training rows | 0.94986 and 0.94967; stack 0.95137 (C=1) against 0.95138 | D23 |
| RealMLP on all rows, 5 epochs | 0.94854; overfits (-0.00095) | D23 |
| TabM (pytabkit) on a Kaggle CPU kernel | fold 0 only: 0.93680 against 0.94892; fold 0 took 6.7 hours; stopped at the 12-hour kernel limit during fold 1 | D23 |
| Own GPT-2, one epoch | CV 0.94396; stack weight about 0; `stack_r12` 0.95134, `stack_pub5` 0.95243 | D24 |
| Own GPT-2, three epochs | CV 0.94717; stack weight 0.18 but `stack_r12` 0.95131 (C=0.1) and 0.95135 (C=1), `stack_pub5` 0.95243 | D26 |
| GPT-2-small member in a public blend (third-party measure) | -0.0029 on the blend | D7 |
| FinTFM zero-shot, 2,000-row context, 20,000 stratified rows of fold 0 | uniform 0.89958, balanced 0.90679, hybrid 0.89648, retrieval 0.78609; blend weight tested on one half and scored on the other: -0.00043 and +0.00014 (LightGBM), -0.00047 and 0.00000 (XGBoost) | D8 |

Both own GPT-2 runs stayed below our RealMLP on every fold: by 0.005 to 0.007 (one epoch, D24) and 0.0018 to 0.0032 (three epochs, D26). A longer or larger run would cost several times the $9.20 of the second (D26).

### 6.5 Stackers

| Idea | `stack_r12` (0.95138) | `stack_pub5` (0.95249) | Decision |
| --- | --- | --- | --- |
| Pairwise (RankNet-style) logistic on member-logit differences, 300,000 pairs, C 0.01 / 0.1 / 1 | 0.95127 / 0.95126 / 0.95125; 2, 2, 1 of 5 up | 0.95010 / 0.94996 / 0.94991; 3, 2, 2 of 5 up | D29 |
| Segment stacker: member logit x merchant-risk quartile (C 0.01 / 0.1) | 0.95107 / 0.95106; 0 of 5 | 0.95232 / 0.95240; 1, 0 of 5 | D29 |
| Segment stacker: x velocity quartile | 0.95115 / 0.95115; 0 of 5 | 0.95240 / 0.95249; 0, 2 of 5 | D29 |
| Segment stacker: x 3 x 3 grid | 0.95100 / 0.95091; 0 of 5 | 0.95226 / 0.95232; 1, 0 of 5 | D29 |
| Tree-found zero-fraud leaves on top of zones 1-4 (min leaf 500 to 5,000) | 0.93214 to 0.95015; 0 of 5 | 0.93286 to 0.95124; 0 of 5 | D29 |
| Members rank-gaussianised over all rows | 0.95101; 1 of 5 | 0.95209; 0 of 5 | D29 |
| Members rank-gaussianised within each fold (after Deotte, S6E3) | 0.95111; 1 of 5 | 0.95212; 0 of 5 | D29 |
| Members + disagreement columns (mean, SD, range), standardised | 0.95118; 0 of 5 | 0.95234; 1 of 5 | D29 |
| Members standardised (control for the row above) | 0.95110; 0 of 5 | 0.95224; 1 of 5 | D29 |
| Residual LightGBM started from the stack's nested logit (4 or 8 leaves) | 0.95137 and 0.95137; 1 and 0 of 5; stops after 1 to 7 rounds | not run | D28 |
| Hill climbing or rank average in place of the logistic stacker | within 0.00002 of the logistic stacker on seven round-two members (rank average 0.00019 lower) | | D11 |

MinMax tail selection was checked against `stack_pubwz` (0.95263) instead: where every member's rank fraction is above `hi` (below `lo`), the row takes the members' maximum (minimum), and every other row keeps the pubwz score. Nothing is fitted, so the OOF figure is honest.

| Members voting | 0.90/0.10 | 0.95/0.05 | 0.98/0.02 | 0.99/0.01 |
| --- | --- | --- | --- | --- |
| stack_pub5 and wide LM stack | -1.1 pts; 2 of 5 up | -0.3; 1 of 5 | -0.1; 2 of 5 | -0.0; 2 of 5 |
| + stack_r12 | -11.6; 0 of 5 | -5.4; 0 of 5 | -1.9; 0 of 5 | -0.5; 0 of 5 |
| the 22 stack_pub5 members (also with the wide LM stack) | -49.8; 0 of 5 | -23.5; 0 of 5 | -7.1; 0 of 5 | -1.5; 0 of 5 |

(1 pt = 0.00001 AUC.) The more members vote, the larger the loss: the maximum hands the agreed top tail to whichever member is most confident, and the stack's weighted order is the better one there (D29).

Leaves with no fraud in the four training folds still hold 43 to 615 frauds on the held-out folds, so tree-found zones do not generalise as the hand zones do (D29). Standardising member logits loses on its own (the control), so at a fixed C the raw-logit stacker regularises better here (D29).

### 6.6 Public members and wide stacks

| Idea | Result | Decision |
| --- | --- | --- |
| `stack_pub5` + all 8 new public members (BPE pack, yekenot TabM) | 0.95247 | D27 |
| `stack_pub5` + 5 of them | 0.95250; 3 up, 1 level, 1 down | D27 |
| `stack_pub5` + BPE LightGBM | 0.95249; 2 up | D27 |
| `stack_pub5` + yekenot TabM | 0.95251; 3 up | D27 |
| `stack_pub5` + RealMLP v2 and BPE TabM | 0.95248; 3 up | D27 |
| `stack_pub5` + megayak LM / megayak RealMLP / DavutKursun XGBoost bag / all three | 0.95244 (2 of 5) / 0.95240 (1 of 5) / 0.95216 (0 of 5) / 0.95198 | D29 |
| Own 21 + 156 public members against own 21 + the wide stack | 0.95233 against 0.95249 | D21 |
| Refitting weights over the wide LM stack | costs more than the members add | D21 |
| Fit-free step from the wide LM stack towards our 21-model mean, best single, best GLM, RealMLP v15, wide stack without LMs | -2.5e-5 (2 of 5), -0.6e-5 (3/5), -1.1e-5 (2/5), +0.5e-5 (4/5), -1.0e-5 (0/5) | D21 |
| Our round-ten members added to the wide stack | 0.95241, below the wide stack alone | D20 |
| Seven further public banks | not imported: their pool gains 0.00004 over the wide LM stack (yashteotia) and their Kaggle metadata states no licence | D29 |
| The 173-member stack without busyaprime's 27 boosters | 0.95219 against 0.95225 | D21 |
| Equal-weight averages of every subset of `stack_pub5`, the wide LM stack, the wide stack without LMs and `stack_r12`, + zones | none beats `stack_pubwz` on more than 2 of 5 folds; best 0.95263 (2 of 5); any subset with `stack_r12` loses 0.0001 to 0.0005 | D29 |
| yashteotia's 278-model wide stack as a third stack to average | not used: only its test file is published, not its out-of-fold scores | D29 |
| Two target-first GPT-2 "Storyteller" members trained on a TPU (mojjeed; DavutKursun seed 2027, 8 epochs), 0.95003 and 0.95020 alone, rank correlation 0.98 with the GPT-2 members already in | `stack_pub5` + either or both: 0.95240 to 0.95242, 2 of 5 folds up, weights 0.16 to 0.31; in the `stack_pubwz` average at fixed shares: +2.4 points at 0.1 (3 of 5), -0.4 at 0.2, -11.6 at a third | D30 |

| megayak's 4-epoch RealMLP (0.94463 alone, overfit) | `stack_pub5`: 0.95247, 3 of 5 folds up; in `stack_pubwz` with the weight chosen inside the folds (0.05 to 0.1): 0.95265, 4 of 5, within selection noise | D31 |
| Tighter convergence of the logistic stacker (lbfgs tol 1e-10, newton-cholesky) | 0.95248 against 0.95249: the default is converged on 22 members | D31 |

### 6.7 Error inspection (not a model)

Segment AUCs of `stack_r12` are flat (0.93 to 0.96) and calibration by segment is close (No Authentication 6.41% predicted against 6.45% actual). The 433,210 rows with every risk flag clear hold 13% of the frauds but 47% of the fraud-side AUC loss (AUC 0.8988 within them); every member scores 0.888 to 0.898 there. Member disagreement does not mark errors. No targeted fix is indicated (D29).

### 6.8 FinTFM readout member (D29)

| Run | CV AUC | Against | Folds up | Verdict |
| --- | --- | --- | --- | --- |
| `r20_fintfm_ro` alone (fintfm 1.0.0 linear readout, entry defaults, three Kaggle CPU kernels) | 0.94054 | | | |
| stack_r12 + readout (C=0.1) | 0.95135 | 0.95138 | 3 of 5 | not adopted |
| stack_pub5 + readout (C=0.01) | 0.95245 | 0.95249 | 1 of 5 | not adopted |

The readout is 0.010 below our best single models and takes a small negative stack weight in both tracks (-0.08 and -0.02), so the stack finds nothing in it to use. With stack_pub5 not improved, the stack_pubwz average was not retested.

## 7. Discussion

### 7.1 The signal is largely additive

Evidence in the logs, and our reading of it:

- Very shallow trees beat deep ones: 4 leaves beat 31 by +0.0006 on raw columns and +0.0008 on the encoded set, all five folds (D10). Two or three leaves are no better on the third-party recipe (D7), and 8 leaves lose 0.00006 (D12).
- Each new feature idea helps the GLM more than the trees, because the GLM cannot build interactions or count effects on its own (D16).
- A residual booster started from the stack's logit stops after 1 to 7 rounds and gains nothing (D28).
- A LightGBM fitted to the 20,000 original rows reaches 0.92782, and fraud rises smoothly in the two risk scores (D28).

Our reading is that the label depends on a sum of per-column effects, with the exact value, digits and counts as extra per-column shapes, plus a few interactions (joint keys, copied pairs). We have not tested this against a purpose-built additive model, so it remains a reading and not a measured claim.

### 7.2 Heterogeneity beats count

- Seed bags did nothing in the stack (D18, D25): a stack of 24 members already averages the seed noise.
- Members built on the same encodings agree too closely: six round-three models add +0.00007 over GLM boost alone (D11). CatBoost and XGBoost GLM boost add +0.00005 to the stack although each loses to LightGBM alone (D12).
- A GPT-2 target-first language model takes weight 0.55 of 47 in `stack_pub1` because it errs differently from every tree and GLM (D15).
- Our own GPT-2 was weaker, and its stack weight came from members carrying the same signal: `r10_glm_tok` fell from 0.56 to 0.29 and `r12_rmlp_emb` from 0.32 to 0.18 (D26).
- Public RealMLPs help the public score more than CV shows (D23).

### 7.3 Wide stacks saturate

The public-member stack levelled off near 0.9522 at 173 members (D20) and reaches 0.95249 with the wide stack (D21). From there, nothing we added passed the 4-of-5 rule apart from the zones and the fit-free average. The wide LM stack alone (0.95257) beats `stack_pub5` before zones (0.95249), and a 278-model pool (yashteotia, on our folds) scores 0.95261 (D29). New members of a family already in the pool do not move it even when strong alone: two TPU-trained GPT-2 members at 0.95003 and 0.95020, rank correlation 0.98 with the GPT-2 members already in, lower `stack_pub5` to 0.95240-0.95242 (D30). Combination rules do not move it either: MinMax tail selection, where the most confident member takes the agreed tails, loses at every threshold (D29).

### 7.4 An honest ceiling near 0.9526, and a board wall

On our folds the CV-backed results meet at about 0.9526: the wide LM stack 0.95257, the 278-model pool 0.95261, and `stack_pubwz` 0.95263 (D29). Averaging four of these stacks built independently reaches 0.95273 (D31), so the wall is partly the variance of any one stack. DavutKursun's 95-member stack (0.95257) and goodpjw2008's 201-member wide stack land there as well; his notebook is titled *The 0.9526 Wall* (D30). The public board sits higher (first place 0.95931 on 2026-11-16, about 0.955 for the top group on 2026-11-19), and the sources we read agree that the top files are public files steered by the board (D7, D21, D27). Our evidence on steering is limited to what others measured: a 0.95388 public file scoring 0.95246 on labelled rows, and a 0.95508 step losing on all five folds (D27). We did not probe the board ourselves. Where the private rows end up is not measured.

### 7.5 What transfers to real fraud work, and what does not

This is judgement. We have not run any of it on real transaction data.

| Likely to transfer | Generator-specific |
| --- | --- |
| In-fold target encoding, cross-fitted inside the training fold | Reuse of the original's exact values (D2) |
| Nested stack CV and adopting a change only if most folds agree | Digit, cents and token keys as signal (D7, D20) |
| Early stopping chosen away from the scored fold (bias 0.00004 to 0.00009, D22) | Zero-fraud zones from synthetic regions (D27, D28) |
| Shallow trees started from a regularised linear model | Copied pairs, value counts, "absent from the original" flags (D14) |
| Picking by CV, not by the public board (D29 simulation) | A public leaderboard that rewards blending public files |
| Checking imported predictions against folds and reported AUCs (section 3.5) | Stack gains from public members whose authors tuned on the same folds (D15) |

## 8. Threats to validity

- **Zone edges chosen on all rows.** The four zones were found on all training rows. The merchant-risk edges are refitted per fold, but the velocity edges stay as published (D27), so the zone gain (+0.00005 in both stacks, 5 of 5 folds) carries a small optimism that the test does not remove. 39,149 held-out rows fall in the refitted zones and none is a fraud (D27).
- **Folds reused across many experiments.** Every model and stack in the results table is scored on the same five folds, and member sets, C and the fit-free weight were chosen on them. D13 estimates a selection effect of the order of 0.00002. The weight 0.5 in `stack_pubwz` is the untuned default; 0.75 also passes, so the result does not hinge on it (D29).
- **Public members' unknown training details.** Their authors chose recipes and blends on these folds and on the public board, so part of the public-member gain may be selection that the private rows will not repeat (D15, D19). The wide stack's members' training details are not all known to us. `stack_pub4` carries at most 0.00006 of optimism from busyaprime's early-stopped boosters (D21). We checked folds and reported AUCs, not how each member was trained.
- **Boosters in the earlier rounds.** Round four to eight boosters were early-stopped on the scored fold. The stack effect of the fix was nil (D22), but single-model CVs from those rounds are inflated by 0.00004 to 0.00009.
- **Private-board shift.** The private leaderboard scores the other 80% of the test rows. Adversarial validation finds no shift between train and test (AUC 0.500, D30). Our public scores differ from CV by -0.00019 to +0.00035 (section 3.4). Whether the private score tracks CV is not measured until the competition closes.
- **One generator, one metric.** All results are for one synthetic dataset and ROC-AUC. Calibration was checked by segment in log-loss terms only (D28, D29).

## 9. Reproducibility

### 9.1 Code and commands

All scripts run from `src/` with `.venv/bin/python` (Python 3.12, `uv pip install -r requirements.txt`). Data is not committed; DATA.md gives the download commands. Pinned versions include LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10, scikit-learn 1.9.1 and tiktoken 0.14.0.

| Step | Command or script |
| --- | --- |
| Train a CV model | `train_gbdt.py --model {lgb,xgb,cat,glm,mlp,realmlp} --fe ... --te ... --keys ... --offset glm --name N` |
| Honest early stopping | `train_gbdt.py ... --es inner` (D22) |
| Encoder seed (new) | `train_gbdt.py ... --enc-seed K` (D29) |
| Learning curve | `train_gbdt.py ... --train-frac F --folds 2` (scores the full validation fold, saves nothing; D30) |
| Ranking objective | `train_gbdt.py --model xgb ... --params '{"objective":"rank:pairwise",...}'` (unit weights only; D30) |
| Adversarial validation | `adv_val.py --fe {raw,art} --oof F [F ...]` (D30) |
| MinMax tail selection | `minmax_tails.py --base F [F ...] --voters F [F ...]` (D29) |
| Own GPT-2 | `train_lm.py --name N --epochs 3 --train-min 150 --oof-orders 4 --test-orders 2`, launched by `scripts/runpod_lm.py` (D26) |
| FinTFM probes | `train_fintfm.py`, `probe_readout.py`, `train_readout.py` (D8, D29) |
| Import public banks | `import_public.py` (including the Storyteller kernel outputs, D30) |
| Stack | `stack.py RUN [RUN ...] [--C c] [--oof-out F] [--out F]` |
| Zones | `zones.py --oof X.npy --sub X.csv` |
| Stacker ideas, residual, original match | `stack_ideas.py`, `residual.py`, `research_ogmatch.py` |
| Data measurements | `research_artefacts.py` |
| Job lists | `jobs/*.txt`, one `train_*.py` call per line, run with `scripts/queue_kernel.sh` (Kaggle) |

Every run writes `runs/<name>/{oof.npy,test.npy,meta.json}` on the shared folds. The fold seed (42) is never changed. `runs/` is not committed.

### 9.2 Compute and runtimes

| Resource | Used for | Recorded runtimes |
| --- | --- | --- |
| Local Mac, CPU, 4 threads (shared with other jobs) | stacking, zones, stacker ideas, residual and original-match tests, data measurements | not recorded |
| Kaggle CPU kernels (private, up to five slots) | all CV model runs, FinTFM probes and readout kernels | `r3_glm` 148 s; `r3_lgb_l4` 828 s; `r3_glmboost` 550 s (D11); CatBoost depth 6, about 75 min each (D11); RealMLP about 20 min a fold (D23); RealMLP bags about 50 min and 85 min a kernel (D25); TabM fold 0 6.7 h (D23); FinTFM about 20 rows/s on 2026-11-03 and about 70 rows/s in the later probe (D5, D8); readout about 100 to 140 rows/s (`train_readout.py`) |
| RunPod A40 pods (own GPT-2 only, $0.59/h each) | `r14_gpt2` (one epoch, 5 pods) and `r18_gpt2_e3` (three epochs, 5 pods) | one epoch: 41 min training, 11 to 12 min to score 140k held-out rows with 2 orders, about 13 min for the 300k test rows with 1 order, about $5 including failed pods, about $7 in total with the fold-1 rerun (D24). Three epochs: 52,497 steps in 122 to 126 min a fold, 183 to 192 min a pod, 15.6 pod hours, about $9.20 (D26) |

The Kaggle GPU quota was not used for the GPT-2 runs (D24, D26). The fold-1 rerun of the one-epoch GPT-2 reproduced 0.94567 exactly with the same seed (D24).

## 10. Credits and licences

Competition data: subject to the competition rules, not included in the repo. Original dataset: CC0-1.0. Code of this repo: MIT Licence (README). Public sources below were used as data, as text read for ideas, or both. Our code is written independently and copies none of theirs. The methods here are published prior art or recombinations of public notebooks' ideas; what is ours is the protocol, the measurements and the code.

| Source (Kaggle name) | Used for | Licence as recorded |
| --- | --- | --- |
| uditjain13, Credit Card Fraud Detection 2026 | original dataset | CC0-1.0 |
| goodpjw2008 (OOF bank, compact LM seed 7, GPT-2 seed 2026, nested wide LM stack, board model) | OOF members, wide stack, zones | CC0 |
| amanatar (rank-1 q17 meta bundle; 16 members collected from other notebooks, named after their authors in it) | 12 members imported | CC0 |
| busyaprime (33 own members, 52 members) | OOF members | CC0 |
| dariushafshar (golem OOF library, GPU members) | OOF members (a, h, i left out) | CC0 |
| alexchilton (OOF predictions) | OOF members; notebook read for recipe | CC0 |
| wenimohamed (diverse BPE, RealMLP, TabM OOF pack) | tested, not adopted | CC0 |
| megayak (GPT-2 target-first LM, 8-seed RealMLP, compact LM) | members tested; notebook read for joint-key and key-count ideas | CC0 for datasets; notebook Apache-2.0 |
| DavutKursun (XGBoost-on-GLM bag, output pinned by goodpjw2008) | tested, not adopted | Apache-2.0 |
| mojjeed and DavutKursun (GPT-2 Storyteller on TPU, notebook outputs) | tested, not adopted (D30) | Apache-2.0 (notebook outputs) |
| DavutKursun (*The 0.9526 Wall*, stack outputs) and dhananjayk2609 (*Part 3: Final Blend*, stack outputs) | two of the four stacks in `stack_pub4wz` (D31) | Apache-2.0 (Kaggle notebook outputs, default) |
| megayak (4-epoch RealMLP member) | tested, not adopted (D31) | CC0 |
| mathsonchain (17 single models) | OOF members | CC BY 4.0 |
| Masaya Kawamata (GPT-2 target-first classifier) | OOF member | CC BY 4.0, attributed |
| yekenot (RealMLP v15, PyTorch RealMLP, TabM notebook outputs) | OOF members | not recorded in DESIGN.md or `import_public.py` |
| Georgy Mamarin, Alex Chilton, evgendvorkin, aphoristicartist | notebooks read as text for the encoding recipe and artefacts | read as text, not run |
| daoviet, goodpjw2008, Weni Mohamed | zero-fraud zones, read as text | read as text |
| asteriospanoras, kagankoral, areebbb, malyshevdanil, saamhm, zohrab, yashteotia | notebooks and discussions read as text; yashteotia's probe simulation, 278-model pool and original-only GLM (0.934) | read as text |
| Chris Deotte | target-first GPT-2 recipe (S6E10 write-up) and within-fold rank-gaussian idea (S6E3), read as text | read as text |
| Borisov et al., GReaT, ICLR 2023 | random-order row text | published paper |
| Philipp Singer (Psi), TabPFN-3.5 starter | shape of the FinTFM starter notebook | read as text |

## 11. Open questions and ideas not tried

| Idea | Why not tried or not finished |
| --- | --- |
| TabM tuned on a GPU | CPU took 6.7 h for fold 0 and hit the 12-hour limit; the Kaggle GPU quota belongs to another project (D23, D24) |
| A larger or longer GPT-2 member | would cost several times the $9.20 of the three-epoch run, for a member still 0.0018 to 0.0032 below our RealMLP (D26) |
| FinTFM fine-tuned or partitioned-context probe on RunPod | prepared and paused by the owner to save money (D24) |
| Full FinTFM OOF and test as a stack member | about 14 kernel-hours; the one-fold probe gave no blend gain (D5, D8) |
| Rerunning the round-four to seven boosters with `--es inner` | measured stack effect of the fix was nil (D22) |
| Importing seven further public banks | gain would be 0.00004 over the wide stack, and no licence is stated (D29) |
| Removing the zone velocity-edge optimism | no reason recorded in the docs |
| Probing the public board to tune the final file | rejected by design: picks are made by CV (D1, D27) |
| A purpose-built additive model to test section 7.1 | no reason recorded; the spline GLM is the closest we ran (D12) |
| Private-board result | not available until the competition closes on 2026-11-30 |
| Factorization machine (field-aware embeddings with explicit pairwise interactions, as in click-through prediction) | proposed in round twenty-one, not chosen by the owner; no model of this form is in the pool, but section 7.1 suggests few interactions to find (D30) |
| XGBoost stumps or one-column-per-tree boosters on the GLM-boost features | from DavutKursun's *Deeper Trees Lose* (stumps 0.95001 mean fold AUC on encoded features, our folds); low expected value: our depth-2 and 4-leaf boosters already sit at 0.9505-0.9511 (D30) |
| RealMLP minus TabM logit difference as a stack column | idea from a Gram-Schmidt notebook whose figures for it are simulated; no evidence it carries signal (D30) |
| Pseudo-labelling | not run. Adversarial validation finds no shift to correct (D30), and a learning curve bounds the gain: on folds 0 and 1 the best booster scores 0.94691 / 0.94896 / 0.94993 / 0.95066 at 140k / 280k / 420k / 560k training rows, so even the 300k test rows with true labels might add about 0.001 to one booster, and pseudo-labels carry no label information beyond the model that made them (D30) |
| Pairwise ranking objective (XGBoost `rank:pairwise`, GLM start) | run, null (D30): fold AUCs level with the logloss booster; stack_pub5 0.95246, up on 1 of 5 folds |
