# Predicting Credit Card Fraud: Kaggle Playground S6E11

[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![LightGBM 4.7](https://img.shields.io/badge/LightGBM-4.7-blue.svg)
![XGBoost 3.4](https://img.shields.io/badge/XGBoost-3.4-blue.svg)
![CatBoost 1.2](https://img.shields.io/badge/CatBoost-1.2-blue.svg)
![Validation: nested 5-fold CV](https://img.shields.io/badge/validation-nested%205--fold%20CV-informational.svg)

A personal entry to
[Playground Series S6E11: Predicting Credit Card Fraud](https://www.kaggle.com/competitions/playground-series-s6e11),
developed in the open. Every run is recorded, including the many that did not help.

The task: given 24 attributes of a card transaction (amount, timing, velocity, distance from home, merchant
risk, authentication method, device, VPN use and other flags), predict the probability that it is fraud.
1.67% of the 699,960 training rows are fraud. Submissions are scored by ROC-AUC on 299,984 test rows. The
competition closes on 2026-11-30.

> **Status (2026-11-25).** The two final picks are marked on Kaggle:
>
> | Final pick | Nested 5-fold CV | Public LB |
> | :--- | ---: | ---: |
> | `stack_pub4wz`: mean of four independently built nested stacks + zero-fraud zones | **0.95273** | **0.95325** |
> | `stack_pubwz`: our stack averaged with a public wide stack + zero-fraud zones | 0.95263 | 0.95293 |
>
> Our own models alone reach 0.95143 CV (0.95164 public). The private score is not known until the close.

## Contents

- [Approach](#approach)
- [Results](#results)
- [What did not help](#what-did-not-help)
- [Validation](#validation)
- [Quick start](#quick-start)
- [Code](#code)
- [Documentation](#documentation)
- [FinTFM](#fintfm)
- [Licences and data](#licences-and-data)
- [Acknowledgements](#acknowledgements)

## Approach

```
train.csv ──► 5 fixed stratified folds ──► per fold, fitted on the training rows only:
                                              target encoding of exact values, digits, tokens, joint keys
                                                              │
                                       ┌──────────────────────┼───────────────────────┐
                                       ▼                      ▼                       ▼
                                 GLM (logistic,      GLM boost: 4-leaf LightGBM,   RealMLP, MLP,
                                 splines)            XGBoost, CatBoost started     plain shallow trees
                                       │             from the GLM's logit                 │
                                       └──────── out-of-fold + test predictions ──────────┘
                                                              │
                         own models ──► nested logistic stack ◄── fold-checked public members
                                                              │
                     equal-weight mean with three public nested stacks (logit ranks, no fitting)
                                                              │
                                     zero-fraud zones, edges refitted in each fold ──► submission
```

1. **One validation scheme for everything.** Five stratified folds, fixed seed. Every model writes
   out-of-fold (OOF) predictions on them, every stack is scored by nested CV, and every decision is made
   on CV, not on the public leaderboard.
2. **Encode what the generator left behind.** The data is synthetic and its generator reuses the values
   of a 20,000-row original dataset. A column's exact value, its cents, its integer part, its leading
   digits and how often it occurs all carry fraud signal that range splits cannot express. Each becomes a
   category, target-encoded inside the fold.
3. **Simple models on rich encodings.** A regularised logistic regression (GLM) on the encodings, then
   4-leaf trees that start from its logit. The signal is largely additive: shallow beats deep.
4. **Diversity over volume.** Members that err differently (language models, RealMLPs, GLMs) earn stack
   weight; seed bags and near-duplicates do not.
5. **Average independent stacks.** Four nested stacks built by different authors on the same folds agree
   at rank correlation 0.996 to 0.998 out of fold but only 0.992 to 0.999 on test. Their fit-free mean
   removes some of that test-side variance.

## Results

Nested 5-fold CV ROC-AUC on the training rows. Each step is adopted only if CV rises and at least four of
five folds rise. The full table, with every failed run, is in
[docs/DESIGN.md](docs/DESIGN.md#results-at-a-glance).

| Step | CV ROC-AUC |
| :--- | ---: |
| LightGBM, 31 leaves, raw columns | 0.94690 |
| + in-fold target encoding of exact values | 0.94854 |
| 4-leaf LightGBM on target encoding and pairs | 0.94923 |
| GLM on m-estimate encodings and the digit-key family | 0.94987 |
| GLM boost: 4-leaf LightGBM started from the GLM's logit | 0.95056 |
| + pooled artefacts, encodings at three smoothing weights | 0.95076 |
| + joint keys of non-additive column pairs, key counts (best single models: 0.95096 to 0.95098) | 0.95096 |
| Logistic stack of 24 own models | 0.95138 |
| + zero-fraud zones (`stack_r12z`) | 0.95143 |
| Own 21 models + a public nested wide stack (`stack_pub5`) | 0.95249 |
| + zero-fraud zones (`stack_pub5z`) | 0.95254 |
| Fit-free mean with the wide stack + zones (`stack_pubwz`, final pick) | 0.95263 |
| Fit-free mean of four nested stacks + zones (`stack_pub4wz`, final pick) | **0.95273** |

Selected submissions (all are listed in [docs/COMPETITION.md](docs/COMPETITION.md#our-submissions)):

| Date | Submission | CV | Public LB |
| :--- | :--- | ---: | ---: |
| 2026-11-21 | `stack_pub4wz`, final pick | 0.95273 | 0.95325 |
| 2026-11-20 | `stack_pubwz`, final pick | 0.95263 | 0.95293 |
| 2026-11-19 | `stack_pub5z` | 0.95254 | 0.95288 |
| 2026-11-19 | `stack_r12z`, own models only | 0.95143 | 0.95164 |
| 2026-11-16 | Logistic stack of nine own models | 0.95079 | 0.95113 |
| 2026-11-03 | LightGBM, default settings (baseline) | 0.9427 (holdout) | 0.94578 |

Independent CV-backed stacks on these folds meet near 0.9526, and averaging four of them reaches 0.95273.
The public files above about 0.954 that we read were steered by the public board, not validated on
labelled rows ([docs/FINDINGS.md §7.4](docs/FINDINGS.md#74-an-honest-ceiling-near-09526-and-a-board-wall)).

## What did not help

Each was measured on the shared folds and rejected by the 4-of-5 rule. Details are in
[docs/FINDINGS.md §6](docs/FINDINGS.md#6-negative-results).

| Idea | Result |
| :--- | :--- |
| Original rows appended to each training fold | -0.0001 |
| Original-data fraud rates, or a model trained on the original, as features | -0.0002 to -0.0013 |
| Deeper trees (8 leaves; CatBoost depth 6) | -0.00006 to -0.00009 |
| Seed bags of members already in the stack | stack -0.00002 to +0.00001 |
| Own target-first GPT-2 member (1 and 3 epochs) | stack -0.00003 to -0.00007 |
| Pairwise, segment, rank-gaussian and disagreement stackers | never above the logistic stacker |
| Residual booster on the stack's logit; tree-found zero-fraud leaves | stops after 1 to 7 rounds; leaves still hold fraud |
| MinMax tail selection | never gains, at any threshold |
| XGBoost `rank:pairwise` objective | stack 0.95246, 1 of 5 folds up |
| FinTFM linear readout as a member | 0.94054 alone; lowers both stacks |
| Undersampling bags (all frauds, 25% of legitimate rows) | stack 1 of 5 folds up |
| Mean and SD per exact value (IEEE-CIS-style client aggregates) | -0.0008, 0 of 5 |
| Isolation Forest and PCA reconstruction error | 0.81 alone with no labels; stack 2 of 5 folds up |
| Bagged stack, nested weights, G-mean, median or least-correlated blends of the stacks | 0.95268 to 0.95273, at most 3 of 5 |

## Validation

- `StratifiedKFold(5, shuffle=True, random_state=42)` over the competition rows only (`src/common.py`).
  Original rows never enter validation.
- Anything fitted on labels (target encoders, the GLM behind GLM boost, stackers, zone edges) is fitted on
  the fold's training rows only. The training rows' own encodings are cross-fitted inside the fold.
- Since D22, boosters choose their rounds on a 10% holdout of the training fold (`--es inner`), never on
  the scored fold.
- Imported public predictions must match our folds row by row and the AUC their authors report.
- A change is adopted when OOF ROC-AUC improves and at least four of five folds improve.
- The public board holds about a thousand fraud rows. Its differences of a few ten-thousandths are noise,
  so it is a sanity check, not a selector. Adversarial validation cannot tell train from test (AUC 0.500).

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and a configured
[Kaggle API](https://github.com/Kaggle/kaggle-api) token. Download the data as described in
[docs/DATA.md](docs/DATA.md); nothing in `data/` is committed.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q                # unit tests, synthetic data, about 15 s
.venv/bin/ruff check src scripts tests       # lint

cd src
# GLM boost with honest early stopping: run r11_gb_all_kc, CV 0.95092 (writes runs/<name>/)
../.venv/bin/python train_gbdt.py --model lgb --offset glm --fe art --orig 0 --te single --keys full \
    --te-smooth 5,20,100 --joint glm --key-counts --es inner --name r11_gb_all_kc --params \
    '{"num_leaves":4,"learning_rate":0.03,"feature_fraction":0.5,"min_child_samples":100,"lambda_l2":1.0,"n_estimators":20000}'
# A one-fold smoke test on 20,000 rows saves nothing
../.venv/bin/python train_gbdt.py --model glm --te single --keys full --orig 0 --rows 20000 --folds 1 --name smoke
# Nested-CV stack of saved runs, its OOF scores, a submission, then zones
../.venv/bin/python stack.py RUN [RUN ...] --no-hill --C 0.1 --oof-out ../runs/s_oof.npy --out ../submissions/s.csv
../.venv/bin/python zones.py --oof ../runs/s_oof.npy --sub ../submissions/s.csv --out ../submissions/s_z.csv
```

Long runs go to private Kaggle CPU kernels: `scripts/queue_kernel.sh K` runs every line of `jobs/K.txt`
when a slot frees and fetches the runs. Environment variables: `S6E11_THREADS` (default 4), `S6E11_DATA`,
`S6E11_RUNS`.

## Code

Scripts live in `src/` and run from there. Every model writes `runs/<name>/{oof.npy,test.npy,meta.json}`
on the shared folds, so any set of runs can be stacked.

| Module | Role |
| :--- | :--- |
| `common.py` | paths, columns, data loading, the fixed folds, `CVScore`, rank transforms, run files |
| `features.py` | feature sets (`--fe`) and target-encoding keys: exact values, digits, GPT-2 tokens, joint keys |
| `encoding.py` | `FoldEncoder`: in-fold, cross-fitted target encoding; hierarchical encodings; key counts |
| `learners.py` | one `Learner` class per model family (GLM, MLP, RealMLP, LightGBM, XGBoost, CatBoost) |
| `train_gbdt.py` | the CV driver: command line, option checks, fold loop, seed and undersampling bags |
| `stack.py` | rank average, logistic and hill-climbing stackers, each scored by nested CV |
| `zones.py` | zero-fraud zones with edges refitted in each fold |
| `stack_average.py` | fit-free mean of nested stacks on logit ranks (`stack_pub4wz`) |
| `import_public.py` | imports public OOF banks after fold, shape and AUC checks |
| `stack_ideas.py`, `minmax_tails.py`, `residual.py` | stacker experiments (D28, D29) |
| `adv_val.py`, `research_artefacts.py`, `research_ogmatch.py` | measurements: train/test shift, generator artefacts |
| `train_lm.py` | target-first GPT-2 member (GPU; launched on RunPod by `scripts/runpod_lm.py`) |
| `train_fintfm.py`, `train_readout.py`, `probe_readout.py` | FinTFM members and probes |

`scripts/kaggle_kernel.py` bundles `src/` and a job list into a private Kaggle kernel, then pushes it,
checks its status or fetches its runs. `tests/` holds unit tests for the shared modules: no leakage
through the encoder, every model family, option checks, zones and import checks. They run on synthetic
data, with no competition files.

## Documentation

| File | Contents |
| :--- | :--- |
| [docs/FINDINGS.md](docs/FINDINGS.md) | the paper-style wrap-up: method, results, negative results, threats to validity |
| [docs/DESIGN.md](docs/DESIGN.md) | the lab notebook: every run and decision, D1 to D32 |
| [docs/COMPETITION.md](docs/COMPETITION.md) | metric, deadline, how to read the leaderboard, every submission |
| [docs/DATA.md](docs/DATA.md) | files, columns, measured facts, download commands |

## FinTFM

[FinTFM](https://github.com/kabartay/fintfm) is a small tabular foundation model by the same author. It
predicts a row from labelled example rows in one forward pass. It was tested here as a stack member: zero-shot
it reaches 0.907 on a fold against 0.937 for LightGBM, and its 1.0.0 linear readout reaches 0.94054 CV. Neither
improves the stack ([D8](docs/DESIGN.md#d8-fintfm-probe-balanced-context-is-best-but-it-does-not-lift-the-trees),
D29). [`notebooks/fintfm-starter/`](notebooks/fintfm-starter/) builds a public, educational
[Kaggle notebook](https://www.kaggle.com/code/muhakabartay/fintfm-starter-playground-series-s6e11) on
in-context learning for tables. [`notebooks/honest-cv/`](notebooks/honest-cv/) builds
[a second one](https://www.kaggle.com/code/muhakabartay/s6e11-own-models-only-honest-5-fold-cv) with six of our
own models, a nested stack and the zones.

## Licences and data

- Code: [MIT Licence](LICENSE).
- Competition data: not included; subject to the
  [competition rules](https://www.kaggle.com/competitions/playground-series-s6e11/rules).
- Original dataset: not included;
  [CC0-1.0](https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026).
- Public predictions used as stack members stay in `data/public/` and are never committed. Their licences
  are listed in [docs/FINDINGS.md §10](docs/FINDINGS.md#10-credits-and-licences).

## Acknowledgements

Thanks to Kaggle for the Playground Series and to the author of the original Credit Card Fraud Detection
2026 dataset. All code here is written independently. Ideas came from public notebooks read as text:

- **Encoding recipe and generator artefacts:**
  [Georgy Mamarin](https://www.kaggle.com/code/georgymamarin/s6e11-fraud-glm-offset-trees-review-budgets),
  [Alex Chilton](https://www.kaggle.com/code/alexchilton/credit-card-fraud-lb-0-9542-shallow-tree-stack) and
  [evgendvorkin](https://www.kaggle.com/code/evgendvorkin/s6e11-deep-eda-artifacts-xgb-oof-0-950285).
- **Joint keys and key counts:**
  [megayak](https://www.kaggle.com/code/megayak/s6e11-per-value-digit-te-shallow-tree-stack) (Apache-2.0).
- **Zero-fraud zones:** daoviet, goodpjw2008 and Weni Mohamed.

Out-of-fold predictions came from:

- [goodpjw2008](https://www.kaggle.com/datasets/goodpjw2008/s6e11-oof-bank) (CC0), including the nested
  [wide LM stack](https://www.kaggle.com/datasets/goodpjw2008/s6e11-fraud-series-artifacts);
- the stacks of DavutKursun (*The 0.9526 Wall*) and dhananjayk2609 (*Part 3: Final Blend*), notebook outputs
  under Apache-2.0;
- [yekenot's RealMLP](https://www.kaggle.com/code/yekenot/ps-s6-e11-realmlp-pytabkit);
- [Aman Atar's member bundle](https://www.kaggle.com/datasets/amanatar/s6e11-rank1-q17-meta-bundle) (CC0);
- [busyaprime](https://www.kaggle.com/datasets/busyaprime/s6e11-33-own-members) (CC0);
- [dariushafshar](https://www.kaggle.com/datasets/dariushafshar/s6e11-golem-oof-library) (CC0);
- [alexchilton](https://www.kaggle.com/datasets/alexchilton/s6e11-oof-predictions) (CC0);
- [mathsonchain](https://www.kaggle.com/datasets/mathsonchain/s6e11-fraud-oof-test-17-single-models) (CC BY 4.0);
- [Masaya Kawamata](https://www.kaggle.com/datasets/masayakawamata/s6e11-gpt2-target-first-oof) (CC BY 4.0).

The target-first GPT-2 method is Chris Deotte's (S6E10 write-up), after Borisov et al., GReaT, ICLR 2023.
The FinTFM notebook takes its shape from
[Psi's TabPFN-3.5 starter](https://www.kaggle.com/code/philippsinger/tabpfn-3-5-starter-playground-series-s6e11).
