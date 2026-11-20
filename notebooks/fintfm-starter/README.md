# FinTFM starter notebook

An educational Kaggle notebook: FinTFM (1.0.0 since notebook version 2) on S6E11, with measured sweeps over context size,
context strategy, partitioned contexts and ensembling, a LightGBM reference on the same holdout, a blend
check, and a submission from the best setting.

- `build_notebook.py` generates `fintfm-starter-s6e11.ipynb`; edit the script, not the notebook.
- Smoke test: extract the code cells and run with `FINTFM_SMOKE=1 MPLBACKEND=Agg` (tiny holdout and test).
- Public on Kaggle since 2026-11-17 (owner's choice): https://www.kaggle.com/code/muhakabartay/fintfm-starter-playground-series-s6e11.
  `kaggle kernels push -p notebooks/fintfm-starter` publishes a new version, which reruns the notebook (about 5.4 h on CPU).
- CPU session by default (about 70 rows/s at a 2,000-row context on Kaggle CPU, measured 2026-11-16, D8); the
  notebook reserves 5 h of its 11 h budget for the test predictions and skips experiments that would not fit.

## First full run (Kaggle CPU, version 1, 2026-11-16)

FinTFM 0.11.0, 4 CPU threads, 20,000-row stratified holdout (334 fraud), context drawn from the other
679,960 rows. Total run time 5.4 h, of which 4.3 h for the 299,984 test rows.

| Setting | ROC-AUC | Average precision | Seconds | Rows/s |
| --- | --- | --- | --- | --- |
| LightGBM, 500 rounds, defaults (reference) | 0.9356 | 0.4735 | 17 | 1,149 |
| uniform, 250 rows | 0.8341 | 0.1158 | 102 | 196 |
| uniform, 500 rows | 0.8556 | 0.1547 | 118 | 170 |
| uniform, 1,000 rows | 0.8741 | 0.1689 | 160 | 125 |
| uniform, 2,000 rows | 0.8857 | 0.2000 | 266 | 75 |
| uniform, 4,000 rows | 0.8990 | 0.2412 | 557 | 36 |
| balanced, 2,000 rows | 0.9063 | 0.2973 | 267 | 75 |
| retrieval, 2,000 rows | 0.7864 | 0.1028 | 630 | 32 |
| hybrid, 2,000 rows | 0.9038 | 0.2779 | 270 | 74 |
| partitioned (depth 2) + balanced, 2,000 rows | 0.9053 | 0.3185 | 290 | 69 |
| balanced, 2,000 rows, n_ensemble=4 | **0.9089** | 0.3000 | 1,108 | 18 |

- AUC rises steadily with context size; a class-balanced context of 2,000 rows beats a uniform one of
  4,000. Retrieval (nearest neighbours) context is worst here.
- Partitioning raises average precision (0.3185) but not AUC; ensembling 4 contexts gives the best AUC.
- Blend with LightGBM on the holdout: best share 0.05 FinTFM, 0.9358 against 0.9356 (rank correlation
  0.866). Within noise on 334 fraud rows.
- The notebook's submission (balanced, 2,000 rows, n_ensemble=4) has not been submitted to the competition.

Version 1 on Kaggle still calls the LightGBM reference "tuned"; the repository's notebook says "untuned"
(1a89412). The correction goes out with the next pushed version.

## Version 2 (fintfm 1.0.0, pushed 2026-11-20 as Kaggle version 3)

Same checkpoint (commit f116bfd, SHA-256 d231b06...), so sections 3-8 reproduce version 1. New section 9: the
linear readout (`fintfm.readout.LinearReadoutModel`, entry defaults), fitted on the whole pool. Probed
beforehand on Kaggle CPU (`src/probe_readout.py`, jobs ro1 and ro2) on the same 20,000-row holdout:

| Readout fitted on | Holdout AUC | Fit | Holdout scoring |
| --- | --- | --- | --- |
| 5,000 rows (first 2,000 holdout rows only) | 0.9207 | 219 s | 39 rows/s |
| 20,000 rows | 0.9112 | 331 s | 138 rows/s |
| 100,000 rows | 0.9295 | 874 s | 138 rows/s |
| 300,000 rows | 0.9308 | 905 s | 138 rows/s |

The readout represents at most 100,000 rows itself, so the fit cost is flat beyond that. Smoke copy
(`SMOKE = True`, private kernel s6e11-ftfm-v2-smoke) ran every section end to end in 0.3 h.

Kaggle version 3 (2026-11-20, 2.2 h on CPU): sections 3-8 reproduced version 1 (balanced 2,000 rows 0.9063,
LightGBM 0.9356); the readout scored 0.9296 on the holdout (AP 0.4531) and was chosen for the submission,
which took 3,149 s for 299,984 rows. Public score 0.94105, against 0.91516 for Kaggle version 2.
