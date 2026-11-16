# FinTFM starter notebook

An educational Kaggle notebook: FinTFM 0.11.0 zero-shot on S6E11, with measured sweeps over context size,
context strategy, partitioned contexts and ensembling, a LightGBM reference on the same holdout, a blend
check, and a submission from the best setting.

- `build_notebook.py` generates `fintfm-starter-s6e11.ipynb`; edit the script, not the notebook.
- Smoke test: extract the code cells and run with `FINTFM_SMOKE=1 MPLBACKEND=Agg` (tiny holdout and test).
- Push (private): `kaggle kernels push -p notebooks/fintfm-starter`. Making it public is the owner's decision.
- CPU session by default (about 70 rows/s at a 2,000-row context on Kaggle CPU, measured 2026-11-16, D8); the
  notebook reserves 5 h of its 11 h budget for the test predictions and skips experiments that would not fit.
