# Own models, honest CV: public notebook draft

A teaching notebook: six of our own models (three GLMs, two shallow LightGBMs, one RealMLP) trained with
`src/train_gbdt.py` on the shared folds, a nested logistic stack, the zero-fraud zones checked with their
edges refitted inside each fold, a measurement of public-board noise, and a table of what did not help.

- `build_notebook.py` generates `s6e11-own-models-honest-cv.ipynb` and embeds `src/common.py`,
  `src/features.py` and `src/train_gbdt.py` as they are; edit the script or src/, not the notebook.
  Build: `uv run --no-project --with nbformat python notebooks/honest-cv/build_notebook.py`.
- Local check on cached runs (2026-11-19, the six members linked from runs/): stack 0.95129, with zones
  0.95135 (5 of 5 folds up), best single 0.95092.
- Kaggle: pushed private first (`kaggle kernels push -p notebooks/honest-cv`); made public only on the
  owner's yes, by setting `is_private` to false and pushing again.
