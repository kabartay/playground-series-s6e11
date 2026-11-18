# Working rules for this repository

Personal Kaggle entry (not Finkele work). Repository, docs and chat are in English.

## Read first

`docs/COMPETITION.md` (metric, deadline, our submissions), `docs/DATA.md` (measured data facts),
`docs/DESIGN.md` (decisions and every result, including failures).

## Competition essentials

- Deadline 2026-11-30 23:59 UTC. Metric ROC-AUC. Ask the owner before each Kaggle submission unless told to submit freely.
- Competition data is never committed or uploaded anywhere public. The original dataset is CC0 but stays in `data/` too.
- The repo is public (since 2026-11-16, owner's choice). Everything committed is visible: no data, credentials or other people's code.

## Code

- Run tools with `.venv/bin/python` (Python 3.12, `uv pip install -r requirements.txt`). Scripts live in `src/` and run from there.
- New code gets type hints and docstrings. Lint with `.venv/bin/ruff check src`.
- Every model writes `runs/<name>/{oof.npy,test.npy,meta.json}` on the shared folds in `src/common.py`; never change the fold seed.
- Record every run, adopted or not, in the results table of `docs/DESIGN.md`.

## Machine

- The Mac is shared with a 24/7 pipeline and other sessions. Check `uptime` and `memory_pressure` first; keep `THREADS` at 4 or below.
- FinTFM runs on CPU only here, and only when the machine is quiet; see the fintfm repo's CLAUDE.md for its memory limits.
- Kaggle GPU quota belongs to fintfm: do not use it here unless the owner says so. Exception (owner, 2026-11-18): one own GPT-2 language-model run of about 10 GPU hours, only after the weekly 30-hour quota resets.

## Git

- Author `mukharbek.organokov@gmail.com`. No Claude or AI attribution trailers.
