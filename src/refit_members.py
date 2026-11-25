"""Refit test predictions for the own members of a stack (D34): one ``train_gbdt.py --refit-only`` per member.

Each member's recipe is read back from its ``runs/<name>/meta.json`` (the command line it was trained with),
so nothing is retyped. Every refit uses ``--es inner``: the members of rounds four to eight early-stopped on
the validation fold, which a fit on all rows does not have. Public members (``pub_*``) are skipped; they keep
their published test scores.

    ../.venv/bin/python refit_members.py ../submissions/stack_pub5.json --print   # job lines for jobs/*.txt
    ../.venv/bin/python refit_members.py ../submissions/stack_pub5.json           # run them here, one by one
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from common import RUNS
from train_gbdt import parse_args

#: Options that describe a run's recipe; everything else in meta.json is a result or a check setting.
RECIPE = (
    "model", "fe", "orig", "params", "te", "te_smooth", "keys", "offset", "joint", "key_counts",
    "te_cv", "enc_seed", "hier", "seeds", "seed0", "neg_frac",
)  # fmt: skip


def refit_argv(name: str) -> list[str]:
    """Command line of ``train_gbdt.py --refit-only`` for a saved run, rebuilt from its meta.json.

    Raises:
        FileNotFoundError: the run has no meta.json.
    """
    meta = json.loads((RUNS / name / "meta.json").read_text())
    defaults = vars(parse_args(["--model", meta["model"], "--name", name]))
    argv = ["--name", name, "--es", "inner", "--refit-only"]
    for key in RECIPE:
        value = meta.get(key, defaults[key])
        if value is None or (value == defaults[key] and key != "model"):
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            argv += [flag] if value else []
        elif key == "params":
            argv += [flag, json.dumps(json.loads(value), separators=(",", ":"))]
        else:
            argv += [flag, str(value)]
    return argv


def main(argv: list[str] | None = None) -> int:
    """Print or run the refits of every own member of the stack; return the number that failed."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stack", type=Path, help="stack JSON written by stack.py --out")
    ap.add_argument("--print", action="store_true", help="print one jobs-file line per member and stop")
    a = ap.parse_args(argv)
    runs = [r for r in json.loads(a.stack.read_text())["runs"] if not r.startswith("pub_")]
    failed = 0
    for r in runs:
        cmd = refit_argv(r)
        if a.print:
            print(" ".join(cmd))
            continue
        print(f"== {r}", flush=True)
        failed += subprocess.run([sys.executable, "train_gbdt.py", *cmd], check=False).returncode != 0
    return failed


if __name__ == "__main__":
    sys.exit(main())
