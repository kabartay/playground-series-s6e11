r"""Run src/train_lm.py folds on a RunPod GPU pod, copy the fold files back, delete the pod.

Adapted from the owner's fintfm launcher (`scripts/runpod/run_kernel_on_pod.py` there):

1. create a pod (default one `NVIDIA A40`, Secure Cloud, host CUDA >= 13.2) whose container starts
   sshd with the account's registered public key, installs torch and transformers, writes
   `/work/READY` and then idles;
2. once SSH answers and the pod is ready, stream `src/` and the two competition CSVs into the pod
   over SSH (the pod is private and deleted afterwards; nothing is uploaded anywhere else);
3. start `train_lm.py --folds <k>` in the background and poll `/work/EXIT` over SSH;
4. stream `/work/runs` back into `runs/` as a tar archive;
5. delete the pod, always: on success, on failure, on Ctrl-C, on SIGTERM and when `--max-hours` passes.

The API key is read from `RUNPOD_API_KEY` (`set -a; . ./.env; set +a` inside the calling command)
and is never printed or sent to the pod. It needs only Pods read/write (Catalog and Billing read
for the checks in `--dry-run`).

Example (fold 0 of the GPT-2 member; one pod per fold, then merge locally)::

    set -a; . ./.env; set +a; .venv/bin/python scripts/runpod_lm.py --name r14_gpt2 --folds 0 \
        -- --train-min 90 --oof-orders 2 --eval-bs 128
    .venv/bin/python src/train_lm.py --name r14_gpt2 --merge-only

Everything after `--` is passed to train_lm.py unchanged.

Exit status: train_lm.py's, `3` when the pod never produced an exit code, `2` on a usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

#: RunPod REST API root.
API = "https://api.runpod.io/v2"
#: Repository root.
ROOT = Path(__file__).resolve().parents[1]
#: Private half of the SSH key registered in the RunPod console.
SSH_KEY = Path.home() / ".ssh" / "id_ed25519"
#: Seconds between status polls.
POLL_S = 60
#: Python packages the pod installs (torch from PyPI ships its own CUDA runtime).
PIP = "torch transformers numpy pandas scikit-learn"


class RunPod:
    """Minimal client for the RunPod v2 REST API.

    Args:
        key: API key. Kept only in the `Authorization` header, never logged.
    """

    def __init__(self, key: str) -> None:
        self._headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def call(self, method: str, path: str, body: dict | None = None, timeout: int = 60) -> dict:
        """Send one request and return the decoded JSON reply (`{}` for an empty body).

        Raises:
            RuntimeError: The API answered with an error status.
        """
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(API + path, data=data, method=method, headers=self._headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode()
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"{method} {path}: HTTP {err.code}: {err.read().decode()[:500]}") from None
        return json.loads(text) if text.strip() else {}


def container_command() -> str:
    """The pod's entrypoint: start sshd, install the Python packages, mark ready, then idle."""
    return (
        "mkdir -p /work /root/.ssh /run/sshd && chmod 700 /root/.ssh && "
        'echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys; '
        "(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server"
        " && /usr/sbin/sshd) > /work/sshd.log 2>&1; "
        f"pip install -q {PIP} > /work/pip.log 2>&1 && touch /work/READY; sleep infinity"
    )


def ssh_args(direct: dict) -> list[str]:
    """The `ssh` command prefix for the pod's direct endpoint (host keys unchecked: each pod is new)."""
    return [
        "ssh", "-i", str(SSH_KEY), "-p", str(direct["port"]),
        "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
        f"{direct['username']}@{direct['host']}",
    ]  # fmt: skip


def remote(direct: dict, script: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run `script` in the pod over SSH and capture its output; returncode 255 means SSH failed."""
    cmd = [*ssh_args(direct), script]
    try:
        return subprocess.run(
            cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 255, "", "timeout")


def upload(direct: dict) -> None:
    """Stream `src/` and the competition CSVs into `/work/repo` in the pod.

    Raises:
        RuntimeError: tar or SSH failed.
    """
    files = ["src", "data/train.csv", "data/test.csv"]
    tar = subprocess.Popen(
        ["tar", "czf", "-", "--exclude=__pycache__", *files], cwd=ROOT, stdout=subprocess.PIPE
    )
    proc = subprocess.run(
        [*ssh_args(direct), "mkdir -p /work/repo && tar xzf - -C /work/repo"],
        stdin=tar.stdout, capture_output=True, timeout=900, check=False,
    )  # fmt: skip
    tar.wait()
    if proc.returncode or tar.returncode:
        raise RuntimeError(
            f"upload failed: ssh {proc.returncode}, tar {tar.returncode}: {proc.stderr.decode()[:300]}"
        )


def fetch(direct: dict, out: Path, name: str, folds: str) -> None:
    """Stream the pod's `/work/runs` and `/work/pod.log` back and unpack them under `out`.

    Raises:
        RuntimeError: tar or SSH failed.
    """
    out.mkdir(parents=True, exist_ok=True)
    archive = out / f"pod_outputs_{name}_f{folds.replace(',', '')}.tgz"  # unique: pods finish together
    with archive.open("wb") as fh:
        proc = subprocess.run(
            [*ssh_args(direct), "cd /work && tar czf - runs pod.log"],
            stdout=fh, stderr=subprocess.PIPE, timeout=900, check=False,
        )  # fmt: skip
    if proc.returncode:
        raise RuntimeError(f"tar over SSH exited {proc.returncode}: {proc.stderr.decode()[:300]}")
    with tarfile.open(archive) as tar:
        tar.extractall(out, filter="data")
    archive.unlink()
    log_file = out / "pod.log"
    if log_file.exists():  # keep each pod's log beside its fold files, not in the repository root
        log_file.rename(out / "runs" / name / f"pod_f{folds.replace(',', '')}.log")
    log(f"copied outputs to {out}")


def log(msg: str) -> None:
    """Print one timestamped progress line."""
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def run(api: RunPod, args: argparse.Namespace, lm_args: list[str]) -> int:
    """Create, prepare, train, copy back and delete; return train_lm.py's exit status (`3` if none)."""
    body = {
        "name": f"s6e11-{args.name}-f{args.folds.replace(',', '')}",
        "image": args.image,
        "entrypoint": ["bash", "-c", container_command()],
        "disk": args.disk_gb,
        "cloud": args.cloud,
        "ports": ["22/tcp"],
        "startSsh": True,
        "gpu": {"id": args.gpu, "count": 1, "minCudaVersion": args.min_cuda,
                "minVcpuCountPerGpu": args.min_vcpu, "minRamPerGpu": args.min_ram_gb},
    }  # fmt: skip
    pod = api.call("POST", "/pods", body)
    pod_id, cost = pod["id"], pod.get("cost", "?")
    log(f"created pod {pod_id} ({args.gpu}, {args.cloud}, ${cost}/h)")
    started = time.monotonic()
    code: int | None = None
    direct: dict | None = None
    stage = "boot"  # boot -> train -> done
    running_since: float | None = None
    last_ok: float | None = None  # last successful SSH probe
    last = ""
    try:
        while time.monotonic() - started < args.max_hours * 3600:
            time.sleep(POLL_S)
            info = api.call("GET", f"/pods/{pod_id}")
            status = info.get("status")
            if status in ("ERROR", "TERMINATED", "EXITED"):
                log(f"pod status {status}; giving up")
                break
            if status != "RUNNING":
                log(f"pod status {status}")
                continue
            running_since = running_since or time.monotonic()
            direct = (info.get("ssh") or {}).get("direct") or direct
            probe = (
                remote(
                    direct,
                    "ls /work; cat /work/EXIT 2>/dev/null; echo ---; tail -n 1 /work/pod.log 2>/dev/null",
                )
                if direct
                else None
            )
            if probe is None or probe.returncode == 255:
                waited = (time.monotonic() - (last_ok or running_since)) / 60
                log(f"SSH not ready after {waited:.0f} min of RUNNING")
                if waited > args.connect_minutes:
                    log(f"no SSH within --connect-minutes {args.connect_minutes}; giving up")
                    break
                continue
            last_ok = time.monotonic()
            head, _, tail = probe.stdout.partition("---")
            if stage == "boot":
                if "READY" not in head.split():
                    log("pod installing packages")
                    if (time.monotonic() - running_since) / 60 > args.connect_minutes + 15:
                        log("packages not installed in time; giving up")
                        break
                    continue
                upload(direct)
                lm = " ".join(shlex.quote(x) for x in ["--name", args.name, "--folds", args.folds, *lm_args])
                # Refuse to train on CPU: a torch wheel that cannot see the GPU would burn the budget.
                job = (
                    "python -c \"import torch; assert torch.cuda.is_available()\" && "
                    f"S6E11_RUNS=/work/runs python train_lm.py {lm}"
                )
                remote(
                    direct,
                    "cd /work/repo/src && setsid nohup sh -c "
                    f"'({job}) > /work/pod.log 2>&1; echo $? > /work/EXIT' < /dev/null > /dev/null 2>&1 &",
                )
                log(f"uploaded and started train_lm.py {lm}")
                stage = "train"
                continue
            line = tail.strip()
            if line and line != last:
                log(f"pod: {line[:200]}")
                last = line
            exit_line = [w for w in head.split() if w.lstrip("-").isdigit()]
            if exit_line:
                code = int(exit_line[-1])
                log(f"train_lm.py exit {code} after {(time.monotonic() - started) / 60:.0f} min")
                break
        else:
            log(f"--max-hours {args.max_hours} passed; deleting the pod")
        if direct and stage == "train":
            try:
                fetch(direct, Path(args.out), args.name, args.folds)
            except (RuntimeError, subprocess.TimeoutExpired) as err:
                log(f"copy failed: {err}")
    finally:
        hours = (time.monotonic() - started) / 3600
        try:
            api.call("DELETE", f"/pods/{pod_id}")
            log(f"deleted pod {pod_id}; about {hours:.2f} h at ${cost}/h")
        except RuntimeError as err:
            log(f"DELETE FAILED, delete pod {pod_id} in the console: {err}")
    return 3 if code is None else code


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse the launcher's options; everything after `--` goes to train_lm.py."""
    lm_args: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, lm_args = argv[:i], argv[i + 1 :]
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--name", required=True, help="run name, e.g. r14_gpt2")
    p.add_argument("--folds", required=True, help="folds this pod trains, e.g. 0 or 0,1")
    p.add_argument("--out", default=str(ROOT), help="local root; outputs land in <out>/runs/<name>/")
    p.add_argument("--gpu", default="NVIDIA A40", help="RunPod GPU type ID")
    p.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    p.add_argument("--image", default="python:3.12")
    p.add_argument("--min-cuda", default="13.2", help="lowest host CUDA version")
    p.add_argument("--min-vcpu", type=int, default=8)
    p.add_argument("--min-ram-gb", type=int, default=32)
    p.add_argument("--disk-gb", type=int, default=40)
    p.add_argument(
        "--max-hours", type=float, default=3.5, help="delete the pod after this long, whatever happens"
    )
    p.add_argument("--connect-minutes", type=float, default=15.0)
    p.add_argument(
        "--dry-run", action="store_true", help="print balance, stock and the pod body; create nothing"
    )
    return p.parse_args(argv), lm_args


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args, lm_args = parse_args(sys.argv[1:] if argv is None else argv)
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("RUNPOD_API_KEY is not set; source .env inside the command", file=sys.stderr)
        return 2
    if not SSH_KEY.exists():
        print(f"{SSH_KEY} is missing", file=sys.stderr)
        return 2
    if not (ROOT / "data" / "train.csv").exists():
        print("data/train.csv is missing", file=sys.stderr)
        return 2
    api = RunPod(key)
    if args.dry_run:
        print("balance:", api.call("GET", "/billing/balance"))
        gpu = urllib.parse.quote(args.gpu)
        print(
            "stock:",
            json.dumps(api.call("GET", f"/catalog/gpus/{gpu}?include=AVAILABILITY&product=POD"))[:800],
        )
        print("train_lm args:", ["--name", args.name, "--folds", args.folds, *lm_args])
        return 0
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    return run(api, args, lm_args)


if __name__ == "__main__":
    sys.exit(main())
