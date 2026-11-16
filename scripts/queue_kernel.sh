#!/bin/bash
# Push a kernel, retrying every 5 min while all CPU slots are busy, then watch it and fetch its runs.
# Usage: scripts/queue_kernel.sh <kernel> [--internet]
cd "$(dirname "$0")/.." || exit 1
k=$1; shift
until out=$(.venv/bin/python scripts/kaggle_kernel.py push --kernel "$k" --jobs "jobs/$k.txt" "$@" 2>&1 | grep -v outdated); echo "$(date +%H:%M) $k: $out"; ! echo "$out" | grep -q "session count"; do sleep 300; done
sleep 60
exec scripts/watch_kernels.sh "$k"
