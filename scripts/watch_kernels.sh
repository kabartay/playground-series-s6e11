#!/bin/bash
# Poll Kaggle kernels every 5 min; fetch each one's runs when it completes; exit when all are done.
# Usage: scripts/watch_kernels.sh lgb1 xgb1 cat1 tfm1
cd "$(dirname "$0")/.." || exit 1
pending=("$@")
while [ ${#pending[@]} -gt 0 ]; do
  left=()
  for k in "${pending[@]}"; do
    st=$(kaggle kernels status "muhakabartay/s6e11-$k" 2>/dev/null | grep -o 'KernelWorkerStatus\.[A-Z_]*')
    case "$st" in
      *COMPLETE) echo "$(date +%H:%M) $k complete"; .venv/bin/python scripts/kaggle_kernel.py fetch --kernel "$k" 2>&1 | grep -v -i "outdated\|downloaded" ;;
      *ERROR|*CANCEL*) echo "$(date +%H:%M) $k $st"; .venv/bin/python scripts/kaggle_kernel.py fetch --kernel "$k" 2>&1 | grep -v -i "outdated\|downloaded" ;;
      *) left+=("$k") ;;
    esac
  done
  pending=("${left[@]}")
  [ ${#pending[@]} -gt 0 ] && sleep 300
done
echo "all kernels finished"
