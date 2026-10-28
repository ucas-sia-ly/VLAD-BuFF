#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source reproduction/env.sh
if [[ "${VLADBUFF_RESOURCE_SCOPE:-0}" != 1 ]]; then
  experiment_scope="${TRAIN_TERMINAL_RUN_DIR:-standalone}"
  exec systemd-run --user --scope --unit="vladbuff-${experiment_scope##*/}-$$" \
    -p MemoryMax=10G -p MemoryHigh=9G -p MemorySwapMax=0 \
    /usr/bin/env VLADBUFF_RESOURCE_SCOPE=1 /usr/bin/bash "$PWD/reproduction/run.sh" "$@"
fi
export VLADBUFF_GPU_ALLOC_GIB=38
exec "$VLADBUFF_PYTHON" -u reproduction/resource_guard.py "$VLADBUFF_PYTHON" -u reproduction/run.py "$@"
