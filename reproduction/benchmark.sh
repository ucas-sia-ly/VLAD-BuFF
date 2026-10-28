#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source reproduction/env.sh
if [[ "${VLADBUFF_RESOURCE_SCOPE:-0}" != 1 ]]; then
  experiment_scope="${TRAIN_TERMINAL_RUN_DIR:-standalone}"
  exec systemd-run --user --scope --unit="vladbuff-data-${experiment_scope##*/}-$$" \
    -p MemoryMax=8G -p MemoryHigh=7G -p MemorySwapMax=0 \
    /usr/bin/env VLADBUFF_RESOURCE_SCOPE=1 VLADBUFF_RAM_MAX_GIB=8 VLADBUFF_RAM_HIGH_GIB=7 \
    /usr/bin/bash "$PWD/reproduction/benchmark.sh" "$@"
fi
export VLADBUFF_GPU_ALLOC_GIB=38
exec "$VLADBUFF_PYTHON" -u reproduction/resource_guard.py "$VLADBUFF_PYTHON" -u reproduction/benchmark.py "$@"
