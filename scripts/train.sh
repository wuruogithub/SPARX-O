#!/usr/bin/env bash
set -euo pipefail
config=${1:-configs/pastis.yaml}
shift || true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" \
  -m sparxo.runtime.train --config "$config" "$@"
