#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bash "${repo_root}/panoptic/scripts/run_sparxo_v2_paps_1_v2u4init_fold_4gpu.sh" "$@"
