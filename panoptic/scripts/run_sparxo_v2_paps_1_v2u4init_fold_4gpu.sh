#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "usage: $0 <PASTIS_test_fold:1-5> <SPARXO_v2u4_checkpoint> <PASTIS-R_root> [output]" >&2
  exit 2
fi

test_fold=$1
source_init=$2
dense_root=$3
if [[ ! "${test_fold}" =~ ^[1-5]$ ]]; then
  echo "PASTIS test fold must be in [1,5], got: ${test_fold}" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
task_root=$(realpath -m "${4:-outputs/sparxo_v2_u4_paps_fold${test_fold}_seed42}")
max_epochs=${SPARXO_PAPS_MAX_EPOCHS:-150}
early_stopping_min_epoch=${SPARXO_PAPS_EARLY_STOPPING_MIN_EPOCH:-70}
early_stopping_patience=${SPARXO_PAPS_EARLY_STOPPING_PATIENCE:-30}

# Official Exchanger configs number Fold5 as Fold0; folds 1-4 retain their ids.
if [[ "${test_fold}" == "5" ]]; then
  official_fold=0
else
  official_fold=${test_fold}
fi
cli_fold=$((official_fold + 1))

temporal_init=${task_root}/sparxo_v2_u4_fold${test_fold}_temporal_init.pth
cfg=configs/pastis/finetune/pastis_panoptic_sparxo_v2_paps_1_Fold${official_fold}.yml
run_dir=${task_root}/PASTIS-R/pastis_panoptic_sparxo_v2_paps_1_Fold${official_fold}/Fold_${official_fold}
best=${run_dir}/best.pth
final=${run_dir}/final_state.pth
checkpoint=${run_dir}/checkpoint.pth.tar
test_result=${run_dir}/panoptic_test_fold${test_fold}_best.json

for value_name in max_epochs early_stopping_min_epoch early_stopping_patience; do
  value=${!value_name}
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${value_name} must be a non-negative integer, got: ${value}" >&2
    exit 2
  fi
done
if (( max_epochs < 1 || early_stopping_min_epoch > max_epochs )); then
  echo "Require max_epochs >= 1 and early_stopping_min_epoch <= max_epochs" >&2
  exit 2
fi

mkdir -p "${task_root}"
test -s "${source_init}"
if [[ ! -s "${temporal_init}" ]]; then
  python "${repo_root}/scripts/export_sparxo_temporal_init.py" \
    --source "${source_init}" --output "${temporal_init}"
fi

cd "${repo_root}"
export PYTHONPATH="${repo_root}:${repo_root}/..:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export EXCHANGER_COMPACT_LOG=1

common=(
  OUTPUT_DIR "${task_root}"
  LOG_DIR "${task_root}/logs"
  DATASET.ROOT "${dense_root}"
  MODEL.PRETRAINED "${temporal_init}"
  TRAIN.END_EPOCH "${max_epochs}"
  TRAIN.EARLY_STOPPING_MIN_EPOCH "${early_stopping_min_epoch}"
  TRAIN.EARLY_STOPPING_PATIENCE "${early_stopping_patience}"
)

if [[ ! -s "${best}" || ! -s "${final}" ]]; then
  resume=()
  if [[ -s "${checkpoint}" ]]; then
    resume=(--resume)
  else
    resume=(--fine-tune)
  fi
  torchrun --standalone --nproc_per_node=4 \
    tools/train_france.py --cfg "${cfg}" --n-fold "${cli_fold}" "${resume[@]}" \
    --world-size 4 --seed 42 --workers 8 "${common[@]}"
fi

test -s "${best}"
test -s "${final}"
if [[ ! -s "${test_result}" ]]; then
  torchrun --standalone --nproc_per_node=4 \
    tools/test.py --cfg "${cfg}" --model "${best}" \
    --save-name "panoptic_test_fold${test_fold}_best" --n-fold "${cli_fold}" \
    --world-size 4 --seed 42 --workers 8 "${common[@]}"
fi

test -s "${test_result}"
echo "[SPARXO v2u4 + PaPs] PASTIS Fold${test_fold} training and test completed"
