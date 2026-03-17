#!/usr/bin/env bash
# Submit the Dolci-Think-DPO preparation as a SLURM job.
#
# Usage:
#   bash prepare_datasets/submit_prepare_dolci_think_dpo.sh [--output /path/to/output] [--num-shards 16]
#
# Defaults:
#   output:     /leonardo_scratch/large/userexternal/knikolao/propella_annotation/data/Dolci-Think-DPO-7B/dolci-think-dpo-prepared
#   num-shards: 7

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults
DATASET_PATH="/leonardo_scratch/large/userexternal/knikolao/propella_annotation/data/Dolci-Instruct-DPO"
OUTPUT="/leonardo_scratch/large/userexternal/knikolao/propella_annotation/data/Dolci-Instruct-DPO/dolci-instruct-dpo-prepared"
NUM_SHARDS=4
MAX_ROWS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --output)       OUTPUT="$2"; shift 2 ;;
        --num-shards)   NUM_SHARDS="$2"; shift 2 ;;
        --max-rows)     MAX_ROWS="$2"; shift 2 ;;
        *)              echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Build optional args
EXTRA_ARGS=""
if [[ -n "${MAX_ROWS}" ]]; then
    EXTRA_ARGS="--max-rows ${MAX_ROWS}"
fi

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "Submitting Dolci-Think preparation job..."
echo "  Dataset:    ${DATASET_PATH}"
echo "  Output:     ${OUTPUT}"
echo "  Num shards: ${NUM_SHARDS}"
[[ -n "${MAX_ROWS}" ]] && echo "  Max rows:   ${MAX_ROWS}"

sbatch \
    --job-name="prepare-dolci-think-dpo" \
    --partition="boost_usr_prod" \
    --account="oellm_prod2026" \
    --qos="boost_qos_dbg" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem="64G" \
    --time="00:15:00" \
    --output="${LOG_DIR}/prepare-dolci-think-dpo-%j.out" \
    --error="${LOG_DIR}/prepare-dolci-think-dpo-%j.err" \
    --wrap="cd ${REPO_ROOT}/inference-hive && pixi run -e cuda-sglang python ../prepare_datasets/prepare_dolci_dpo.py --dataset-path ${DATASET_PATH} --output ${OUTPUT} --num-shards ${NUM_SHARDS} ${EXTRA_ARGS}"

echo "Job submitted. Check logs in ${LOG_DIR}/"