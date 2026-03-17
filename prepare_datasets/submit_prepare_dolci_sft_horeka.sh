#!/usr/bin/env bash
# Submit the Dolci-Instruct preparation as a SLURM job on the debug queue.
#
# Usage:
#   bash prepare_datasets/submit_prepare_dolci_instruct.sh [--output /path/to/output] [--num-shards 8]
#
# Defaults:
#   output:     /home/hk-project-p0024002/orcid_swz4229/propella_annotation/data/Dolci-Instruct-SFT/dolci-instruct-sft-prepared
#   num-shards: 8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults
DATASET_PATH="/home/hk-project-p0024002/orcid_swz4229/propella_annotation/data/Dolci-Instruct-SFT"
OUTPUT="/home/hk-project-p0024002/orcid_swz4229/propella_annotation/data/Dolci-Instruct-SFT/dolci-instruct-sft-prepared"
NUM_SHARDS=8
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

echo "Submitting Dolci-Instruct preparation job..."
echo "  Dataset:    ${DATASET_PATH}"
echo "  Output:     ${OUTPUT}"
echo "  Num shards: ${NUM_SHARDS}"
[[ -n "${MAX_ROWS}" ]] && echo "  Max rows:   ${MAX_ROWS}"

sbatch \
    --job-name="prepare-dolci-instruct" \
    --partition="dev_cpuonly" \
    --account="hk-project-p0024002" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem="64G" \
    --time="00:10:00" \
    --output="${LOG_DIR}/prepare-dolci-instruct-%j.out" \
    --error="${LOG_DIR}/prepare-dolci-instruct-%j.err" \
    --wrap="cd ${REPO_ROOT}/inference-hive && pixi run -e cuda-sglang python ../prepare_datasets/prepare_dolci_instruct.py --dataset-path ${DATASET_PATH} --output ${OUTPUT} --num-shards ${NUM_SHARDS} ${EXTRA_ARGS}"

echo "Job submitted. Check logs in ${LOG_DIR}/"
