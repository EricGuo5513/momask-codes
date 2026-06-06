#!/bin/bash
set -e

VIDEO_NAME="${1:?usage: bash run_bridge.sh <video_name> <experiment_name> \"<prompt>\" [mask_start] [mask_end]}"
EXP_NAME="${2:?missing experiment_name}"
PROMPT="${3:?missing prompt}"
MASK_START="${4:-0.4}"
MASK_END="${5:-0.7}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_VIDEO="${PROJECT_ROOT}/video_bridge/inputs/${VIDEO_NAME}.mp4"
EXTRACTED_NPY="${PROJECT_ROOT}/video_bridge/extracted/${VIDEO_NAME}.npz"

if [ ! -f "${INPUT_VIDEO}" ]; then
    echo "Error: input video not found at ${INPUT_VIDEO}"
    exit 1
fi

cd "${PROJECT_ROOT}"

if [ -f "${EXTRACTED_NPY}" ]; then
    echo "Reusing existing extraction: ${EXTRACTED_NPY}"
else
    echo "Step 1: Extracting pose from ${VIDEO_NAME}.mp4"
    python video_bridge/video_to_humanml3d.py \
        --video "${INPUT_VIDEO}" \
        --output "${EXTRACTED_NPY}" \
        --visualize
fi

echo "Step 2: Running MoMask edit_t2m"
echo "   prompt: ${PROMPT}"
echo "   mask: ${MASK_START}-${MASK_END}"

python edit_t2m.py --gpu_id -1 --ext "${EXP_NAME}" \
    --use_res_model \
    -msec "${MASK_START},${MASK_END}" \
    --source_motion "${EXTRACTED_NPY}" \
    --text_prompt "${PROMPT}"

echo ""
echo "Done. Outputs in: ${PROJECT_ROOT}/generation/${EXP_NAME}/animation/"
ls "${PROJECT_ROOT}/generation/${EXP_NAME}/animation/" 2>/dev/null || echo "(generation folder not found)"
