#!/bin/bash
# Train the track-conditioned dual-view world model (track_dualview).
#
# Usage (from any directory):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash third_party/track_dualview/scripts/train.sh [CONFIG.yaml]
#
# Default config: third_party/track_dualview/config/train_track_realworld_dualview_192.yaml
# (per-view 192x192, stacked 384x192, initialised from hybrid-checkpoint-128000).
# Relative paths inside the config (checkpoints/, data/, outputs/) resolve against the repo root.
# NOTE: train_csvd.py deletes `output_dir` at start-up unless resume_from_checkpoint == "latest";
#       keep output_dir pointed at a dedicated run folder.

set -e

TD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$TD_ROOT/../.." && pwd)"
CONFIG="${1:-$TD_ROOT/config/train_track_realworld_dualview_192.yaml}"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

export PYTHONPATH="$TD_ROOT:$PYTHONPATH"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=1
else
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

echo "=========================================="
echo "Track-Conditioned Dual-View World Model Training"
echo "  Config: $CONFIG"
echo "  GPUs:   $NUM_GPUS"
echo "=========================================="

cd "$REPO_ROOT"

if [ "$NUM_GPUS" -gt 1 ]; then
    accelerate launch \
        --num_processes=$NUM_GPUS \
        --multi_gpu \
        --mixed_precision=fp16 \
        "$TD_ROOT/train_code/train_csvd.py" \
        --config "$CONFIG"
else
    accelerate launch \
        --num_processes=1 \
        --mixed_precision=fp16 \
        "$TD_ROOT/train_code/train_csvd.py" \
        --config "$CONFIG"
fi
