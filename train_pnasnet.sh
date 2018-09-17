#/bin/bash
# Usage:
# ./scripts/finetune_official.sh GPU
#
# Example:
# ./scripts/finetune_official.sh 1

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1

LOG="logs/pnasnet.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cat train_pnasnet.py

CUDA_VISIBLE_DEVICES=${GPU_ID} python train_pnasnet.py
