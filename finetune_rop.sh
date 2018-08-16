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

LOG="log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

cat finetune_rop.py

CUDA_VISIBLE_DEVICES=${GPU_ID} python finetune_rop.py
