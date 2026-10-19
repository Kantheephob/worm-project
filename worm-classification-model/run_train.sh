#!/bin/bash
# usage: ./run_train.sh <dataset> <model|family> [experiment] [extra train.py args...]
#   ./run_train.sh 2class_raw convnext
#   ./run_train.sh 2class_raw convnext_tiny speed_test --epochs 1

DATASET=${1:?dataset key required}      # 1st parameter required (? std error)
MODEL=${2:?model or family required}    # 2nd parameter required
EXPERIMENT=${3:-model-benchmark-v2}     # 3rd default parameter if not include
# shift first 3 parameters, hide error if send less than 3 parameters or shift $# (total parameters)
shift 3 2>/dev/null || shift $#
EXTRA_ARGS=("$@")  # remaining parameters (array keeps arguments with spaces intact)

# env setup
source /mnt/data/worm/env/segmentation/bin/activate

# move to script directory FIRST, so logs/ is created next to train.py
cd "$(dirname "$0")" || exit 1
mkdir -p logs

export PYTHONUNBUFFERED=1   # see the output in real time

# Define the number of GPUs per node (override: NPROC_PER_NODE=1 ./run_train.sh ...)
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}

# CPU threads per training process. DataLoader workers (config.NUM_WORKERS per GPU) do the
# image work; the main process needs few threads. Raise this (e.g. 8) if you force --cpu-offload on,
# because then AdamW runs on the CPU.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

# Create a timestamp to replace Slurm's %j job ID for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/worm-train_${MODEL}_${TIMESTAMP}.log"

echo "Starting training with torchrun across $NPROC_PER_NODE GPUs (OMP_NUM_THREADS=$OMP_NUM_THREADS)..."
echo "Logging output to: $LOG_FILE"

# Run with torchrun, routing output to both the terminal screen and the log file
torchrun --nproc_per_node="$NPROC_PER_NODE" \
    train.py -ds "$DATASET" -m "$MODEL" -e "$EXPERIMENT" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

# exit with torchrun's status, not tee's (tee almost always returns 0)
STATUS=${PIPESTATUS[0]}
exit "$STATUS"