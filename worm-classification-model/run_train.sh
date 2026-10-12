#!/bin/bash

DATASET=${1:?dataset key required}      # 1st parameter required
MODEL=${2:?model or family required}    # 2nd parameter required
EXPERIMENT=${3:-model_benchmark_v2}     # 3rd default parameter if not include
# shift first 3 parameters, hide error if send less than 3 parameters or shift $# (total parameters)
shift 3 2>/dev/null || shift $#
EXTRA_ARGS="$@" # remaining parameters

# env setup
source /mnt/data/worm/env/segmentation/bin/activate

export OMP_NUM_THREADS=8    # set number of threads for parallel processing
export PYTHONUNBUFFERED=1   # set python output to be unbuffered, so that we can see the output in real time

mkdir -p logs

# move to script directory
cd "$(dirname "$0")" # $0 store sctipt path and name, dirname get the directory path of the script

# Create a timestamp to replace Slurm's %j job ID for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/worm-train_${TIMESTAMP}.log"

echo "Starting training..."
echo "Logging output to: $LOG_FILE"

# Run Python, routing output to both the terminal screen and the log file
python train.py -ds "$DATASET" -m "$MODEL" -e "$EXPERIMENT" $EXTRA_ARGS 2>&1 | tee "$LOG_FILE"