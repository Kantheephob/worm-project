#!/bin/bash
#SBATCH -p gpu                      # LANTA GPU partition
#SBATCH -N 1
#SBATCH --ntasks-per-node=1         # 1 task; python spawns 1 process per GPU itself
#SBATCH --gpus-per-node=4
#SBATCH -c 64                       # CPU cores (NUM_WORKERS=8 per GPU x 4 GPUs + headroom)
#SBATCH -t 48:00:00
#SBATCH -A lt200264
#SBATCH -J worm-train
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# usage:
#   sbatch run_train.sh <dataset> <model_or_family> [experiment] [extra args...]
#   sbatch run_train.sh 2class_blackout resnet
#   sbatch run_train.sh 5class_raw vit model_benchmark_v2
#   sbatch run_train.sh 2class_blackout resnet18 smoke_test --epochs 2

DATASET=${1:?dataset key required}
MODEL=${2:?model or family required}
EXPERIMENT=${3:-model_benchmark_v2}
shift 3 2>/dev/null || shift $#
EXTRA_ARGS="$@"

# ---- environment: EDIT these two lines to match your setup ----
module load Mamba/23.11.0-0
conda activate worm          # <- your conda env name

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

python train.py -ds "$DATASET" -m "$MODEL" -e "$EXPERIMENT" $EXTRA_ARGS