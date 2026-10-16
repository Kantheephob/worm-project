#!/bin/bash
# usage: ./train_all.sh [extra train.py args...]
#   ./train_all.sh
#   ./train_all.sh --patience 5

# Define the datasets and model families you want to benchmark
# (a family name trains every variant: 'swin' also includes swin_v2_*)
DATASETS=("2class_raw")
MODELS=("yolo")
# models list: 'convnext' 'efficientnet_v2' 'resnet' 'swin' 'vit' 'yolo'
# train queue: "vit" "yolo"
# untrainable models: convnext_large efficientnet_v2_l swin_v2_s swin_v2_b vit_l_16 vit_l_32 vit_h_14
EXPERIMENT="model_benchmark_v2"
EXTRA_ARGS=("$@")

cd "$(dirname "$0")" || exit 1

echo "Starting batch training pipeline..."
echo "Datasets scheduled: ${DATASETS[*]}"
echo "Models scheduled: ${MODELS[*]}"
echo "Extra args: ${EXTRA_ARGS[*]:-none}"
echo ""

# Loop through each dataset
for ds in "${DATASETS[@]}"; do
    # Loop through each model
    for model in "${MODELS[@]}"; do
        echo "=========================================================="
        echo " [START] Dataset: $ds | Model: $model "
        echo "=========================================================="

        # Execute the training script; run_train.sh now returns torchrun's real exit code
        if ! ./run_train.sh "$ds" "$model" "$EXPERIMENT" "${EXTRA_ARGS[@]}"; then
            echo " [ERROR] Training failed or was interrupted during $model on $ds."
            echo " Aborting train_all.sh to prevent cascading failures."
            echo " Rerun the same command to resume from the last checkpoint."
            exit 1
        fi

        echo " [DONE] Finished $model on $ds"
        echo ""
    done
done

echo "All scheduled training jobs completed successfully!"