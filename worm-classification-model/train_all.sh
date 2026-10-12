#!/bin/bash

# Define the datasets and model families you want to benchmark
DATASETS=("2class_raw")
MODELS=("convnext", "efficientnetv2", "resnet", "swin", "vit", "yolo")
EXPERIMENT="model_benchmark_v2"

echo "Starting batch training pipeline..."
echo "Datasets scheduled: ${DATASETS[*]}"
echo "Models scheduled: ${MODELS[*]}"
echo ""

# Loop through each dataset
for ds in "${DATASETS[@]}"; do
    # Loop through each model
    for model in "${MODELS[@]}"; do
        echo "=========================================================="
        echo " [START] Dataset: $ds | Model: $model "
        echo "=========================================================="
        
        # Execute the training script
        ./run_train.sh "$ds" "$model" "$EXPERIMENT"
        
        # Check if the training script crashed or was manually interrupted (Ctrl+C)
        if [ $? -ne 0 ]; then
            echo " [ERROR] Training failed or was interrupted during $model on $ds."
            echo " Aborting train_all.sh to prevent cascading failures."
            exit 1
        fi
        
        echo " [DONE] Finished $model on $ds"
        echo ""
    done
done

echo "All scheduled training jobs completed successfully!"