#!/bin/bash
# Script to run DILI CV array for all pretrained models
# Usage: bash run_dili_cv_array_all_models.sh <splits_dir> <config> <save_dir> <base_model_path> [downstream_seeds] [learning_rates] [weight_decays]

SPLITS_DIR=$1
BASE_CONFIG=$2
SAVE_DIR=$3
BASE_MODEL_PATH=$4
DOWNSTREAM_SEEDS=${5:-"0 1 2 3 4"}
LEARNING_RATES=${6:-"1e-5 5e-5 1e-4 5e-4 1e-3"}
WEIGHT_DECAYS=${7:-"0.0 0.01 0.1 1.0"}

# List of all pretrained model types
MODEL_TYPES=(
    "Soft_Clip_with_Frozen_Teacher"
    "Soft_Clip_with_Teacher"
    "Soft_Clip_with_Teacher_with_centering"
    "Vanilla_Clip_with_Frozen_VAE"
    "Vanilla_Clip_with_VAE"
    "Vanilla_Clip_without_VAE"
)

echo "=========================================="
echo "Running DILI CV Array for All Pretrained Models"
echo "=========================================="
echo "SPLITS_DIR: $SPLITS_DIR"
echo "BASE_CONFIG: $BASE_CONFIG"
echo "SAVE_DIR: $SAVE_DIR"
echo "BASE_MODEL_PATH: $BASE_MODEL_PATH"
echo "DOWNSTREAM_SEEDS: $DOWNSTREAM_SEEDS"
echo "LEARNING_RATES: $LEARNING_RATES"
echo "WEIGHT_DECAYS: $WEIGHT_DECAYS"
echo "MODEL_TYPES: ${MODEL_TYPES[*]}"
echo "=========================================="

# Submit jobs for each model type
for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    echo "Submitting jobs for model type: $MODEL_TYPE"
    
    # Check if model directory exists
    MODEL_DIR="$BASE_MODEL_PATH/$MODEL_TYPE"
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Warning: Model directory not found: $MODEL_DIR"
        continue
    fi
    
    # Find available seeds for this model type
    AVAILABLE_SEEDS=()
    for SEED_DIR in "$MODEL_DIR"/Vanilla_Clip_seed_*; do
        if [ -d "$SEED_DIR" ]; then
            # Extract seed number from directory name
            SEED=$(basename "$SEED_DIR" | sed -E 's/.*seed_([0-9]+).*/\1/')
            CHECKPOINT_PATH="$SEED_DIR/checkpoints/last.ckpt"
            if [ -f "$CHECKPOINT_PATH" ]; then
                AVAILABLE_SEEDS+=("$SEED")
                echo "  Found seed $SEED with checkpoint: $CHECKPOINT_PATH"
            else
                echo "  Warning: No last.ckpt found for seed $SEED in $SEED_DIR"
            fi
        fi
    done
    
    if [ ${#AVAILABLE_SEEDS[@]} -eq 0 ]; then
        echo "  No valid checkpoints found for $MODEL_TYPE, skipping..."
        continue
    fi
    
    # Create model-specific save directory
    MODEL_SAVE_DIR="$SAVE_DIR/$MODEL_TYPE"
    
    # Submit separate SLURM array job for each pretrained seed
    for PRETRAIN_SEED in "${AVAILABLE_SEEDS[@]}"; do
        echo "  Submitting SLURM array job for $MODEL_TYPE with pretrained seed: $PRETRAIN_SEED"
        
        # Create model-specific save directory for this pretrained seed
        MODEL_SEED_SAVE_DIR="$MODEL_SAVE_DIR/pretrained_seed_${PRETRAIN_SEED}"
        
        # Submit SLURM array job for this specific pretrained seed
        sbatch scripts/run_dili_cv_array.sh \
            "$SPLITS_DIR" \
            "$BASE_CONFIG" \
            "$MODEL_SEED_SAVE_DIR" \
            "PLACEHOLDER" \
            "$BASE_MODEL_PATH" \
            "$PRETRAIN_SEED" \
            "$DOWNSTREAM_SEEDS" \
            "$LEARNING_RATES" \
            "$WEIGHT_DECAYS"
        
        echo "  Job submitted for $MODEL_TYPE with pretrained seed $PRETRAIN_SEED"
    done
    echo ""
done

echo "=========================================="
echo "All jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "=========================================="
