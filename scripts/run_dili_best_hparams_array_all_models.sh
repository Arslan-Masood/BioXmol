#!/bin/bash
# Script to run DILI training with best hyperparameters for all pretrained models
# Usage: bash run_dili_best_hparams_array_all_models.sh <splits_dir> <config> <save_dir> <base_model_path> <best_hp_csv>

SPLITS_DIR=$1
BASE_CONFIG=$2
SAVE_DIR=$3
BASE_MODEL_PATH=$4
BEST_HP_CSV=$5

echo "=========================================="
echo "Running DILI Training with Best Hyperparameters for All Models"
echo "=========================================="
echo "SPLITS_DIR: $SPLITS_DIR"
echo "BASE_CONFIG: $BASE_CONFIG"
echo "SAVE_DIR: $SAVE_DIR"
echo "BASE_MODEL_PATH: $BASE_MODEL_PATH"
echo "BEST_HP_CSV: $BEST_HP_CSV"
echo "=========================================="

# Validate inputs
if [ ! -f "$BEST_HP_CSV" ]; then
  echo "Error: BEST_HP_CSV not found: $BEST_HP_CSV"
  exit 1
fi

# List of all pretrained model types (same as CV array)
MODEL_TYPES=(
    "Soft_Clip_with_Frozen_Teacher"
    "Soft_Clip_with_Teacher"
    "Soft_Clip_with_Teacher_with_centering"
    "Vanilla_Clip_with_Frozen_VAE"
    "Vanilla_Clip_with_VAE"
    "Vanilla_Clip_without_VAE"
)

# List of pretrained seeds
PRETRAINED_SEEDS=(0 1 2)

# Submit jobs for each model type
for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    echo "Processing model type: $MODEL_TYPE"
    
    # Check if model directory exists
    MODEL_DIR="$BASE_MODEL_PATH/$MODEL_TYPE"
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Warning: Model directory not found: $MODEL_DIR"
        continue
    fi
    
    # Create model-specific save directory
    MODEL_SAVE_DIR="$SAVE_DIR/$MODEL_TYPE"
    
    # Submit separate SLURM array job for each pretrained seed
    for PRETRAIN_SEED in "${PRETRAINED_SEEDS[@]}"; do
        echo "  Submitting SLURM array job for $MODEL_TYPE with pretrained seed: $PRETRAIN_SEED"
        
        # Create model-specific save directory for this pretrained seed
        MODEL_SEED_SAVE_DIR="$MODEL_SAVE_DIR/pretrained_seed_${PRETRAIN_SEED}"
        
        # Submit SLURM array job for this specific pretrained seed
        sbatch scripts/run_dili_best_hparams_array.sh \
            "$SPLITS_DIR" \
            "$BASE_CONFIG" \
            "$MODEL_SEED_SAVE_DIR" \
            "$BASE_MODEL_PATH" \
            "$PRETRAIN_SEED" \
            "$BEST_HP_CSV"
        
        echo "  Job submitted for $MODEL_TYPE with pretrained seed $PRETRAIN_SEED"
    done
    echo ""
done

echo "=========================================="
echo "All jobs submitted!"
echo "Check job status with: squeue -u \$USER"
echo "=========================================="
