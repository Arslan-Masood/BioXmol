#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-17    # Run all 36 combinations (3 arch × 3 norm × 4 LR)
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/vae_hp_search_%a.out

# Set up environment
WORK_DIR="/scratch/work/masooda1/Multi_Modal_Contrastive"
DATA_PATH="/scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/cell_fetures_with_smiles_2000.parquet"
VENV_PATH="/scratch/work/masooda1/.conda_envs/mocop"
LOG_DIR="/scratch/work/masooda1/Multi_Modal_Contrastive/vae_logs"
BASE_CONFIG="/scratch/work/masooda1/Multi_Modal_Contrastive/configs/train_ae_config.yaml"

# Control model weight saving for the entire sweep
SAVE_MODEL_WEIGHTS=false  # Set to true to save model weights, false for JSON-only results

# Define hyperparameter arrays (from vae_sweep.yaml with some discretized)
MODEL_TYPES=("ae" "vae")
ARCHITECTURES=("vanilla" "medium" "large")
NORM_TYPES=("batchnorm" "layernorm" "none")
BETA_VALUES=(0.1 0.5 1.0 5.0 10.0)  
#LR_VALUES=(0.0001 0.001 0.005 0.01)  
LR_VALUES=(0.00001 0.00005)  

LATENT_DIM=128  # Fixed
DROPOUT_VALUES=(0.0)  # Fixed to single value
WD_VALUES=(0.0)  # Fixed to single value
BATCH_SIZE=64  # Fixed

# Build all combinations with single loop structure
COMBINATIONS=()

for model in "${MODEL_TYPES[@]}"; do
  for arch in "${ARCHITECTURES[@]}"; do
    for dropout in "${DROPOUT_VALUES[@]}"; do
      for norm in "${NORM_TYPES[@]}"; do
        for lr in "${LR_VALUES[@]}"; do
          for wd in "${WD_VALUES[@]}"; do
            if [ "$model" == "vae" ]; then
              # VAE: include beta variations
              for beta in "${BETA_VALUES[@]}"; do
                COMBINATIONS+=("$model,$arch,$LATENT_DIM,$dropout,$norm,$beta,$lr,$wd,$BATCH_SIZE")
              done
            else
              # AE: use fixed beta=1.0 (ignored)
              COMBINATIONS+=("$model,$arch,$LATENT_DIM,$dropout,$norm,1.0,$lr,$wd,$BATCH_SIZE")
            fi
          done
        done
      done
    done
  done
done

TOTAL_COMBINATIONS=${#COMBINATIONS[@]}
echo "Total hyperparameter combinations: $TOTAL_COMBINATIONS"

# Get this job's combination
IDX=$SLURM_ARRAY_TASK_ID
if [ $IDX -ge $TOTAL_COMBINATIONS ]; then
  echo "SLURM_ARRAY_TASK_ID $IDX exceeds total combinations $TOTAL_COMBINATIONS"
  exit 1
fi

IFS=',' read MODEL_TYPE ARCHITECTURE LATENT_DIM DROPOUT NORM_TYPE BETA LEARNING_RATE WEIGHT_DECAY BATCH_SIZE <<< "${COMBINATIONS[$IDX]}"

# Create experiment name to check if results already exist
EXP_NAME="vae_hp_${MODEL_TYPE}_${ARCHITECTURE}_lat${LATENT_DIM}_dr${DROPOUT}_${NORM_TYPE}_b${BETA}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}_bs${BATCH_SIZE}_job${SLURM_ARRAY_TASK_ID}"
RESULTS_FILE="$LOG_DIR/$EXP_NAME/results/final_results.json"

# Check if this experiment already completed successfully
if [ -f "$RESULTS_FILE" ]; then
    echo "Experiment $EXP_NAME already completed. Skipping..."
    echo "Results found at: $RESULTS_FILE"
    exit 0
fi

echo "Running experiment: $EXP_NAME (results not found, proceeding...)"

echo "Hyperparameter search job started at: $(date)"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

echo "Hyperparameters for this job:"
echo "  Model Type: $MODEL_TYPE"
echo "  Architecture: $ARCHITECTURE"
echo "  Latent Dim: $LATENT_DIM"
echo "  Dropout: $DROPOUT"
echo "  Norm Type: $NORM_TYPE"
echo "  Beta: $BETA"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Batch Size: $BATCH_SIZE"

# Change to working directory
cd $WORK_DIR

# Activate conda environment
echo "Activating conda environment: $VENV_PATH"
module load mamba
source activate "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi

# Create necessary directories
mkdir -p $(dirname $SLURM_JOB_OUTPUT)
mkdir -p $LOG_DIR

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found at $DATA_PATH"
    exit 1
fi

# Check if base config exists
if [ ! -f "$BASE_CONFIG" ]; then
    echo "Error: Base config not found at $BASE_CONFIG"
    exit 1
fi

echo "Data file found: $DATA_PATH"
echo "Base config found: $BASE_CONFIG"

# Create temporary config file
TMP_CONFIG="/tmp/vae_config_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"
cp $BASE_CONFIG $TMP_CONFIG

# Update hyperparameters in the config file using sed
sed -i "s/model_type: .*/model_type: \"$MODEL_TYPE\"/" $TMP_CONFIG
sed -i "s/architecture: .*/architecture: \"$ARCHITECTURE\"/" $TMP_CONFIG
sed -i "s/latent_dim: .*/latent_dim: $LATENT_DIM/" $TMP_CONFIG
sed -i "s/dropout: .*/dropout: $DROPOUT/" $TMP_CONFIG
sed -i "s/norm_type: .*/norm_type: \"$NORM_TYPE\"/" $TMP_CONFIG
sed -i "s/beta: .*/beta: $BETA/" $TMP_CONFIG
sed -i "s/learning_rate: .*/learning_rate: $LEARNING_RATE/" $TMP_CONFIG
sed -i "s/weight_decay: .*/weight_decay: $WEIGHT_DECAY/" $TMP_CONFIG
sed -i "s/batch_size: .*/batch_size: $BATCH_SIZE/" $TMP_CONFIG
sed -i "s/save_model_weights: .*/save_model_weights: $SAVE_MODEL_WEIGHTS/" $TMP_CONFIG

# Update experiment name to include hyperparameters
sed -i "s/experiment_name: .*/experiment_name: \"$EXP_NAME\"/" $TMP_CONFIG

# Set CUDA device if available
if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA devices available: $CUDA_VISIBLE_DEVICES"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
fi

echo "Running VAE training with config: $TMP_CONFIG"
echo "Experiment name: $EXP_NAME"
echo "Save model weights: $SAVE_MODEL_WEIGHTS"
    
    # Run training
srun python train_vae.py --config $TMP_CONFIG

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully for job $SLURM_ARRAY_TASK_ID"
else
    echo "Training failed for job $SLURM_ARRAY_TASK_ID"
fi

# Clean up
rm -f $TMP_CONFIG

echo "Hyperparameter search job completed at: $(date)" 