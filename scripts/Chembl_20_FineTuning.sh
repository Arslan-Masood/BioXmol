#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-89     # Modified: 6 fractions × 1 seed × 3 splits = 18 jobs
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/Downstream/Chembl_20_FineTuning_%a.out

SAVE_DIR=$1
CONFIG_FILE=$2
BASE_MODEL_PATH=$3
CONDA_ENV=$4

LR=5.0e-05
WORKERS=8
BATCH_SIZE=128

# Define arrays for Model Type, FRAC, SEED
MODEL_TYPES=(
    "Soft_Clip_with_Frozen_Teacher"
    "Soft_Clip_with_Teacher"
    "Soft_Clip_with_Teacher_with_centering"
    "Vanilla_Clip_with_VAE"
    "Vanilla_Clip_without_VAE"
)
FRAC_ARRAY=(100 50 25 10 5 1)
SEED_ARRAY=(0 1 2)

# Get array sizes dynamically
NUM_MODEL_TYPES=${#MODEL_TYPES[@]}
NUM_FRACS=${#FRAC_ARRAY[@]}
NUM_SEEDS=${#SEED_ARRAY[@]}

# Calculate total combinations for bounds checking
TOTAL_COMBOS=$((NUM_MODEL_TYPES * NUM_FRACS * NUM_SEEDS))

# Robust nested indexing (works for any array sizes)
# Structure: FRAC (outermost) -> MODEL_TYPE -> SEED (innermost)
# This groups all jobs with the same fraction together (e.g., all FRAC=100 jobs first)
REMAINING=$SLURM_ARRAY_TASK_ID

# Extract indices from innermost to outermost
SEED_INDEX=$((REMAINING % NUM_SEEDS))
REMAINING=$((REMAINING / NUM_SEEDS))

MODEL_TYPE_INDEX=$((REMAINING % NUM_MODEL_TYPES))
REMAINING=$((REMAINING / NUM_MODEL_TYPES))

FRAC_INDEX=$((REMAINING % NUM_FRACS))

MODEL_TYPE=${MODEL_TYPES[$MODEL_TYPE_INDEX]}
FRAC=${FRAC_ARRAY[$FRAC_INDEX]}
SEED=${SEED_ARRAY[$SEED_INDEX]}

LOGGER_NAME="Chembl_20_FineTuning_ModelType_${MODEL_TYPE}_frac${FRAC}_seed${SEED}_split${SEED}_LR_${LR}"

LOGGER_PROJECT="arslan-masood/Chembl-20-FineTuning"

# Set Neptune API token
export HYDRA_FULL_ERROR=1
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ODlkMzRkMC1jYmM1LTQ5MjctOTBiMi1hYWQxNDg0ZGIxODMifQ=="
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Running baseline job for MODEL_TYPE=${MODEL_TYPE}, FRAC=${FRAC}, SEED=${SEED}, SPLIT=${SEED}"
echo "SAVE_DIR: ${SAVE_DIR}"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "LR: ${LR}"
echo "WORKERS: ${WORKERS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"

echo 'Starting script execution...'
module load mamba
echo "Activating conda environment: ${CONDA_ENV}"
source activate "${CONDA_ENV}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

# Create run directory
RUN_DIR="${SAVE_DIR}/${MODEL_TYPE}_frac${FRAC}_seed${SEED}_split${SEED}_LR_${LR}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

# -----------------------------------------------------------------------------
# Step 1: Resolve checkpoint path and extract molecular encoder
# -----------------------------------------------------------------------------
# If CHECKPOINT is PLACEHOLDER, resolve it based on model type and pretrained seed
PRETRAINED_CHECKPOINT="$BASE_MODEL_PATH/$MODEL_TYPE/Vanilla_Clip_seed_${SEED}_split_${SEED}_LR_0.001_batch_256_N_GPUS_4_WORKERS_4/checkpoints/last.ckpt"

EXTRACTED_ENCODER_DIR="${RUN_DIR}/extracted_molecular_encoders"
mkdir -p "$EXTRACTED_ENCODER_DIR"
EXTRACTED_ENCODER_PATH="$EXTRACTED_ENCODER_DIR/molecular_encoder_seed_${SEED}.ckpt"
LOCK_FILE="$EXTRACTED_ENCODER_DIR/.extraction_lock_seed_${SEED}"

# Use file lock to prevent concurrent extraction
if [ ! -f "$EXTRACTED_ENCODER_PATH" ]; then
  echo "Step: Extracting molecular encoder from checkpoint..."
  echo "Source checkpoint: $PRETRAINED_CHECKPOINT"
  
  # Wait for lock with timeout (max 10 minutes)
  LOCK_TIMEOUT=600
  LOCK_WAIT=0
  while [ -f "$LOCK_FILE" ] && [ $LOCK_WAIT -lt $LOCK_TIMEOUT ]; do
    echo "Waiting for extraction lock... (${LOCK_WAIT}s/${LOCK_TIMEOUT}s)"
    sleep 10
    LOCK_WAIT=$((LOCK_WAIT + 10))
  done
  
  if [ -f "$LOCK_FILE" ]; then
    echo "Error: Extraction lock timeout. Another process may be stuck."; exit 1
  fi
  
  # Create lock file
  echo "$$" > "$LOCK_FILE"
  
  # Double-check if file was created by another process while waiting
  if [ -f "$EXTRACTED_ENCODER_PATH" ]; then
    echo "Molecular encoder was created by another process while waiting."
    rm -f "$LOCK_FILE"
  else
    echo "Proceeding with molecular encoder extraction..."
    # First remapping: encoder_a -> model
    python bin/remap_state_dict.py \
      -i "$PRETRAINED_CHECKPOINT" \
      -o "$EXTRACTED_ENCODER_PATH" \
      --map_from "encoder_a" \
      --map_to "model"
    if [ $? -ne 0 ]; then
      echo "Error: Failed to remap encoder_a to model."; rm -f "$LOCK_FILE"; exit 1
    fi
    # Second remapping: model.fc_layers.1 -> model.fc_layers.1.0
    python bin/remap_state_dict.py \
      -i "$EXTRACTED_ENCODER_PATH" \
      -o "$EXTRACTED_ENCODER_PATH" \
      --map_from "model.fc_layers.1" \
      --map_to "model.fc_layers.1.0"
    if [ $? -ne 0 ]; then
      echo "Error: Failed to remap final layer structure."; rm -f "$LOCK_FILE"; exit 1
    fi
    echo "Molecular encoder extracted successfully: $EXTRACTED_ENCODER_PATH"
    # Remove lock file
    rm -f "$LOCK_FILE"
  fi
else
  echo "Using existing extracted molecular encoder: $EXTRACTED_ENCODER_PATH"
fi



echo 'Running training script...'
srun python /scratch/work/masooda1/Multi_Modal_Contrastive/bin/train.py -cn ${CONFIG_FILE} \
                    seed=${SEED} \
                    dataloaders.splits.train=data/chembl20/filtered_splits/chembl20-frac${FRAC}-split${SEED}-train.csv \
                    dataloaders.splits.val=data/chembl20/filtered_splits/chembl20-split${SEED}-val.csv \
                    dataloaders.splits.test=data/chembl20/filtered_splits/chembl20-split${SEED}-test.csv \
                    model._args_.0=${EXTRACTED_ENCODER_PATH} \
                    optimizer.lr=${LR} \
                    dataloaders.num_workers=${WORKERS} \
                    dataloaders.batch_size=${BATCH_SIZE} \
                    trainer.callbacks.1.dirpath=${CHECKPOINT_DIR} \
                    trainer.logger.project=\"${LOGGER_PROJECT}\" \
                    trainer.logger.name=\"${LOGGER_NAME}\" \
                    trainer.logger.api_key=\"${NEPTUNE_API_TOKEN}\"
echo 'Chembl_20_FineTuning completed successfully.'

# -----------------------------------------------------------------------------
# Testing: evaluate best checkpoint on test split
# -----------------------------------------------------------------------------
BEST_CKPT_PATTERN="${CHECKPOINT_DIR}/best-epoch-*.ckpt"
BEST_CKPT_FILES=($(ls ${BEST_CKPT_PATTERN} 2>/dev/null || true))
if [ ${#BEST_CKPT_FILES[@]} -gt 0 ]; then
  BEST_CKPT_PATH="${BEST_CKPT_FILES[0]}"
else
  if [ -f "${CHECKPOINT_DIR}/last.ckpt" ]; then
    BEST_CKPT_PATH="${CHECKPOINT_DIR}/last.ckpt"
  else
    echo "Error: No checkpoint found in ${CHECKPOINT_DIR} for testing." >&2
    exit 1
  fi
fi

echo "Using checkpoint for testing: ${BEST_CKPT_PATH}"

TEST_RESULTS_DIR="${RUN_DIR}/test_results"
mkdir -p "${TEST_RESULTS_DIR}"
TEST_RESULTS_FILENAME="Chembl20_${MODEL_TYPE}_frac_${FRAC}_seed_${SEED}_split_${SEED}_lr_${LR}"

echo 'Running testing script...'
srun python /scratch/work/masooda1/Multi_Modal_Contrastive/bin/test.py -cn ${CONFIG_FILE} \
  seed=${SEED} \
  dataloaders.splits.train=data/chembl20/filtered_splits/chembl20-frac${FRAC}-split${SEED}-train.csv \
  dataloaders.splits.val=data/chembl20/filtered_splits/chembl20-split${SEED}-val.csv \
  dataloaders.splits.test=data/chembl20/filtered_splits/chembl20-split${SEED}-test.csv \
  dataloaders.num_workers=${WORKERS} \
  dataloaders.batch_size=${BATCH_SIZE} \
  trainer.logger.project=\"${LOGGER_PROJECT}\" \
  trainer.logger.name=\"${LOGGER_NAME}_test\" \
  test_model.checkpoint_path=\"${BEST_CKPT_PATH}\" \
  test_model_ckpt=\"${BEST_CKPT_PATH}\" \
  test_results_dir=\"${TEST_RESULTS_DIR}\" \
  test_results_filename=\"${TEST_RESULTS_FILENAME}\"

echo "Testing completed. Results at: ${TEST_RESULTS_DIR}/${TEST_RESULTS_FILENAME}*"