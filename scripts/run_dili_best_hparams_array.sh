#!/bin/bash
#SBATCH --time=00:180:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-24    # Run for each combination (5 downstream seeds × 5 folds)
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/Downstream/DILI_Best_HP_Array_%j_%a.out

# -----------------------------------------------------------------------------
# Exports and global defaults
# -----------------------------------------------------------------------------
VENV_PATH="/scratch/work/masooda1/.conda_envs/mocop"
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ODlkMzRkMC1jYmM1LTQ5MjctOTBiMi1hYWQxNDg0ZGIxODMifQ=="
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export HYDRA_FULL_ERROR=1

# Global defaults
WORKERS=${SLURM_CPUS_PER_TASK:-4}
BATCH_SIZE=32

# Configuration arrays
DOWNSTREAM_SEEDS=(0 1 2 3 4)
FOLDS=(1 2 3 4 5)

# -----------------------------------------------------------------------------
# Positional args
# -----------------------------------------------------------------------------
SPLITS_DIR=$1
BASE_CONFIG=$2   # Hydra config name, e.g., Downstream_DILIGold_splits
SAVE_DIR=$3
BASE_MODEL_PATH=$4
PRETRAIN_SEED=$5  # Pretrained model seed (e.g., 0, 1, 2)
BEST_HP_CSV=$6   # Path to best_hyperparameters_by_val_loss.csv

# Extract model type from SAVE_DIR path (handle pretrained_seed_X subdirectory)
if [[ "$SAVE_DIR" == *"/pretrained_seed_"* ]]; then
  MODEL_TYPE=$(basename "$(dirname "$SAVE_DIR")")
else
  MODEL_TYPE=$(basename "$SAVE_DIR")
fi

# Validate required arguments
if [ -z "$PRETRAIN_SEED" ]; then
  echo "Error: PRETRAIN_SEED is required as 5th argument"; exit 1
fi
if [ -z "$BEST_HP_CSV" ]; then
  echo "Error: BEST_HP_CSV is required as 6th argument"; exit 1
fi

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
echo "=========================================="
echo "DILI Best HP Training (Downstream×Fold)"
echo "=========================================="
echo "SPLITS_DIR: $SPLITS_DIR"
echo "CONFIG_NAME: $BASE_CONFIG"
echo "SAVE_DIR: $SAVE_DIR"
echo "MODEL_TYPE: $MODEL_TYPE"
echo "BASE_MODEL_PATH: $BASE_MODEL_PATH"
echo "PRETRAIN_SEED: $PRETRAIN_SEED"
echo "BEST_HP_CSV: $BEST_HP_CSV"
echo "ARRAY_ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "=========================================="

# -----------------------------------------------------------------------------
# Validate inputs
# -----------------------------------------------------------------------------
if [ ! -d "$SPLITS_DIR" ]; then
  echo "Error: SPLITS_DIR not found: $SPLITS_DIR"; exit 1
fi
if [ ! -f "$BEST_HP_CSV" ]; then
  echo "Error: BEST_HP_CSV not found: $BEST_HP_CSV"; exit 1
fi

mkdir -p "$SAVE_DIR"

# -----------------------------------------------------------------------------
# Step 1: Build combinations and select current job
# -----------------------------------------------------------------------------
# Build combinations (downstream_seed, fold)
COMBINATIONS=()
for downstream_seed in "${DOWNSTREAM_SEEDS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    COMBINATIONS+=("${downstream_seed},${fold}")
  done
done

IDX=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=${#COMBINATIONS[@]}
if [ "$IDX" -ge "$TOTAL" ]; then
  echo "SLURM_ARRAY_TASK_ID $IDX exceeds total combinations $TOTAL"; exit 1
fi

IFS=',' read DOWNSTREAM_SEED FOLD <<< "${COMBINATIONS[$IDX]}"

echo "Job selection => downstream_seed=$DOWNSTREAM_SEED | fold=$FOLD"

# -----------------------------------------------------------------------------
# Step 2: Get best hyperparameters from CSV
# -----------------------------------------------------------------------------
# Find the row in CSV that matches our combination
CSV_LINE=$(grep "^${MODEL_TYPE},${PRETRAIN_SEED},${FOLD},${DOWNSTREAM_SEED}," "$BEST_HP_CSV")
if [ -z "$CSV_LINE" ]; then
  echo "Error: No matching row found in CSV for model_type=$MODEL_TYPE, pretrained_seed=$PRETRAIN_SEED, fold=$FOLD, downstream_seed=$DOWNSTREAM_SEED"; exit 1
fi

# Parse CSV line to get best hyperparameters using column names
# Read header to get column positions
HEADER=$(head -n 1 "$BEST_HP_CSV")
IFS=',' read -ra HEADER_COLS <<< "$HEADER"

# Find column indices
LR_COL=$(for i in "${!HEADER_COLS[@]}"; do [[ "${HEADER_COLS[$i]}" == "LR" ]] && echo $i; done)
WEIGHT_DECAY_COL=$(for i in "${!HEADER_COLS[@]}"; do [[ "${HEADER_COLS[$i]}" == "weight_decay" ]] && echo $i; done)
BEST_EPOCH_COL=$(for i in "${!HEADER_COLS[@]}"; do [[ "${HEADER_COLS[$i]}" == "best_epoch" ]] && echo $i; done)

# Parse the CSV line using column indices
IFS=',' read -ra CSV_FIELDS <<< "$CSV_LINE"
LR="${CSV_FIELDS[$LR_COL]}"
WEIGHT_DECAY="${CSV_FIELDS[$WEIGHT_DECAY_COL]}"
BEST_EPOCH="${CSV_FIELDS[$BEST_EPOCH_COL]}"

echo "Best hyperparameters => lr=$LR | wd=$WEIGHT_DECAY | best_epoch=$BEST_EPOCH"

# Enforce a minimum of 10 epochs
TRAIN_EPOCHS=$BEST_EPOCH
if [ -z "$TRAIN_EPOCHS" ] || [ "$TRAIN_EPOCHS" -lt 10 ]; then
  TRAIN_EPOCHS=10
fi
echo "Planned training epochs (min 10): $TRAIN_EPOCHS"

# -----------------------------------------------------------------------------
# Step 3: Setup paths and check if result already exists
# -----------------------------------------------------------------------------
# Complete train set and heldout test set paths
TRAIN_CSV="$SPLITS_DIR/train_fold_seed_${DOWNSTREAM_SEED}_fold_${FOLD}.csv"
TEST_CSV="$SPLITS_DIR/test_seed_${DOWNSTREAM_SEED}_fold_${FOLD}.csv"

echo "TRAIN_CSV: $TRAIN_CSV"
echo "TEST_CSV: $TEST_CSV"

# Validate split files exist
for f in "$TRAIN_CSV" "$TEST_CSV"; do
  if [ ! -f "$f" ]; then
    echo "Error: split file not found: $f"; exit 1
  fi
done

# Create run-specific directory
RUN_DIR="$SAVE_DIR/best_hp_fold_${FOLD}_downstream_seed_${DOWNSTREAM_SEED}"
mkdir -p "$RUN_DIR"

# Setup test results path (centralized location)
BASE_SAVE_DIR=$(dirname "$(dirname "$SAVE_DIR")")
TEST_RESULTS_DIR="$BASE_SAVE_DIR/best_hp_test_results"
TEST_RESULTS_FILENAME="DILI_${MODEL_TYPE}_pretrained_seed_${PRETRAIN_SEED}_downstream_seed_${DOWNSTREAM_SEED}_fold_${FOLD}_best_hp"
RESULTS_JSON="${TEST_RESULTS_DIR}/${TEST_RESULTS_FILENAME}.json"

# Check if result file already exists
if [ -f "$RESULTS_JSON" ]; then
  echo "=========================================="
  echo "Result file already exists: $RESULTS_JSON"
  echo "Skipping everything for this combination."
  echo "model_type=$MODEL_TYPE | pretrained_seed=$PRETRAIN_SEED | fold=$FOLD | downstream_seed=$DOWNSTREAM_SEED"
  echo "=========================================="
  exit 0
fi

echo "Result file not found, proceeding with training and testing..."

# -----------------------------------------------------------------------------
# Step 4: Setup encoder paths
# -----------------------------------------------------------------------------
# Use existing extracted molecular encoder from previous CV runs
# The extracted encoders are in the CV runs directory, not the new best HP runs directory
EXTRACTED_ENCODER_DIR="$BASE_MODEL_PATH/$MODEL_TYPE/pretrained_seed_${PRETRAIN_SEED}/extracted_molecular_encoders"
EXTRACTED_ENCODER_PATH="$EXTRACTED_ENCODER_DIR/molecular_encoder_seed_${PRETRAIN_SEED}.ckpt"


# Check if extracted encoder exists
if [ ! -f "$EXTRACTED_ENCODER_PATH" ]; then
  echo "Error: Extracted molecular encoder not found: $EXTRACTED_ENCODER_PATH"
  echo "Please run the CV array jobs first to extract the encoders."
  exit 1
fi

echo "Using existing extracted molecular encoder: $EXTRACTED_ENCODER_PATH"

# Activate conda environment
echo "Activating conda environment: $VENV_PATH"
module load mamba
source activate "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 5: Training with best hyperparameters
# -----------------------------------------------------------------------------
echo "Step 5: Training with best hyperparameters..."
echo "Training on complete train set: $TRAIN_CSV"
echo "Learning rate: $LR"
echo "Weight decay: $WEIGHT_DECAY"
echo "Best epoch from CV: $BEST_EPOCH"

# Setup checkpoint directory and logger
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"
LOGGER_PROJECT="arslan-masood/DILI-Gold-BestHP"
LOGGER_NAME="DILI_${MODEL_TYPE}_pretrain_seed_${PRETRAIN_SEED}_downstream_seed_${DOWNSTREAM_SEED}_fold_${FOLD}_best_hp"

echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Logger project: $LOGGER_PROJECT"
echo "Logger name: $LOGGER_NAME"

# Train on complete train set
srun python bin/train.py -cn ${BASE_CONFIG} \
  dataloaders.splits.train="${TRAIN_CSV}" \
  dataloaders.splits.val="${TEST_CSV}" \
  model._args_.0="${EXTRACTED_ENCODER_PATH}" \
  seed=${DOWNSTREAM_SEED} \
  dataloaders.seed=${DOWNSTREAM_SEED} \
  dataloaders.num_workers=${WORKERS} \
  dataloaders.batch_size=${BATCH_SIZE} \
  optimizer.lr=${LR} \
  optimizer.weight_decay=${WEIGHT_DECAY} \
  trainer.min_epochs=10 \
  trainer.max_epochs=${TRAIN_EPOCHS} \
  trainer.callbacks.1.dirpath="${CHECKPOINT_DIR}" \
  trainer.logger.project="${LOGGER_PROJECT}" \
  trainer.logger.name="${LOGGER_NAME}"

TRAIN_STATUS=$?

if [ $TRAIN_STATUS -ne 0 ]; then
  echo "==========================================" >&2
  echo "Training failed" >&2
  echo "model_type=$MODEL_TYPE | pretrained_seed=$PRETRAIN_SEED | fold=$FOLD | downstream_seed=$DOWNSTREAM_SEED" >&2
  echo "Run dir: $RUN_DIR" >&2
  echo "==========================================" >&2
  exit $TRAIN_STATUS
fi

echo "Training completed successfully!"

# -----------------------------------------------------------------------------
# Step 6: Testing on heldout test set
# -----------------------------------------------------------------------------
echo "Step 6: Testing on heldout test set..."

# Find last checkpoint (final trained model)
LAST_CKPT_PATTERN="${CHECKPOINT_DIR}/last.ckpt"
if [ ! -f "$LAST_CKPT_PATTERN" ]; then
  echo "Error: No last checkpoint found in ${CHECKPOINT_DIR}" >&2
  exit 1
fi

LAST_CKPT_PATH="$LAST_CKPT_PATTERN"
echo "Using last checkpoint for testing: ${LAST_CKPT_PATH}"
echo "Testing on final trained model (last epoch)"

# Setup test results directory
mkdir -p "$TEST_RESULTS_DIR"

echo "Test results dir: $TEST_RESULTS_DIR"
echo "Test results filename: $TEST_RESULTS_FILENAME"

# Run testing on heldout test set
srun python bin/test.py -cn ${BASE_CONFIG} \
  seed=${DOWNSTREAM_SEED} \
  dataloaders.seed=${DOWNSTREAM_SEED} \
  dataloaders.num_workers=${WORKERS} \
  dataloaders.batch_size=${BATCH_SIZE} \
  dataloaders.splits.train="${TRAIN_CSV}" \
  dataloaders.splits.val="${TEST_CSV}" \
  trainer.logger.project="${LOGGER_PROJECT}" \
  trainer.logger.name="${LOGGER_NAME}_test" \
  test_model.checkpoint_path="${LAST_CKPT_PATH}" \
  test_model_ckpt="${LAST_CKPT_PATH}" \
  test_results_dir="${TEST_RESULTS_DIR}" \
  test_results_filename="${TEST_RESULTS_FILENAME}"

TEST_STATUS=$?

if [ $TEST_STATUS -eq 0 ]; then
  echo "=========================================="
  echo "Job completed successfully"
  echo "model_type=$MODEL_TYPE | pretrained_seed=$PRETRAIN_SEED | fold=$FOLD | downstream_seed=$DOWNSTREAM_SEED"
  echo "Run dir: $RUN_DIR"
  echo "Checkpoints: $CHECKPOINT_DIR"
  echo "Test results: ${TEST_RESULTS_DIR}/${TEST_RESULTS_FILENAME}"
  echo "=========================================="
  
  # Annotate results JSON with training epochs and hyperparameters
  if [ -f "$RESULTS_JSON" ]; then
    python -c "
import json, sys
p, e, lr, wd = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
d = json.load(open(p))
d['training_epochs'] = int(e) if e.isdigit() else e
d['best_lr'] = float(lr)
d['best_weight_decay'] = float(wd)
json.dump(d, open(p,'w'), indent=2)
" "$RESULTS_JSON" "$BEST_EPOCH" "$LR" "$WEIGHT_DECAY"
    echo "Annotated results with training_epochs=${BEST_EPOCH}, lr=${LR}, wd=${WEIGHT_DECAY}."
  else
    echo "Warning: Results JSON not found to annotate: $RESULTS_JSON" >&2
  fi
else
  echo "==========================================" >&2
  echo "Testing failed" >&2
  echo "model_type=$MODEL_TYPE | pretrained_seed=$PRETRAIN_SEED | fold=$FOLD | downstream_seed=$DOWNSTREAM_SEED" >&2
  echo "Run dir: $RUN_DIR" >&2
  echo "==========================================" >&2
fi

# Clean up checkpoint directory to save space
#rm -rf "$CHECKPOINT_DIR"

exit $TEST_STATUS
