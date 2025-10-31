#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-499    # Run all combinations (5 downstream seeds × 5 folds × 5 LR × 4 WD = 500)
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/Downstream/DILI_CV_Array_%j_%a.out

# -----------------------------------------------------------------------------
# Exports and global defaults (moved before header)
# -----------------------------------------------------------------------------
VENV_PATH="/scratch/work/masooda1/.conda_envs/mocop"
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ODlkMzRkMC1jYmM1LTQ5MjctOTBiMi1hYWQxNDg0ZGIxODMifQ=="
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export HYDRA_FULL_ERROR=1

# Global defaults
WORKERS=${SLURM_CPUS_PER_TASK:-4}
BATCH_SIZE=32

# -----------------------------------------------------------------------------
# Positional args
# -----------------------------------------------------------------------------
SPLITS_DIR=$1
BASE_CONFIG=$2   # Hydra config name, e.g., Downstream_DILIGold_splits
SAVE_DIR=$3
CHECKPOINT=$4
BASE_MODEL_PATH=$5
PRETRAIN_SEED=$6  # Pretrained model seed (e.g., 0, 1, 2)
DOWNSTREAM_SEEDS_STR=$7  # Downstream training seeds
LR_LIST_STR=$8    # Learning rates
WD_LIST_STR=$9    # Weight decays

# Extract model type from SAVE_DIR path (handle pretrained_seed_X subdirectory)
if [[ "$SAVE_DIR" == *"/pretrained_seed_"* ]]; then
  MODEL_TYPE=$(basename "$(dirname "$SAVE_DIR")")
else
  MODEL_TYPE=$(basename "$SAVE_DIR")
fi

# Validate required arguments
if [ -z "$BASE_MODEL_PATH" ]; then
  echo "Error: BASE_MODEL_PATH is required as 5th argument"; exit 1
fi
if [ -z "$PRETRAIN_SEED" ]; then
  echo "Error: PRETRAIN_SEED is required as 6th argument"; exit 1
fi
if [ -z "$DOWNSTREAM_SEEDS_STR" ]; then
  echo "Error: DOWNSTREAM_SEEDS_STR is required as 7th argument"; exit 1
fi
if [ -z "$LR_LIST_STR" ]; then
  echo "Error: LR_LIST_STR is required as 8th argument"; exit 1
fi
if [ -z "$WD_LIST_STR" ]; then
  echo "Error: WD_LIST_STR is required as 9th argument"; exit 1
fi

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
echo "=========================================="
echo "DILI Array Training (Seed×Fold×LR×WD)"
echo "=========================================="
echo "SPLITS_DIR: $SPLITS_DIR"
echo "CONFIG_NAME: $BASE_CONFIG"
echo "SAVE_DIR: $SAVE_DIR"
echo "MODEL_TYPE: $MODEL_TYPE"
echo "BASE_MODEL_PATH: $BASE_MODEL_PATH"
echo "CHECKPOINT: $CHECKPOINT"
echo "PRETRAIN_SEED: $PRETRAIN_SEED"
echo "DOWNSTREAM_SEEDS: $DOWNSTREAM_SEEDS_STR"
echo "LR_LIST: $LR_LIST_STR"
echo "WD_LIST: $WD_LIST_STR"
echo "ARRAY_ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "=========================================="

# -----------------------------------------------------------------------------
# Validate inputs
# -----------------------------------------------------------------------------
if [ ! -d "$SPLITS_DIR" ]; then
  echo "Error: SPLITS_DIR not found: $SPLITS_DIR"; exit 1
fi
# BASE_CONFIG is a Hydra config name; no file existence check here

mkdir -p "$SAVE_DIR"

# Env already exported above

# Parse lists
read -r -a DOWNSTREAM_SEEDS <<< "$DOWNSTREAM_SEEDS_STR"
read -r -a LRS <<< "$LR_LIST_STR"
read -r -a WDS <<< "$WD_LIST_STR"

# -----------------------------------------------------------------------------
# Step 1: Build all combinations and select current job
# -----------------------------------------------------------------------------
# Build combinations (downstream_seed, fold, lr, wd)
COMBINATIONS=()
for downstream_seed in "${DOWNSTREAM_SEEDS[@]}"; do
  for fold in 1 2 3 4 5; do
    for lr in "${LRS[@]}"; do
      for wd in "${WDS[@]}"; do
        COMBINATIONS+=("${downstream_seed},${fold},${lr},${wd}")
      done
    done
  done
done

IDX=${SLURM_ARRAY_TASK_ID:-0}
TOTAL=${#COMBINATIONS[@]}
if [ "$IDX" -ge "$TOTAL" ]; then
  echo "SLURM_ARRAY_TASK_ID $IDX exceeds total combinations $TOTAL"; exit 1
fi

IFS=',' read DOWNSTREAM_SEED FOLD LR WD <<< "${COMBINATIONS[$IDX]}"

echo "Job selection => downstream_seed=$DOWNSTREAM_SEED | fold=$FOLD | lr=$LR | wd=$WD"
echo "Splits dir   => $SPLITS_DIR"
echo "Config name  => $BASE_CONFIG"

# -----------------------------------------------------------------------------
# Step 1.5: Check if result file already exists - if so, skip everything
# -----------------------------------------------------------------------------
# Setup test results path (centralized location)
BASE_SAVE_DIR=$(dirname "$(dirname "$SAVE_DIR")")
TEST_RESULTS_DIR="$BASE_SAVE_DIR/test_results"
TEST_RESULTS_FILENAME="DILI_${MODEL_TYPE}_pretrained_seed_${PRETRAIN_SEED}_downstream_seed_${DOWNSTREAM_SEED}_fold_${FOLD}_lr_${LR}_wd_${WD}"
RESULTS_JSON="${TEST_RESULTS_DIR}/${TEST_RESULTS_FILENAME}.json"

if [ -f "$RESULTS_JSON" ]; then
  echo "=========================================="
  echo "Result file already exists: $RESULTS_JSON"
  echo "Skipping everything for this combination."
  echo "downstream_seed=$DOWNSTREAM_SEED | fold=$FOLD | lr=$LR | wd=$WD"
  echo "=========================================="
  exit 0
fi

echo "Result file not found, proceeding with training and testing..."

# Activate conda environment
echo "Activating conda environment: $VENV_PATH"
module load mamba
source activate "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi
# -----------------------------------------------------------------------------
# Step 2: Resolve split file paths and create run directories
# -----------------------------------------------------------------------------
# Resolve split file paths (train on train_inner, evaluate on val)
TRAIN_INNER_CSV="$SPLITS_DIR/train_inner_fold_seed_${DOWNSTREAM_SEED}_fold_${FOLD}.csv"
VAL_CSV="$SPLITS_DIR/val_seed_${DOWNSTREAM_SEED}_fold_${FOLD}.csv"

for f in "$TRAIN_INNER_CSV" "$VAL_CSV"; do
  if [ ! -f "$f" ]; then
    echo "Error: split file not found: $f"; exit 1
  fi
done

# Create run-specific directory (flattened structure)
RUN_DIR="$SAVE_DIR/seed_${DOWNSTREAM_SEED}_fold_${FOLD}_lr_${LR}_wd_${WD}"
mkdir -p "$RUN_DIR"

# Logger and checkpoint setup (similar to Downstream_DILIGold.sh)
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"
LOGGER_PROJECT="arslan-masood/DILI-Gold"
# PRETRAIN_SEED is now passed as a separate argument
LOGGER_NAME="DILI_${MODEL_TYPE}_pretrain_seed_${PRETRAIN_SEED}_downstream_seed_${DOWNSTREAM_SEED}_fold_${FOLD}_lr_${LR}_wd_${WD}"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Logger project: $LOGGER_PROJECT"
echo "Logger name: $LOGGER_NAME"

# -----------------------------------------------------------------------------
# Step 3: Resolve checkpoint path and extract molecular encoder
# -----------------------------------------------------------------------------
# If CHECKPOINT is PLACEHOLDER, resolve it based on model type and pretrained seed
if [ "$CHECKPOINT" = "PLACEHOLDER" ]; then
  CHECKPOINT="$BASE_MODEL_PATH/$MODEL_TYPE/Vanilla_Clip_seed_${PRETRAIN_SEED}_split_${PRETRAIN_SEED}_LR_0.001_batch_256_N_GPUS_4_WORKERS_4/checkpoints/last.ckpt"
fi

# Validate checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
  echo "Error: Checkpoint not found: $CHECKPOINT"; exit 1
fi

EXTRACTED_ENCODER_DIR="$SAVE_DIR/extracted_molecular_encoders"
mkdir -p "$EXTRACTED_ENCODER_DIR"
EXTRACTED_ENCODER_PATH="$EXTRACTED_ENCODER_DIR/molecular_encoder_seed_${PRETRAIN_SEED}.ckpt"
LOCK_FILE="$EXTRACTED_ENCODER_DIR/.extraction_lock_seed_${PRETRAIN_SEED}"

# Use file lock to prevent concurrent extraction
if [ ! -f "$EXTRACTED_ENCODER_PATH" ]; then
  echo "Step: Extracting molecular encoder from checkpoint..."
  echo "Source checkpoint: $CHECKPOINT"
  
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
      -i "$CHECKPOINT" \
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

# -----------------------------------------------------------------------------
# Step 4: Check if best checkpoint exists, skip training if it does
# -----------------------------------------------------------------------------
BEST_CKPT_PATTERN="${CHECKPOINT_DIR}/best-epoch-*.ckpt"
BEST_CKPT_FILES=($(ls ${BEST_CKPT_PATTERN} 2>/dev/null || true))

if [ ${#BEST_CKPT_FILES[@]} -gt 0 ]; then
  echo "Found existing best checkpoint: ${BEST_CKPT_FILES[0]}"
  echo "Skipping training and proceeding directly to testing..."
  TRAIN_STATUS=0
else
  echo "No existing checkpoint found. Launching training..."

  # Treat BASE_CONFIG as Hydra config name (e.g., Downstream_DILIGold_splits)
  srun python bin/train.py -cn ${BASE_CONFIG} \
    dataloaders.splits.train="${TRAIN_INNER_CSV}" \
    dataloaders.splits.val="${VAL_CSV}" \
    model._args_.0="${EXTRACTED_ENCODER_PATH}" \
    seed=${DOWNSTREAM_SEED} \
    dataloaders.seed=${DOWNSTREAM_SEED} \
    dataloaders.num_workers=${WORKERS} \
    dataloaders.batch_size=${BATCH_SIZE} \
    optimizer.lr=${LR} \
    optimizer.weight_decay=${WD} \
    trainer.callbacks.1.dirpath="${CHECKPOINT_DIR}" \
    trainer.logger.project="${LOGGER_PROJECT}" \
    trainer.logger.name="${LOGGER_NAME}"
  
  TRAIN_STATUS=$?
fi


if [ $TRAIN_STATUS -ne 0 ]; then
  echo "==========================================" >&2
  echo "Training failed" >&2
  echo "downstream_seed=$DOWNSTREAM_SEED | fold=$FOLD | lr=$LR | wd=$WD" >&2
  echo "Run dir: $RUN_DIR" >&2
  echo "==========================================" >&2
  exit $TRAIN_STATUS
fi

echo "Training completed successfully!"
echo ""

# Refresh best checkpoint discovery after training (new files may have been created)
BEST_CKPT_FILES=($(ls ${BEST_CKPT_PATTERN} 2>/dev/null || true))

# -----------------------------------------------------------------------------
# Step 5: Run testing with best checkpoint
# -----------------------------------------------------------------------------
echo "Step 5: Running testing with best checkpoint..."

# Re-validate best checkpoint exists and select it
if [ ${#BEST_CKPT_FILES[@]} -eq 0 ]; then
  echo "Error: No best checkpoint found in ${CHECKPOINT_DIR}" >&2
  exit 1
fi
BEST_CKPT_PATH="${BEST_CKPT_FILES[0]}"
echo "Using best checkpoint for testing: ${BEST_CKPT_PATH}"
BEST_EPOCH=$(basename "$BEST_CKPT_PATH" | sed -E 's/.*best-epoch-([0-9]+)-.*/\1/')
echo "Best epoch parsed from checkpoint: ${BEST_EPOCH}"

# Setup test results (centralized location)
# Extract base save directory (parent of model-specific directories)
BASE_SAVE_DIR=$(dirname "$(dirname "$SAVE_DIR")")
TEST_RESULTS_DIR="$BASE_SAVE_DIR/test_results"
TEST_RESULTS_FILENAME="DILI_${MODEL_TYPE}_pretrained_seed_${PRETRAIN_SEED}_downstream_seed_${DOWNSTREAM_SEED}_fold_${FOLD}_lr_${LR}_wd_${WD}"
mkdir -p "$TEST_RESULTS_DIR"

echo "Test results dir: $TEST_RESULTS_DIR"
echo "Test results filename: $TEST_RESULTS_FILENAME"

# Run testing (evaluate on validation set)
srun python bin/test.py -cn ${BASE_CONFIG} \
  seed=${DOWNSTREAM_SEED} \
  dataloaders.seed=${DOWNSTREAM_SEED} \
  dataloaders.num_workers=${WORKERS} \
  dataloaders.batch_size=${BATCH_SIZE} \
  dataloaders.splits.train="${TRAIN_INNER_CSV}" \
  dataloaders.splits.val="${VAL_CSV}" \
  trainer.logger.project="${LOGGER_PROJECT}" \
  trainer.logger.name="${LOGGER_NAME}_test" \
  test_model.checkpoint_path="${BEST_CKPT_PATH}" \
  test_model_ckpt="${BEST_CKPT_PATH}" \
  test_results_dir="${TEST_RESULTS_DIR}" \
  test_results_filename="${TEST_RESULTS_FILENAME}"

TEST_STATUS=$?
if [ $TEST_STATUS -eq 0 ]; then
  echo "=========================================="
  echo "Job completed successfully"
  echo "downstream_seed=$DOWNSTREAM_SEED | fold=$FOLD | lr=$LR | wd=$WD"
  echo "Run dir: $RUN_DIR"
  echo "Checkpoints: $CHECKPOINT_DIR"
  echo "Test results: ${TEST_RESULTS_DIR}/${TEST_RESULTS_FILENAME}"
  echo "=========================================="
  # Annotate results JSON with best_epoch using a compact in-place update
  RESULTS_JSON="${TEST_RESULTS_DIR}/${TEST_RESULTS_FILENAME}.json"
  if [ -f "$RESULTS_JSON" ]; then
    python -c "import json,sys; p,e=sys.argv[1],sys.argv[2]; d=json.load(open(p)); d['best_epoch']=int(e) if e.isdigit() else e; json.dump(d, open(p,'w'), indent=2)" "$RESULTS_JSON" "$BEST_EPOCH"
    echo "Annotated results with best_epoch=${BEST_EPOCH}."
  else
    echo "Warning: Results JSON not found to annotate: $RESULTS_JSON" >&2
  fi
else
  echo "==========================================" >&2
  echo "Testing failed" >&2
  echo "downstream_seed=$DOWNSTREAM_SEED | fold=$FOLD | lr=$LR | wd=$WD" >&2
  echo "Run dir: $RUN_DIR" >&2
  echo "==========================================" >&2
fi

rm -rf "$CHECKPOINT_DIR"

exit $TEST_STATUS


