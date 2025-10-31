#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=1-5  # 6 different weight decay values for DILI evaluation
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/downstream/DILI_finetuning_wd_%a.out

SAVE_DIR=$1
CHECKPOINT=$2  # Path to trained CellLineTripleInputEncoder checkpoint
CONFIG_FILE=$3
CONDA_ENV=$4

# Fixed seed for all runs
SEED=42
LR=1.0e-3

# Define weight decay values array
weight_decay_values=(0.0 0.01 0.1 1.0 5.0 10.0)

# Calculate weight decay based on array task ID
weight_decay=${weight_decay_values[$SLURM_ARRAY_TASK_ID]}
WORKERS=4
BATCH_SIZE=32
LOGGER_NAME="DILI_finetuning_without_pretrained_VAE_Adam_seed_${SEED}_LR_${LR}_wd_${weight_decay}_batch_${BATCH_SIZE}"

LOGGER_PROJECT="MultiModal-finetuning-DILI"

# Set Neptune API token
export HYDRA_FULL_ERROR=1
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ODlkMzRkMC1jYmM1LTQ5MjctOTBiMi1hYWQxNDg0ZGIxODMifQ=="
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Running DILI finetuning job for SEED=${SEED}, weight_decay=${weight_decay}"
echo "SAVE_DIR: ${SAVE_DIR}"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "LR: ${LR}"
echo "weight_decay: ${weight_decay}"
echo "WORKERS: ${WORKERS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "LOGGER_NAME: ${LOGGER_NAME}"

echo 'Starting script execution...'
module load mamba
echo "Activating conda environment: ${CONDA_ENV}"
source activate "${CONDA_ENV}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

# Check if source checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Source CellLineTripleInputEncoder checkpoint not found at ${CHECKPOINT}"
    exit 1
fi

# Create directory for extracted molecular encoder checkpoint
EXTRACTED_ENCODER_DIR="${SAVE_DIR}/extracted_molecular_encoders"
mkdir -p "${EXTRACTED_ENCODER_DIR}"

# Define extracted checkpoint path (use simple name to avoid Hydra parsing issues with = characters)
EXTRACTED_ENCODER_PATH="${EXTRACTED_ENCODER_DIR}/molecular_encoder.ckpt"

echo "=========================================="
echo "DILI Finetuning with CellLine MoCop Molecular Encoder"
echo "=========================================="
echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "SEED: ${SEED}"
echo "Source checkpoint: ${CHECKPOINT}"
echo "Extracted encoder will be saved to: ${EXTRACTED_ENCODER_PATH}"
echo ""

# Step 1: Extract molecular encoder if it doesn't exist
if [ ! -f "$EXTRACTED_ENCODER_PATH" ]; then
    echo "Step 1: Extracting molecular encoder from CellLineTripleInputEncoder checkpoint..."
    echo "Using existing remap_state_dict.py utility..."
    
    # First remapping: encoder_a -> model
    python /scratch/work/masooda1/Multi_Modal_Contrastive/bin/remap_state_dict.py \
        -i "${CHECKPOINT}" \
        -o "${EXTRACTED_ENCODER_PATH}" \
        --map_from "encoder_a" \
        --map_to "model"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to remap encoder_a to model."
        exit 1
    fi
    
    # Second remapping: model.fc_layers.1 -> model.fc_layers.1.0 (for 3+ layer compatibility)
    python /scratch/work/masooda1/Multi_Modal_Contrastive/bin/remap_state_dict.py \
        -i "${EXTRACTED_ENCODER_PATH}" \
        -o "${EXTRACTED_ENCODER_PATH}" \
        --map_from "model.fc_layers.1" \
        --map_to "model.fc_layers.1.0"
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to remap final layer structure."
        exit 1
    fi
    
    echo "Molecular encoder extracted and remapped successfully!"
else
    echo "Step 1: Using existing extracted molecular encoder at ${EXTRACTED_ENCODER_PATH}"
fi

echo ""

# Step 2: Set up directories for this specific run
RUN_DIR="${SAVE_DIR}/DILI_finetuning_Adam_seed_${SEED}_LR_${LR}_wd_${weight_decay}_batch_${BATCH_SIZE}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

echo "Step 2: Training setup"
echo "Run directory: ${RUN_DIR}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo ""

# Step 3: Run training
echo "Step 3: Running DILI finetuning training..."
srun python /scratch/work/masooda1/Multi_Modal_Contrastive/bin/train.py -cn ${CONFIG_FILE} \
                    seed=${SEED} \
                    model._args_.0=${EXTRACTED_ENCODER_PATH} \
                    model.freeze=false \
                    optimizer.lr=${LR} \
                    optimizer.weight_decay=${weight_decay} \
                    dataloaders.num_workers=${WORKERS} \
                    dataloaders.batch_size=${BATCH_SIZE} \
                    dataloaders.seed=${SEED} \
                    scheduler.max_lr=${LR} \
                    trainer.callbacks.1.dirpath=${CHECKPOINT_DIR} \
                    trainer.logger.project=\"${LOGGER_PROJECT}\" \
                    trainer.logger.name=\"${LOGGER_NAME}\" \
                    trainer.logger.api_key=\"${NEPTUNE_API_TOKEN}\"

if [ $? -ne 0 ]; then
    echo "Error: Training failed."
    exit 1
fi

echo 'Training completed successfully!'
echo ""

# Step 4: Run testing
echo "Step 4: Running DILI testing..."

# Find the last checkpoint (should be saved as last.ckpt)
LAST_CKPT="${CHECKPOINT_DIR}/last.ckpt"
echo "LAST_CKPT: ${LAST_CKPT}"

if [ ! -f "$LAST_CKPT" ]; then
    echo "Error: No last checkpoint found in ${CHECKPOINT_DIR}"
    exit 1
fi

echo "Using last checkpoint for testing: ${LAST_CKPT}"

TEST_RESULTS_DIR="${SAVE_DIR}/test_results/DILI_finetuning"
TEST_RESULTS_FILENAME="DILI_finetuning_Adam_seed_${SEED}_LR_${LR}_wd_${weight_decay}_batch_${BATCH_SIZE}"

mkdir -p "${TEST_RESULTS_DIR}"

srun python /scratch/work/masooda1/Multi_Modal_Contrastive/bin/test.py -cn ${CONFIG_FILE} \
                    seed=${SEED} \
                    dataloaders.num_workers=${WORKERS} \
                    dataloaders.batch_size=${BATCH_SIZE} \
                    dataloaders.seed=${SEED} \
                    trainer.logger.project=\"MultiModal-finetuning-DILI\" \
                    trainer.logger.name=\"${LOGGER_NAME}_test\" \
                    trainer.logger.api_key=\"${NEPTUNE_API_TOKEN}\" \
                    test_model.checkpoint_path=${LAST_CKPT} \
                    test_results_dir=${TEST_RESULTS_DIR} \
                    test_results_filename=${TEST_RESULTS_FILENAME}

if [ $? -ne 0 ]; then
    echo "Error: Testing failed."
    exit 1
fi

echo 'Testing completed successfully!'
echo ""
echo "=========================================="
echo "DILI Finetuning Completed Successfully!"
echo "=========================================="
echo "Results saved to: ${TEST_RESULTS_DIR}/${TEST_RESULTS_FILENAME}"
echo "Logs and checkpoints: ${RUN_DIR}"
echo ""
