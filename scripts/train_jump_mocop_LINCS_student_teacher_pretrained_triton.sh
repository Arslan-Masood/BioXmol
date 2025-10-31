#!/bin/bash
#SBATCH --time=00:60:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g,gpu-v100-16g,gpu-h200-18g-ia
#SBATCH --array=0-120
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/Contrastive_training/Student_Teacher_with_pretrained_VAE_%A_%a.out

SAVE_DIR=$1
CONFIG_FILE=$2
CONDA_ENV=$3


# Define arrays of temperature values to test
TEMP_MAIN_VALUES=(10)  # Fixed based on your best results
TEMP_MOMENTUM_MORPH_VALUES=(1 5 10 20 30 40 50 100 200 500 1000)  # Test around your best morphological result (10)
TEMP_MOMENTUM_GENOMIC_VALUES=(1 5 10 20 30 40 50 100 200 500 1000)  # Test around your best genomic result (50)

# Calculate parameters based on array task ID (11x11 grid)
SEED=0  # Fixed seed for this experiment
SPLIT=0  # Fixed split for this experiment
LR=0.01
WORKERS=4
BATCH_SIZE=256

# Convert 1D array index to 2D grid coordinates
TEMP_MAIN=10  # Fixed based on your best results
TEMP_MOMENTUM_MORPH_IDX=$((SLURM_ARRAY_TASK_ID / 11))
TEMP_MOMENTUM_GENOMIC_IDX=$((SLURM_ARRAY_TASK_ID % 11))

# Temperature parameters for student-teacher architecture
TEMP_MOMENTUM_MORPH=${TEMP_MOMENTUM_MORPH_VALUES[$TEMP_MOMENTUM_MORPH_IDX]}
TEMP_MOMENTUM_GENOMIC=${TEMP_MOMENTUM_GENOMIC_VALUES[$TEMP_MOMENTUM_GENOMIC_IDX]}
LOGGER_NAME="Dummy_Student_Frozen_Teacher_No_Centering_TempMain_${TEMP_MAIN}_TempMomentumMorph_${TEMP_MOMENTUM_MORPH}_TempMomentumGenomic_${TEMP_MOMENTUM_GENOMIC}_seed_${SEED}_split_${SPLIT}_LR_${LR}"
#LOGGER_NAME="dummy_pretrained_${SEED}_split_${SPLIT}_LR_${LR}_batch_${BATCH_SIZE}"

LOGGER_PROJECT="arslan-masood/multimodal-contrastive-training"

# Set Neptune API token
export HYDRA_FULL_ERROR=1
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ODlkMzRkMC1jYmM1LTQ5MjctOTBiMi1hYWQxNDg0ZGIxODMifQ=="
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Running separate momentum temperature sweep job"
echo "Grid coordinates: TEMP_MOMENTUM_MORPH_IDX=${TEMP_MOMENTUM_MORPH_IDX}, TEMP_MOMENTUM_GENOMIC_IDX=${TEMP_MOMENTUM_GENOMIC_IDX}"
echo "SEED: ${SEED}, SPLIT: ${SPLIT}"
echo "SAVE_DIR: ${SAVE_DIR}"
echo "CONDA_ENV: ${CONDA_ENV}"
echo "LR: ${LR}"
echo "WORKERS: ${WORKERS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "TEMP_MAIN: ${TEMP_MAIN} (fixed)"
echo "TEMP_MOMENTUM_MORPH: ${TEMP_MOMENTUM_MORPH} (from values: ${TEMP_MOMENTUM_MORPH_VALUES[*]})"
echo "TEMP_MOMENTUM_GENOMIC: ${TEMP_MOMENTUM_GENOMIC} (from values: ${TEMP_MOMENTUM_GENOMIC_VALUES[*]})"

echo 'Starting script execution...'
module load mamba
echo "Activating conda environment: ${CONDA_ENV}"
source activate "${CONDA_ENV}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

# Create run directory with temperature information
RUN_DIR="${SAVE_DIR}/Student_Teacher_pretrained_VAE_TempMain_${TEMP_MAIN}_TempMomentumMorph_${TEMP_MOMENTUM_MORPH}_TempMomentumGenomic_${TEMP_MOMENTUM_GENOMIC}_seed_${SEED}_split_${SPLIT}_LR_${LR}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

echo 'Running training script...'
srun python /scratch/work/masooda1/Multi_Modal_Contrastive/bin/train.py -cn ${CONFIG_FILE} \
                    seed=${SEED} \
                    split=${SPLIT} \
                    optimizer.lr=${LR} \
                    scheduler.max_lr=${LR} \
                    dataloaders.num_workers=${WORKERS} \
                    dataloaders.batch_size=${BATCH_SIZE} \
                    model.temperature_main=${TEMP_MAIN} \
                    model.temperature_momentum_morph=${TEMP_MOMENTUM_MORPH} \
                    model.temperature_momentum_genomic=${TEMP_MOMENTUM_GENOMIC} \
                    trainer.callbacks.1.dirpath=${CHECKPOINT_DIR} \
                    trainer.logger.project=\"${LOGGER_PROJECT}\" \
                    trainer.logger.name=\"${LOGGER_NAME}\" \
                    trainer.logger.api_key=\"${NEPTUNE_API_TOKEN}\"

echo "Training with student-teacher architecture completed successfully for TEMP_MAIN=${TEMP_MAIN}, TEMP_MOMENTUM_MORPH=${TEMP_MOMENTUM_MORPH}, TEMP_MOMENTUM_GENOMIC=${TEMP_MOMENTUM_GENOMIC}."
