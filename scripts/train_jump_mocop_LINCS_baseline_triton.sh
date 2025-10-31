#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-32g,gpu-v100-16g,gpu-h200-18g-ia
#SBATCH --array=0-2
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/Contrastive_training/Vanilla_Clip_without_VAE_%a.out

SAVE_DIR=$1
CONFIG_FILE=$2
CONDA_ENV=$3


# Calculate SEED and SPLIT based on array task ID
SEED=$SLURM_ARRAY_TASK_ID
SPLIT=$SEED
LR=0.01
WORKERS=8
BATCH_SIZE=256
LOGGER_NAME="Vanilla_Clip_without_VAE_seed_${SEED}_split_${SPLIT}_LR_${LR}"
#LOGGER_NAME="dummy_pretrained_${SEED}_split_${SPLIT}_LR_${LR}_batch_${BATCH_SIZE}"

LOGGER_PROJECT="arslan-masood/multimodal-contrastive-training"

# Set Neptune API token
export HYDRA_FULL_ERROR=1
export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ODlkMzRkMC1jYmM1LTQ5MjctOTBiMi1hYWQxNDg0ZGIxODMifQ=="
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Running baseline job for SEED=${SEED}, SPLIT=${SPLIT}"
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
RUN_DIR="${SAVE_DIR}/Vanilla_Clip_without_VAE_seed_${SEED}_split_${SPLIT}_LR_${LR}"
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
                    trainer.callbacks.1.dirpath=${CHECKPOINT_DIR} \
                    trainer.logger.project=\"${LOGGER_PROJECT}\" \
                    trainer.logger.name=\"${LOGGER_NAME}\" \
                    trainer.logger.api_key=\"${NEPTUNE_API_TOKEN}\"

echo 'Baseline training completed successfully.'