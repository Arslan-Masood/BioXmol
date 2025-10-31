#!/bin/bash
#SBATCH --time=110:00:00
#SBATCH --mem=400G
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu-a100-80g,gpu-h100-80g,gpu-h200-141g-ellis,gpu-h200-141g-short,gpu-v100-32g
#SBATCH --array=0-2
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/Contrastive_training/Vanilla_Clip_multi_gpu_%a.out

SAVE_DIR=$1
CONFIG_FILE=$2
CONDA_ENV=$3


# Calculate SEED and SPLIT based on array task ID
SEED=$SLURM_ARRAY_TASK_ID
SPLIT=$SEED

WORKERS=$SLURM_CPUS_PER_GPU
N_GPUS=$SLURM_GPUS_ON_NODE
BASE_LR=0.001
SCALED_LR=0.001 #$(echo "$BASE_LR * $N_GPUS" | bc)
BATCH_SIZE=256
LOGGER_NAME="Vanilla_Clip_seed_${SEED}_split_${SPLIT}_LR_${SCALED_LR}_batch_${BATCH_SIZE}_N_GPUS_${N_GPUS}_WORKERS_${WORKERS}"
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
echo "N_GPUS: ${N_GPUS}"
echo "SCALED_LR: ${SCALED_LR}"

echo 'Starting script execution...'
module load mamba
echo "Activating conda environment: ${CONDA_ENV}"
source activate "${CONDA_ENV}"
if [ $? -ne 0 ]; then
    echo 'Error: Failed to activate conda environment.'
    exit 1
fi

# Create run directory
RUN_DIR="${SAVE_DIR}/Vanilla_Clip_seed_${SEED}_split_${SPLIT}_LR_${SCALED_LR}_batch_${BATCH_SIZE}_N_GPUS_${N_GPUS}_WORKERS_${WORKERS}"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"

echo 'Running training script...'
srun python /scratch/work/masooda1/Multi_Modal_Contrastive/bin/train.py -cn ${CONFIG_FILE} \
                    seed=${SEED} \
                    split=${SPLIT} \
                    optimizer.lr=${SCALED_LR} \
                    scheduler.max_lr=${SCALED_LR} \
                    dataloaders.num_workers=${WORKERS} \
                    dataloaders.batch_size=${BATCH_SIZE} \
                    trainer.devices=${N_GPUS} \
                    trainer.callbacks.1.dirpath=${CHECKPOINT_DIR} \
                    trainer.logger.project=\"${LOGGER_PROJECT}\" \
                    trainer.logger.name=\"${LOGGER_NAME}\" \
                    trainer.logger.api_key=\"${NEPTUNE_API_TOKEN}\"

echo 'Training with pretrained VAE completed successfully.'