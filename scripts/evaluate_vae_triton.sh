#!/bin/bash -l
#SBATCH --time=00:15:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_vae
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/vae_evaluation.out

# Set up environment
VENV_PATH="/scratch/work/masooda1/.conda_envs/mocop"
output_dir="/scratch/work/masooda1/Multi_Modal_Contrastive/evaluation_results"

# Arguments
CHECKPOINT_PATH="$1"
CONFIG_FILE="$2"

if [ -z "$CHECKPOINT_PATH" ] || [ -z "$CONFIG_FILE" ]; then
    echo "Error: Missing arguments!"
    echo "Usage: sbatch $0 <checkpoint_path> <config_file>"
    echo "Example: sbatch $0 vae_logs/jump_ae_vanilla/checkpoints/last.ckpt configs/train_vae_config.yaml"
    exit 1
fi

echo "Evaluation job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Config file: $CONFIG_FILE"

# Activate conda environment
echo "Activating conda environment: $VENV_PATH"
module load mamba
source activate "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi

# Create output directory
mkdir -p evaluation_results

# Run evaluation
echo "Starting VAE evaluation..."
python evaluate_vae.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --config "$CONFIG_FILE" \
    --output_dir evaluation_results

echo "Evaluation completed at: $(date)" 