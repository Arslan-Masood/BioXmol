#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=jump_vae
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/vae_training.out

# Arguments: config_file conda_env
CONFIG_FILE="$1"
CONDA_ENV="$2"

# Validate arguments
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file provided!"
    echo "Usage: sbatch $0 <config_file_path> <conda_env>"
    echo "Example: sbatch $0 configs/train_vae_config.yaml mocop"
    exit 1
fi

if [ -z "$CONDA_ENV" ]; then
    echo "Error: No conda environment provided!"
    echo "Usage: sbatch $0 <config_file_path> <conda_env>"
    echo "Example: sbatch $0 configs/train_vae_config.yaml mocop"
    exit 1
fi

echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Config file: $CONFIG_FILE"
echo "Conda environment: $CONDA_ENV"

# Activate conda environment
echo "Activating conda environment: $CONDA_ENV"
module load mamba
source activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p /scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs

# Run training with config file
echo "Starting VAE training..."
python train_vae.py --config "$CONFIG_FILE"

echo "Training completed at: $(date)"