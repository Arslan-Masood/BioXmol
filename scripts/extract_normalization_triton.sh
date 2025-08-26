#!/bin/bash -l
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=extract_norm
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/extract_normalization.out

# Arguments
CONFIG_FILE="$1"
VENV_PATH="$2"

if [ -z "$CONFIG_FILE" ] || [ -z "$VENV_PATH" ]; then
    echo "Error: Missing arguments!"
    echo "Usage: sbatch $0 <config_file> <venv_path>"
    echo "Example: sbatch $0 configs/train_ae_jump_cp_config.yaml mocop"
    exit 1
fi

echo "Normalization extraction job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Config file: $CONFIG_FILE"
echo "Virtual environment: $VENV_PATH"

# Activate conda environment
echo "Activating conda environment: $VENV_PATH"
module load mamba
source activate "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi

echo "Starting normalization parameter extraction..."

# Run extraction script
python extract_normalization_params.py \
    --config "$CONFIG_FILE"

echo "Normalization extraction completed at: $(date)"
