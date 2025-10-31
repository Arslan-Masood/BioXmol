#!/bin/bash -l
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_vae
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/vae_evaluation.out

# Arguments
CHECKPOINT_PATH="$1"
CONFIG_FILE="$2"
OUTPUT_DIR="$3"
VENV_PATH="$4"

if [ -z "$CHECKPOINT_PATH" ] || [ -z "$CONFIG_FILE" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$VENV_PATH" ]; then
    echo "Error: Missing arguments!"
    echo "Usage: sbatch $0 <checkpoint_path> <config_file> <output_dir> <venv_path>"
    echo "Example: sbatch $0 vae_logs/jump_ae_vanilla/checkpoints/last.ckpt configs/train_vae_config.yaml evaluation_results/jump_ae_best mocop"
    exit 1
fi

echo "Evaluation job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Virtual environment: $VENV_PATH"

# Activate conda environment
echo "Activating conda environment: $VENV_PATH"
module load mamba
source activate "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "Starting VAE evaluation..."
python evaluate_vae.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR"

echo "Evaluation completed at: $(date)" 