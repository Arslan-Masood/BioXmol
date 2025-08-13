#!/bin/bash
#SBATCH --job-name=jump_both_agg
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/both_aggregate.out

# Load required modules
module load mamba

# Activate conda environment
source activate mocop

# Change to the working directory
cd /scratch/work/masooda1/Multi_Modal_Contrastive

# Set data directories
DATA_DIR="/scratch/work/masooda1/datasets/jump_data"
OUTPUT_DIR="/scratch/work/masooda1/datasets/jump_data"

echo "Starting aggregation of both original and centered data..."
echo "Input directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"

# First: Aggregate original data (without --is_centered flag)
echo "=== Aggregating ORIGINAL data ==="
#python data/_jump_aggregate.py -d $DATA_DIR -o $OUTPUT_DIR

# Second: Aggregate centered data (with --is_centered flag)  
echo "=== Aggregating CENTERED data ==="
python data/_jump_aggregate.py -d $DATA_DIR -o $OUTPUT_DIR --is_centered

echo "Both aggregations completed!"
echo "Output files:"
echo "- ${OUTPUT_DIR}/filtered.parquet (original data)"
echo "- ${OUTPUT_DIR}/centered.filtered.parquet (normalized data)"
