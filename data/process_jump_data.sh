#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --mem=120G
#SBATCH --job-name=process_jump_data
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/process_jump_data.out

# Check if arguments are provided
if [ $# -ne 2 ]; then
    echo "❌ ERROR: This script requires exactly 2 arguments"
    echo "Usage: $0 <preprocessed_directory> <conda_environment>"
    echo "Example: $0 /scratch/work/masooda1/datasets/jump_preprocessed mocop"
    exit 1
fi

# Get arguments
PREPROCESSED_DIR=$1
CONDA_ENV=$2

# Data directories based on preprocessed dir
SPLITS_DIR="$PREPROCESSED_DIR/jump_data_splits"
DUMMY_DATA_DIR="$PREPROCESSED_DIR/dummy_data"

echo "🚀 Starting JUMP data processing..."
echo "📁 Preprocessed data directory: $PREPROCESSED_DIR"
echo "📁 Splits directory: $SPLITS_DIR"
echo "📁 Dummy data directory: $DUMMY_DATA_DIR"
echo "🐍 Conda environment: $CONDA_ENV"

# Setup conda environment
echo "🐍 Activating conda environment: $CONDA_ENV"
module load mamba
source activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate conda environment."
    exit 1
fi

# First: Aggregate original data (without --is_centered flag)
echo "=== Aggregating ORIGINAL data ==="
python -u /scratch/work/masooda1/Multi_Modal_Contrastive/data/_jump_aggregate.py -d $PREPROCESSED_DIR -o $PREPROCESSED_DIR
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to aggregate original data."
    exit 1
fi

# Second: Aggregate centered data (with --is_centered flag)  
echo "=== Aggregating CENTERED data ==="
python -u /scratch/work/masooda1/Multi_Modal_Contrastive/data/_jump_aggregate.py -d $PREPROCESSED_DIR -o $PREPROCESSED_DIR --is_centered
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to aggregate centered data."
    exit 1
fi

echo "=== Aggregation completed! ==="

# Cleanup individual plate files after aggregation
echo "=== Starting cleanup of individual plate files ==="
python -u /scratch/work/masooda1/Multi_Modal_Contrastive/data/cleanup_individual_plates.py $PREPROCESSED_DIR
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to cleanup individual plate files."
    exit 1
fi

echo "=== Creating data splits ==="
mkdir -p ${SPLITS_DIR}
python -u /scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data_splits.py \
    ${PREPROCESSED_DIR}/centered.filtered.parquet \
    ${SPLITS_DIR}
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to create data splits."
    exit 1
fi

echo "=== Creating dummy data ==="
mkdir -p ${DUMMY_DATA_DIR}
python -u /scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/script_for_dummy_data/cellular_data_split.py \
    ${PREPROCESSED_DIR}/centered.filtered.parquet \
    ${DUMMY_DATA_DIR}
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to create dummy data."
    exit 1
fi

echo "✅ All processing completed successfully!"

