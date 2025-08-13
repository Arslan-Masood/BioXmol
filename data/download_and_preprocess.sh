#!/bin/bash

# Check if arguments are provided
if [ $# -ne 2 ]; then
    echo "❌ ERROR: This script requires exactly 2 arguments"
    echo "Usage: $0 <output_directory> <conda_environment>"
    echo "Example: $0 data/processed multi_modal_contrastive"
    exit 1
fi

OUTPUT_DIR=$1
CONDA_ENV=$2

echo "🚀 Starting download and preprocessing..."
echo "📁 Output directory: $OUTPUT_DIR"
echo "🐍 Conda environment: $CONDA_ENV"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Uncompressing ChEMBL20 data and splits"
tar -xzvf data/chembl20.tar.gz --directory "$OUTPUT_DIR/"

echo "Uncompressing JUMP-CP splits"
tar -xzvf data/jump.tar.gz --directory "$OUTPUT_DIR/"

echo "Cloning JUMP-CP metadata repo"
git clone https://github.com/jump-cellpainting/datasets
METADATA_PATH=datasets

# Create preprocessed data directory
PREPROCESSED_DIR="$OUTPUT_DIR/jump_preprocessed"
mkdir -p "$PREPROCESSED_DIR"

echo "Downloading and normalizing JUMP-CP compound plates"
sbatch  --time=5-00 \
        --mem=40G \
        --array=0-4 \
        --cpus-per-task=4 \
        --partition=cpu \
        --wait \
        --export=CONDA_ENV=${CONDA_ENV},OUTPUT_DIR=${PREPROCESSED_DIR},METADATA_PATH=${METADATA_PATH} \
        --wrap "module load miniconda && \
                source activate \${CONDA_ENV} && \
                source ./.env && \
                python data/_jump_download_single_plate.py -o \${OUTPUT_DIR} -m \${METADATA_PATH}"

# First: Aggregate original data (without --is_centered flag)
echo "=== Aggregating ORIGINAL data ==="
python data/_jump_aggregate.py -d $PREPROCESSED_DIR -o $PREPROCESSED_DIR

# Second: Aggregate centered data (with --is_centered flag)  
echo "=== Aggregating CENTERED data ==="
python data/_jump_aggregate.py -d $PREPROCESSED_DIR -o $PREPROCESSED_DIR --is_centered

echo "=== Aggregation completed! ==="

# Cleanup individual plate files after aggregation
echo "=== Starting cleanup of individual plate files ==="
python data/cleanup_individual_plates.py $PREPROCESSED_DIR

echo "✅ Done! All processing completed successfully."