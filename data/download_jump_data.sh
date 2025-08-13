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
SCRIPT_OUTPUT_DIR="$OUTPUT_DIR/script_outputs"

mkdir -p "$PREPROCESSED_DIR"

echo "Downloading and normalizing JUMP-CP compound plates"
sbatch  --time=00:60:00 \
        --mem=40G \
        --array=0-1729 \
        --cpus-per-task=1 \
        --wait \
        --output=${SCRIPT_OUTPUT_DIR}/slurm-%a.out \
        --export=ALL,CONDA_ENV=${CONDA_ENV},PREPROCESSED_DIR=${PREPROCESSED_DIR},JUMP_METADATA_PATH=${METADATA_PATH} \
        --wrap "source /appl/scibuilder-mamba/aalto-rhel9/prod/software/mamba/2024-01/39cf5e1/etc/profile.d/conda.sh && \
                conda activate \${CONDA_ENV} && \
                python data/_jump_download_single_plate.py -o \${PREPROCESSED_DIR} -m \${JUMP_METADATA_PATH}"

# Check if sbatch job succeeded
if [ $? -ne 0 ]; then
    echo "❌ Error: SLURM job failed during JUMP data download."
    exit 1
fi

echo "✅ Download and initial preprocessing completed!"
echo "📋 Next step: Run data/process_data_trirton.sh for aggregation and cleanup"