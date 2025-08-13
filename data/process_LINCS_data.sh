#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH --job-name=LINCS_proc
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/process_lincs_data.out

# Check if arguments are provided
if [ $# -ne 3 ]; then
    echo "❌ ERROR: This script requires exactly 3 arguments"
    echo "Usage: $0 <input_data_directory> <output_directory> <conda_environment>"
    echo "Example: $0 /scratch/cs/pml/AI_drug/molecular_representation_learning/LINCS/ /scratch/work/masooda1/datasets/LINCS_processed/ cmappy_env"
    exit 1
fi

# Get arguments
INPUT_DIR=$1
OUTPUT_DIR=$2
CONDA_ENV=$3

echo "🚀 Starting LINCS data processing..."
echo "📁 Input directory: $INPUT_DIR"
echo "📁 Output directory: $OUTPUT_DIR"
echo "🐍 Conda environment: $CONDA_ENV"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "🐍 Activating conda environment: $CONDA_ENV"
module load mamba
source activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate conda environment."
    exit 1
fi

echo "=== Running LINCS processing script ==="
python /scratch/work/masooda1/Multi_Modal_Contrastive/data/process_LINCS_data.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \

# Check if first Python script succeeded
if [ $? -ne 0 ]; then
    echo "❌ Error: LINCS processing script failed."
    exit 1
fi

echo "=== Running LINCS dummy data ==="
python /scratch/work/masooda1/Multi_Modal_Contrastive/data/process_LINCS_data.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --test_mode true \

# Check if second Python script succeeded
if [ $? -ne 0 ]; then
    echo "❌ Error: LINCS dummy data processing script failed."
    exit 1
fi

echo "✅ LINCS processing completed successfully!"