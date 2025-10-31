#!/bin/bash -l
#SBATCH --time=00:60:00
#SBATCH --mem=80G
#SBATCH --job-name=check_smiles
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/check_invalid_smiles.out

# Check if arguments are provided
if [ $# -ne 4 ]; then
    echo "❌ ERROR: This script requires exactly 4 arguments"
    echo "Usage: $0 <cell_data_path> <genomic_data_path> <output_path> <conda_environment>"
    echo "Example: $0 /path/to/cell_data.parquet /path/to/genomic_data.parquet /path/to/output.csv mocop"
    exit 1
fi

# Get arguments
CELL_DATA_PATH=$1
GENOMIC_DATA_PATH=$2
OUTPUT_PATH=$3
CONDA_ENV=$4

echo "🚀 Starting SMILES validation..."
echo "📁 Cell data: $CELL_DATA_PATH"
echo "📁 Genomic data: $GENOMIC_DATA_PATH"
echo "📁 Output: $OUTPUT_PATH"
echo "🐍 Conda environment: $CONDA_ENV"

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"

echo "🐍 Activating conda environment: $CONDA_ENV"
module load mamba
source activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate conda environment."
    exit 1
fi

echo "=== Running SMILES validation script ==="
python /scratch/work/masooda1/Multi_Modal_Contrastive/check_invalid_smiles.py \
    --cell_data "$CELL_DATA_PATH" \
    --genomic_data "$GENOMIC_DATA_PATH" \
    --output "$OUTPUT_PATH" \
    --smiles_col Metadata_SMILES

# Check if script succeeded
if [ $? -ne 0 ]; then
    echo "❌ Error: SMILES validation script failed."
    exit 1
fi

echo "✅ SMILES validation completed successfully!"
echo "📄 Report saved to: $OUTPUT_PATH"
