#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --job-name=create_lincs_splits
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/create_lincs_splits.out

# Check if arguments are provided
if [ $# -ne 4 ]; then
    echo "❌ ERROR: This script requires exactly 4 arguments"
    echo "Usage: $0 <genomic_data_path> <jump_data_path> <output_dir> <conda_environment>"
    echo "Example: $0 /path/to/lincs_data.parquet /path/to/jump_splits /path/to/output mocop"
    exit 1
fi

# Get arguments
GENOMIC_DATA_PATH="$1"
JUMP_DATA_PATH="$2"
OUTPUT_DIR="$3"
CONDA_ENV="$4"

echo "🚀 Starting LINCS data splitting..."
echo "📊 Genomic data path: $GENOMIC_DATA_PATH"
echo "📁 JUMP data path: $JUMP_DATA_PATH"
echo "📁 Output directory: $OUTPUT_DIR"
echo "🐍 Conda environment: $CONDA_ENV"

# Check if input files exist
if [ ! -f "$GENOMIC_DATA_PATH" ]; then
    echo "❌ Error: Genomic data file not found: $GENOMIC_DATA_PATH"
    exit 1
fi

if [ ! -d "$JUMP_DATA_PATH" ]; then
    echo "❌ Error: JUMP data directory not found: $JUMP_DATA_PATH"
    exit 1
fi

# Setup conda environment
echo "🐍 Activating conda environment: $CONDA_ENV"
module load mamba
source activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate conda environment."
    exit 1
fi

echo "=== Creating output directory ==="
mkdir -p "$OUTPUT_DIR"

# Use absolute path to Python script (like in process_jump_data.sh)
PYTHON_SCRIPT="/scratch/work/masooda1/Multi_Modal_Contrastive/data/LINCS_splitting_all_cell_lines.py"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

echo "=== Running LINCS splitting script ==="
python -u "$PYTHON_SCRIPT" \
    --genomic_data_path "$GENOMIC_DATA_PATH" \
    --jump_data_path "$JUMP_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --seeds 0 1 2

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to create LINCS splits."
    exit 1
fi

echo "=== Verifying output files ==="
# List created files
if ls "$OUTPUT_DIR"/*.csv 1> /dev/null 2>&1; then
    echo "✅ Created files:"
    ls -la "$OUTPUT_DIR"/*.csv
    
    # Count files
    LINCS_FILES=$(ls "$OUTPUT_DIR"/LINCS-compound-split-*.csv 2>/dev/null | wc -l)
    COMBINED_FILES=$(ls "$OUTPUT_DIR"/JUMP-LINCS-compound-split-*.csv 2>/dev/null | wc -l)
    
    echo "📊 Summary:"
    echo "   - LINCS-only split files: $LINCS_FILES"
    echo "   - Combined JUMP-LINCS split files: $COMBINED_FILES"
else
    echo "❌ Error: No CSV files found in output directory"
    exit 1
fi

echo "✅ All LINCS splitting completed successfully!"
