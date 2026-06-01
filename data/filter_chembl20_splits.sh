#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --job-name=filter_chembl_splits
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/filter_chembl20_splits.out

# Usage: filter_chembl20_splits.sh <splits_dir> <invalid_report_csv> <out_dir> <conda_env> <master_file>

if [ $# -ne 5 ]; then
    echo "❌ ERROR: This script requires exactly 5 arguments"
    echo "Usage: $0 <splits_dir> <invalid_report_csv> <out_dir> <conda_env> <master_file>"
    echo "Example: $0 data/chembl20 data/chembl20/invalid_smiles.csv data/chembl20/filtered_splits mocop data/chembl20/chembl20.csv"
    exit 1
fi

SPLITS_DIR=$1
INVALID_REPORT=$2
OUT_DIR=$3
CONDA_ENV=$4
MASTER_FILE=$5

echo "🚀 Filtering split CSVs..."
echo "📁 Splits dir: $SPLITS_DIR"
echo "🚫 Invalid report: $INVALID_REPORT"
echo "📤 Output dir: $OUT_DIR"
echo "🐍 Conda env: $CONDA_ENV"
echo "📄 Master file: $MASTER_FILE"

module load mamba
source activate "$CONDA_ENV"
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate conda environment."
    exit 1
fi

python data/filter_chembl20_splits.py \
  --splits_dir "$SPLITS_DIR" \
  --invalid_report "$INVALID_REPORT" \
  --out_dir "$OUT_DIR" \
  --master_file "$MASTER_FILE"

if [ $? -ne 0 ]; then
  echo "❌ Error: Filtering script failed."
  exit 1
fi

echo "✅ Filtering completed. Outputs in: $OUT_DIR"


