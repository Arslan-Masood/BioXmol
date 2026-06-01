#!/bin/bash -l
#SBATCH --time=00:60:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/pyaivs_pipeline.out

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_csv> <output_dir> <conda_env>"
    echo "Example: $0 /path/to/DILIrank_2.0_binary_standard.csv /path/to/pyaivs pyaivs_env"
    exit 1
fi

# Arguments
INPUT_CSV=$1
OUTPUT_DIR=$2
CONDA_ENV=$3

# Derive CPU count from Slurm allocation (fallback to Num_CPU env var, then 4)
CPUS=${SLURM_CPUS_PER_TASK:-${Num_CPU:-4}}

set -euo pipefail

echo "Running PyaiVS pipeline"
echo "Input CSV: $INPUT_CSV"
echo "Output directory: $OUTPUT_DIR"
echo "Conda environment: $CONDA_ENV"
echo "CPUs: $CPUS"

set +u
module load mamba
source activate "$CONDA_ENV"
set -u
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi

python scripts/run_pyaivs_pipeline.py \
  --input_csv "$INPUT_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --cpus "$CPUS"

echo "PyaiVS pipeline finished."

