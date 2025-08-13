#!/bin/bash
#SBATCH --job-name=jump_umap
#SBATCH --time=00:30:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/umap_job.out

# Configuration parameters
DATA_DIR="/scratch/work/masooda1/datasets/jump_data"
OUTPUT_DIR="/scratch/work/masooda1/Multi_Modal_Contrastive/Figures/Data_plots"
MAX_PLATES=20
MAX_DRUGS=20
N_NEIGHBORS=15
MIN_DIST=0.1
RANDOM_SEED=0

# Load required modules
module load mamba

# Activate conda environment
source activate rapids-25.06

# Change to the working directory
cd /scratch/work/masooda1/Multi_Modal_Contrastive

echo "Starting UMAP analysis with parameters:"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Plate sampling: max $MAX_PLATES plates"
echo "Drug sampling: max $MAX_DRUGS drugs"
echo "UMAP n_neighbors: $N_NEIGHBORS"
echo "UMAP min_dist: $MIN_DIST"
echo "Random seed: $RANDOM_SEED"

# Run UMAP analysis
python plot_aggregated_umap.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --max-plates $MAX_PLATES \
    --max-drugs $MAX_DRUGS \
    --n-neighbors $N_NEIGHBORS \
    --min-dist $MIN_DIST \
    --random-seed $RANDOM_SEED

echo "UMAP analysis completed!"
echo "Check $OUTPUT_DIR directory for results"
