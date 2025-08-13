#!/bin/bash -l
#SBATCH --time=00:30:00
#SBATCH --mem=40G
#SBATCH --job-name=aggregate_data
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/aggregate_data.out

OUTPUT_DIR="/scratch/work/masooda1/datasets/jump_data"
SPLITS_DIR="/scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data_splits"
echo "Data directory: $OUTPUT_DIR"
VENV_PATH="/scratch/work/masooda1/.conda_envs/mocop"

echo "Activating conda environment: $VENV_PATH"
module load mamba
source activate "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi

echo "Running aggregation script..."
#python -u /scratch/work/masooda1/Multi_Modal_Contrastive/data/_jump_aggregate.py -d ${OUTPUT_DIR} -o ${OUTPUT_DIR} --is_centered

echo "Running splits creation script..."
mkdir -p ${SPLITS_DIR}
#python -u /scratch/work/masooda1/Multi_Modal_Contrastive/data/jump_data_splits.py \
#    ${OUTPUT_DIR}/centered.filtered.parquet \
#    ${SPLITS_DIR}

echo "Running cellular data split script for dummy data generation..."
DUMMY_DATA_DIR="/scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data"
mkdir -p ${DUMMY_DATA_DIR}
python -u /scratch/work/masooda1/Multi_Modal_Contrastive/data/dummy_data/script_for_dummy_data/cellular_data_split.py \
    ${OUTPUT_DIR}/centered.filtered.parquet \
    ${DUMMY_DATA_DIR}

echo "Done!"

