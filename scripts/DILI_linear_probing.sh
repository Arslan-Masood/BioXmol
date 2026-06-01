#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/dili_linear_probing.out

# Usage (all arguments required when called via DVC):
#   sbatch scripts/DILI_linear_probing.sh <conda_env> <features_root> <output_root>
# Example:
#   sbatch scripts/DILI_linear_probing.sh mocop /path/to/features_root /path/to/output_root

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <conda_env> <features_root> <output_root>"
  exit 1
fi

CONDA_ENV=$1
FEATURES_ROOT=$2
OUTPUT_ROOT=$3

SMILES_COL="SMILES_Normalized"
LABEL_COL="binary_label"
C_VALUES="1,0.1,0.01,0.001,0.0001"
PENALTIES="l1,l2"
SEED_LIST=(0 1 2 3 4)
INNER_SPLITS=5
COMPOUND_NAME_COL="Name"
EVAL_MODES=("activity_cliff" "cv")

# Traditional features (non-GNN)
TRADITIONAL_FEATURES=(
  "ECFP_1024_2"
  #"ChemBERTa-MLM"
  #"ChemBERTa-MTR"
  #"MolFormer_MoLFormer-XL-both-10pct"
)

# GNN model types and layers
GNN_MODEL_TYPES=(
  #"Soft_Clip_with_Frozen_Teacher"
  #"Soft_Clip_with_Teacher"
  #"Soft_Clip_with_Teacher_with_centering"
  #"Vanilla_Clip_with_VAE"
  "Vanilla_Clip_without_VAE"
)
GNN_SEEDS=(0 1 2)
GNN_LAYERS=("GNN" "first_fc" "second_fc")

echo "Activating conda environment: ${CONDA_ENV}"
module load mamba
source activate "${CONDA_ENV}" || { echo "Failed to activate ${CONDA_ENV}"; exit 1; }

# Calculate total runs
TOTAL_TRADITIONAL=$((${#TRADITIONAL_FEATURES[@]} * ${#SEED_LIST[@]}))
TOTAL_GNN=$((${#GNN_MODEL_TYPES[@]} * ${#GNN_SEEDS[@]} * ${#GNN_LAYERS[@]} * ${#SEED_LIST[@]}))
TOTAL_RUNS=$((TOTAL_TRADITIONAL + TOTAL_GNN))
CURRENT=0

echo "Total feature runs: ${TOTAL_RUNS}"
echo "  - Traditional features: ${TOTAL_TRADITIONAL}"
echo "  - GNN features: ${TOTAL_GNN}"
echo ""

# Process traditional features
for EVAL_MODE in "${EVAL_MODES[@]}"; do
  for SEED in "${SEED_LIST[@]}"; do
    for FEAT in "${TRADITIONAL_FEATURES[@]}"; do
      CURRENT=$((CURRENT + 1))
      FEATURES_FILE="${FEATURES_ROOT}/DILIrank_2.0_normalized_${FEAT}.csv"
      OUT_DIR="${OUTPUT_ROOT}/${EVAL_MODE}/${FEAT}/seed_${SEED}"

      echo "[${CURRENT}/${TOTAL_RUNS}] Running LR nested CV for feature set: ${FEAT} (seed=${SEED}) in ${EVAL_MODE} mode"
      if [ ! -f "${FEATURES_FILE}" ]; then
        echo "  ⚠️  Skipping: features file not found at ${FEATURES_FILE}"
        continue
      fi

      mkdir -p "${OUT_DIR}"

      srun python /scratch/work/masooda1/Multi_Modal_Contrastive/scripts/DILI_linear_probing_updated.py \
        --features_file "${FEATURES_FILE}" \
        --label_col "${LABEL_COL}" \
        --smiles_col "${SMILES_COL}" \
        --output_dir "${OUT_DIR}" \
        --seed "${SEED}" \
        --inner_splits "${INNER_SPLITS}" \
        --c_values "${C_VALUES}" \
        --penalties "${PENALTIES}"
        --eval_mode "${EVAL_MODE}"
        --compound_name_col "${COMPOUND_NAME_COL}"

      STATUS=$?
      if [ ${STATUS} -ne 0 ]; then
        echo "  ❌ Job failed for ${FEAT} (seed=${SEED}, exit ${STATUS}) in ${EVAL_MODE} mode"
      else
        echo "  ✅ Completed ${FEAT} (seed=${SEED}), results in ${OUT_DIR} in ${EVAL_MODE} mode"
      fi
    done
  done
done

# Process GNN features
for EVAL_MODE in "${EVAL_MODES[@]}"; do
  for CV_SEED in "${SEED_LIST[@]}"; do
    for MODEL_TYPE in "${GNN_MODEL_TYPES[@]}"; do
      for GNN_SEED in "${GNN_SEEDS[@]}"; do
        for LAYER in "${GNN_LAYERS[@]}"; do
          CURRENT=$((CURRENT + 1))
          # Clean model type name for filename (replace spaces/special chars)
          MODEL_NAME_CLEAN=$(echo "$MODEL_TYPE" | sed 's/[^a-zA-Z0-9_]/_/g')
          FEATURES_FILE="${FEATURES_ROOT}/DILIrank_2.0_normalized_GNN_${MODEL_NAME_CLEAN}_seed${GNN_SEED}_${LAYER}.csv"
          OUT_DIR="${OUTPUT_ROOT}/${EVAL_MODE}/GNN_${MODEL_NAME_CLEAN}_seed${GNN_SEED}_${LAYER}/seed_${CV_SEED}"

          echo "[${CURRENT}/${TOTAL_RUNS}] Running LR nested CV for: GNN ${MODEL_TYPE} seed${GNN_SEED} ${LAYER} (CV seed=${CV_SEED}) in ${EVAL_MODE} mode"
          if [ ! -f "${FEATURES_FILE}" ]; then
            echo "  ⚠️  Skipping: features file not found at ${FEATURES_FILE}"
            continue
          fi

          mkdir -p "${OUT_DIR}"

          srun python /scratch/work/masooda1/Multi_Modal_Contrastive/scripts/DILI_linear_probing_updated.py \
            --features_file "${FEATURES_FILE}" \
            --label_col "${LABEL_COL}" \
            --smiles_col "${SMILES_COL}" \
            --output_dir "${OUT_DIR}" \
            --seed "${CV_SEED}" \
            --inner_splits "${INNER_SPLITS}" \
            --c_values "${C_VALUES}" \
            --penalties "${PENALTIES}"
            --eval_mode "${EVAL_MODE}"
            --compound_name_col "${COMPOUND_NAME_COL}"

          STATUS=$?
          if [ ${STATUS} -ne 0 ]; then
            echo "  ❌ Job failed for GNN ${MODEL_TYPE} seed${GNN_SEED} ${LAYER} (CV seed=${CV_SEED}, exit ${STATUS}) in ${EVAL_MODE} mode"
          else
            echo "  ✅ Completed GNN ${MODEL_TYPE} seed${GNN_SEED} ${LAYER} (CV seed=${CV_SEED}), results in ${OUT_DIR} in ${EVAL_MODE} mode"
          fi
        done
      done
    done
  done
done

echo ""
echo "=========================================="
echo "✅ All feature runs completed!"
echo "=========================================="
echo "Total runs: ${TOTAL_RUNS}"
echo "  - Traditional features: ${TOTAL_TRADITIONAL}"
echo "  - GNN features: ${TOTAL_GNN}"
