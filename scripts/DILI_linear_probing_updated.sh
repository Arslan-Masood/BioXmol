#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/dili_linear_probing.out

# =============================================================================
# DILI Linear Probing - Unified CV + Activity Cliff Evaluation
# =============================================================================
#
# Usage:
#   sbatch DILI_linear_probing.sh <conda_env> <features_root> <output_root>
#
# Example:
#   sbatch DILI_linear_probing.sh mocop /path/to/features /path/to/output
#
# This script runs UNIFIED evaluation for each feature set:
#   - 5-fold CV on non-cliff molecules
#   - Activity cliff evaluation on held-out pairs
#   Both outputs saved to: output_root/<feature>/seed_<N>/
#
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <conda_env> <features_root> <output_root>"
  echo ""
  echo "Arguments:"
  echo "  conda_env      Conda environment name"
  echo "  features_root  Directory containing feature CSV files"
  echo "  output_root    Directory to save results"
  exit 1
fi

CONDA_ENV=$1
FEATURES_ROOT=$2
OUTPUT_ROOT=$3

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_PATH="/scratch/work/masooda1/Multi_Modal_Contrastive/scripts/DILI_linear_probing_updated.py"

# Column names
SMILES_COL="SMILES_Normalized"
LABEL_COL="binary_label"
COMPOUND_NAME_COL="Name"  # Required for activity cliff identification

# Hyperparameter search space
C_VALUES="1,0.1,0.01,0.001,0.0001"
PENALTIES="l1,l2"
INNER_SPLITS=5

# Seeds for repeated experiments
SEED_LIST=(0 1 2 3 4)

# -----------------------------------------------------------------------------
# Feature sets
# -----------------------------------------------------------------------------
# Traditional features (non-GNN)
TRADITIONAL_FEATURES=(
  "ECFP_1024_2"
  "ChemBERTa-MLM"
  "ChemBERTa-MTR"
  "MoLFormer-XL-both-10pct"
)

# GNN configurations
GNN_MODEL_TYPES=(
  "Soft_Clip_with_Frozen_Teacher"
  "Soft_Clip_with_Teacher"
  "Soft_Clip_with_Teacher_with_centering"
  "Vanilla_Clip_with_VAE"
  "Vanilla_Clip_without_VAE"
)
GNN_SEEDS=(0 1 2)
GNN_LAYERS=("GNN" "first_fc" "second_fc")

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
echo "=============================================="
echo "DILI Linear Probing Pipeline (Unified)"
echo "=============================================="
echo "Conda env:     ${CONDA_ENV}"
echo "Features root: ${FEATURES_ROOT}"
echo "Output root:   ${OUTPUT_ROOT}"
echo "Script:        ${SCRIPT_PATH}"
echo ""

echo "Activating conda environment: ${CONDA_ENV}"
module load mamba
source activate "${CONDA_ENV}" || { echo "Failed to activate ${CONDA_ENV}"; exit 1; }

# -----------------------------------------------------------------------------
# Helper function to run evaluation
# -----------------------------------------------------------------------------
run_evaluation() {
  local FEATURES_FILE=$1
  local FEATURE_NAME=$2
  local SEED=$3
  local CURRENT=$4
  local TOTAL=$5
  
  local OUT_DIR="${OUTPUT_ROOT}/${FEATURE_NAME}/seed_${SEED}"
  
  echo "[${CURRENT}/${TOTAL}] ${FEATURE_NAME} | seed=${SEED}"
  
  # Skip run if results are already available 
  if [ -f "${OUT_DIR}/cv_metrics.csv" ] && \
     [ -f "${OUT_DIR}/activity_cliff_summary.csv" ] && \
     [ -f "${OUT_DIR}/activity_cliff_pairs.csv" ]; then
    echo "  ⏭️  Skipping: Results already exist"
    return 0
  fi

  # Check if features file exists
  if [ ! -f "${FEATURES_FILE}" ]; then
    echo "  ⚠️  Skipping: ${FEATURES_FILE} not found"
    return 1
  fi
  
  mkdir -p "${OUT_DIR}"
  
  # Build command - runs both CV and activity cliff in one call
  local CMD="srun python ${SCRIPT_PATH} \
    --features_file ${FEATURES_FILE} \
    --label_col ${LABEL_COL} \
    --smiles_col ${SMILES_COL} \
    --compound_name_col ${COMPOUND_NAME_COL} \
    --output_dir ${OUT_DIR} \
    --seed ${SEED} \
    --inner_splits ${INNER_SPLITS} \
    --c_values ${C_VALUES} \
    --penalties ${PENALTIES}"
  
  # Execute
  eval ${CMD}
  
  local STATUS=$?
  if [ ${STATUS} -ne 0 ]; then
    echo "  ❌ Failed (exit ${STATUS})"
    return ${STATUS}
  else
    echo "  ✅ Done → ${OUT_DIR}"
    return 0
  fi
}

# -----------------------------------------------------------------------------
# Calculate total runs
# -----------------------------------------------------------------------------
N_TRADITIONAL=${#TRADITIONAL_FEATURES[@]}
N_GNN=$((${#GNN_MODEL_TYPES[@]} * ${#GNN_SEEDS[@]} * ${#GNN_LAYERS[@]}))
N_FEATURES=$((N_TRADITIONAL + N_GNN))
N_SEEDS=${#SEED_LIST[@]}
TOTAL_RUNS=$((N_FEATURES * N_SEEDS))

echo "=============================================="
echo "Run Summary"
echo "=============================================="
echo "Traditional features: ${N_TRADITIONAL}"
echo "GNN features:         ${N_GNN}"
echo "Total features:       ${N_FEATURES}"
echo "Seeds:                ${N_SEEDS}"
echo "----------------------------------------------"
echo "Total runs:           ${TOTAL_RUNS}"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Run evaluations
# -----------------------------------------------------------------------------
CURRENT=0
FAILED=0
SKIPPED=0

# Process traditional features
echo ""
echo "=============================================="
echo "Traditional Features"
echo "=============================================="

for SEED in "${SEED_LIST[@]}"; do
  for FEAT in "${TRADITIONAL_FEATURES[@]}"; do
    CURRENT=$((CURRENT + 1))
    FEATURES_FILE="${FEATURES_ROOT}/DILIrank_2.0_normalized_${FEAT}.csv"
    
    run_evaluation "${FEATURES_FILE}" "${FEAT}" "${SEED}" "${CURRENT}" "${TOTAL_RUNS}"
    
    STATUS=$?
    if [ ${STATUS} -eq 1 ]; then
      SKIPPED=$((SKIPPED + 1))
    elif [ ${STATUS} -ne 0 ]; then
      FAILED=$((FAILED + 1))
    fi
  done
done

# Process GNN features
echo ""
echo "=============================================="
echo "GNN Features"
echo "=============================================="

for SEED in "${SEED_LIST[@]}"; do
  for MODEL_TYPE in "${GNN_MODEL_TYPES[@]}"; do
    for GNN_SEED in "${GNN_SEEDS[@]}"; do
      for LAYER in "${GNN_LAYERS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        # Clean model type name for filename
        MODEL_NAME_CLEAN=$(echo "$MODEL_TYPE" | sed 's/[^a-zA-Z0-9_]/_/g')
        FEATURE_NAME="GNN_${MODEL_NAME_CLEAN}_seed${GNN_SEED}_${LAYER}"
        FEATURES_FILE="${FEATURES_ROOT}/DILIrank_2.0_normalized_${FEATURE_NAME}.csv"
        
        run_evaluation "${FEATURES_FILE}" "${FEATURE_NAME}" "${SEED}" "${CURRENT}" "${TOTAL_RUNS}"
        
        STATUS=$?
        if [ ${STATUS} -eq 1 ]; then
          SKIPPED=$((SKIPPED + 1))
        elif [ ${STATUS} -ne 0 ]; then
          FAILED=$((FAILED + 1))
        fi
      done
    done
  done
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Pipeline Complete"
echo "=============================================="
echo "Total runs:  ${TOTAL_RUNS}"
echo "Successful:  $((TOTAL_RUNS - FAILED - SKIPPED))"
echo "Failed:      ${FAILED}"
echo "Skipped:     ${SKIPPED}"
echo ""
echo "Results saved to: ${OUTPUT_ROOT}/"
echo ""
echo "Each feature folder contains:"
echo "  - cv_metrics.csv              (5-fold CV results)"
echo "  - activity_cliff_summary.csv  (cliff evaluation summary)"
echo "  - activity_cliff_pairs.csv    (per-pair breakdown)"
echo "=============================================="