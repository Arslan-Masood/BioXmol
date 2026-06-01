#!/bin/bash -l
#SBATCH --time=10:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=process_dilirank_v2
#SBATCH --output=/scratch/work/masooda1/Multi_Modal_Contrastive/script_outputs/process_dilirank_v2.out

# Check if arguments are provided (4 required, 2 optional)
if [ $# -lt 4 ] || [ $# -gt 6 ]; then
    echo "❌ ERROR: This script requires 4 required arguments and 2 optional arguments"
    echo "Usage: $0 <input_file> <output_dir> <env_step1> <env_step2_3> [checkpoint_path] [gnn_layers]"
    echo "Example (without GNN): $0 /scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2/DILIRank_raw.xlsx /scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2 pubchem_env mocop"
    echo "Example (with GNN): $0 /scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2/DILIRank_raw.xlsx /scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2 pubchem_env mocop /path/to/checkpoint.ckpt \"GNN,first_fc,second_fc\""
    exit 1
fi

# Get required arguments
INPUT_FILE=$1
OUTPUT_DIR=$2
ENV_STEP1=$3
ENV_STEP2_3=$4

# Get optional arguments (for GNN extraction)
CHECKPOINT_PATH=${5:-""}
GNN_LAYERS=${6:-"GNN,first_fc,second_fc"}

echo "🚀 Starting DILIRank v2.0 data processing pipeline..."
echo "📁 Input file: $INPUT_FILE"
echo "📁 Output directory: $OUTPUT_DIR"
echo "🐍 Step 1 environment: $ENV_STEP1"
echo "🐍 Step 2 & 3 environment: $ENV_STEP2_3"
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "🧠 GNN checkpoint: $CHECKPOINT_PATH"
    echo "🧠 GNN layers to extract: $GNN_LAYERS"
else
    echo "ℹ️  GNN extraction: Skipped (no checkpoint provided)"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define intermediate file paths
COMPLETE_FILE="$OUTPUT_DIR/DILIrank_2.0_complete.csv"
NORMALIZED_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized.csv"
BINARY_STANDARD_FILE="$OUTPUT_DIR/DILIrank_2.0_binary_standard.csv"
BINARY_CONSERVATIVE_FILE="$OUTPUT_DIR/DILIrank_2.0_binary_conservative.csv"
TRAIN_FILE="$OUTPUT_DIR/DILIrank_2.0_train.csv"
TEST_FILE="$OUTPUT_DIR/DILIrank_2.0_test.csv"
ECFP_FEATURES_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized_ECFP_1024_2.csv"
MOLFORMER_FEATURES_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized_MolFormer_MoLFormer-XL-both-10pct.csv"
CHEMBERTA_MTR_FEATURES_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized_ChemBERTa-MTR.csv"
CHEMBERTA_MLM_FEATURES_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized_ChemBERTa-MLM.csv"
GNN_GNN_FEATURES_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized_GNN_GNN.csv"
GNN_FIRST_FC_FEATURES_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized_GNN_first_fc.csv"
GNN_SECOND_FC_FEATURES_FILE="$OUTPUT_DIR/DILIrank_2.0_normalized_GNN_second_fc.csv"
FDA_PRODUCTS_FILE="$OUTPUT_DIR/products.txt"
DRUGBANK_FILE="/scratch/work/masooda1/datasets/downstream_datasets/drugbank.tsv"

# Define worker counts
# Step 1: Use 1 worker for PubChem/ChEMBL data fetching (to avoid rate limiting)
STEP1_WORKERS=1
# Use SLURM allocated CPUs if available for other steps
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-}

# ============================================
# Step 1: Process DILIRank data (fetch SMILES and dates)
# ============================================
echo ""
echo "=========================================="
echo "Step 1: Processing DILIRank v2.0 data"
echo "=========================================="

if [ -f "$COMPLETE_FILE" ]; then
    echo "⏭️  Skipping Step 1: Output file already exists"
    echo "   Found: $COMPLETE_FILE"
else
    echo "🐍 Activating conda environment: $ENV_STEP1"
    module load mamba
    source activate "$ENV_STEP1"
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to activate conda environment."
        exit 1
    fi

    echo "Using $STEP1_WORKERS worker for Step 1 (PubChem/ChEMBL data fetching)"
    python /scratch/work/masooda1/Multi_Modal_Contrastive/data/process_DILIRank_v2.py \
            --input_file "$INPUT_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --num_workers "$STEP1_WORKERS" \
            --fda_products_file "$FDA_PRODUCTS_FILE" \
            --drugbank_file "$DRUGBANK_FILE"

    # Check if Step 1 succeeded
    if [ $? -ne 0 ]; then
        echo "❌ Error: DILIRank v2.0 processing script failed."
        exit 1
    fi
fi

# ============================================
# Step 2: Normalize SMILES (uses env_step2_3 environment)
# ============================================
echo ""
echo "=========================================="
echo "Step 2: Normalizing SMILES"
echo "=========================================="

if [ -f "$NORMALIZED_FILE" ]; then
    echo "⏭️  Skipping Step 2: Output file already exists"
    echo "   Found: $NORMALIZED_FILE"
else
    # Check if input file exists
    if [ ! -f "$COMPLETE_FILE" ]; then
        echo "❌ Error: Step 1 output not found: $COMPLETE_FILE"
        echo "   Please run Step 1 first."
        exit 1
    fi
    
    echo "🐍 Activating conda environment: $ENV_STEP2_3"
    module load mamba
    source activate "$ENV_STEP2_3"
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to activate $ENV_STEP2_3 environment."
        exit 1
    fi

    if [ -n "$NUM_WORKERS" ]; then
        python /scratch/work/masooda1/Multi_Modal_Contrastive/data/SMILES_normalization.py \
                --input_file "$COMPLETE_FILE" \
                --output_file "$NORMALIZED_FILE" \
                --num_workers "$NUM_WORKERS"
    else
        python /scratch/work/masooda1/Multi_Modal_Contrastive/data/SMILES_normalization.py \
                --input_file "$COMPLETE_FILE" \
                --output_file "$NORMALIZED_FILE"
    fi

    # Check if Step 2 succeeded
    if [ $? -ne 0 ]; then
        echo "❌ Error: SMILES normalization script failed."
        exit 1
    fi
fi

# ============================================
# Step 3: Binarize labels (uses env_step2_3 environment)
# ============================================
echo ""
echo "=========================================="
echo "Step 3: Binarizing DILIRank labels"
echo "=========================================="

if [ -f "$BINARY_STANDARD_FILE" ] && [ -f "$BINARY_CONSERVATIVE_FILE" ]; then
    echo "⏭️  Skipping Step 3: Output files already exist"
    echo "   Found: $BINARY_STANDARD_FILE"
    echo "   Found: $BINARY_CONSERVATIVE_FILE"
else
    # Check if input file exists
    if [ ! -f "$NORMALIZED_FILE" ]; then
        echo "❌ Error: Step 2 output not found: $NORMALIZED_FILE"
        echo "   Please run Step 2 first."
        exit 1
    fi
    
    echo "🐍 Ensuring conda environment: $ENV_STEP2_3"
    module load mamba
    source activate "$ENV_STEP2_3"
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to activate $ENV_STEP2_3 environment."
        exit 1
    fi

    python /scratch/work/masooda1/Multi_Modal_Contrastive/data/DILI_v2_binarization.py \
            --input_file "$NORMALIZED_FILE" \
            --output_dir "$OUTPUT_DIR"

    # Check if Step 3 succeeded
    if [ $? -ne 0 ]; then
        echo "❌ Error: Label binarization script failed."
        exit 1
    fi
fi

# ============================================
# Step 4: Split data into train/test (temporal split)
# ============================================
echo ""
echo "=========================================="
echo "Step 4: Splitting data into train/test"
echo "=========================================="

if [ -f "$TRAIN_FILE" ] && [ -f "$TEST_FILE" ]; then
    echo "⏭️  Skipping Step 4: Output files already exist"
    echo "   Found: $TRAIN_FILE"
    echo "   Found: $TEST_FILE"
else
    # Check if input file exists (use standard binary file)
    if [ ! -f "$BINARY_STANDARD_FILE" ]; then
        echo "❌ Error: Step 3 output not found: $BINARY_STANDARD_FILE"
        echo "   Please run Step 3 first."
        exit 1
    fi
    
    echo "🐍 Ensuring conda environment: $ENV_STEP2_3"
    module load mamba
    source activate "$ENV_STEP2_3"
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to activate $ENV_STEP2_3 environment."
        exit 1
    fi

    python /scratch/work/masooda1/Multi_Modal_Contrastive/data/DILI_v2_split_temporal.py \
            --input_file "$BINARY_STANDARD_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --cutoff_year 2010 \
            --smiles_col "Normalized_SMILES_combined" \
            --label_col "binary_label" \
            --year_col "ChEMBL_First_Approval"

    # Check if Step 4 succeeded
    if [ $? -ne 0 ]; then
        echo "❌ Error: Data splitting script failed."
        exit 1
    fi
fi

# ============================================
# Step 5: Compute molecular features
# ============================================
echo ""
echo "=========================================="
echo "Step 5: Computing molecular features"
echo "=========================================="

if [ -f "$ECFP_FEATURES_FILE" ] && [ -f "$MOLFORMER_FEATURES_FILE" ] && \
   [ -f "$CHEMBERTA_MTR_FEATURES_FILE" ] && [ -f "$CHEMBERTA_MLM_FEATURES_FILE" ]; then
    echo "⏭️  Skipping Step 5: All feature files already exist"
    echo "   Found: $(basename $ECFP_FEATURES_FILE)"
    echo "   Found: $(basename $MOLFORMER_FEATURES_FILE)"
    echo "   Found: $(basename $CHEMBERTA_MTR_FEATURES_FILE)"
    echo "   Found: $(basename $CHEMBERTA_MLM_FEATURES_FILE)"
else
    # Check if input file exists
    if [ ! -f "$NORMALIZED_FILE" ]; then
        echo "❌ Error: Step 2 output not found: $NORMALIZED_FILE"
        echo "   Please run Step 2 first."
        exit 1
    fi
    
    echo "🐍 Ensuring conda environment: $ENV_STEP2_3"
    module load mamba
    source activate "$ENV_STEP2_3"
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to activate $ENV_STEP2_3 environment."
        exit 1
    fi

    # Compute features for normalized dataset (all types: ECFP, MolFormer, ChemBERTa-MTR, ChemBERTa-MLM)
    echo "Computing features for normalized dataset..."
    echo "  This will generate 4 feature files: ECFP, MolFormer, ChemBERTa-MTR, and ChemBERTa-MLM..."
    python /scratch/work/masooda1/Multi_Modal_Contrastive/data/fearurizer.py \
            --input_file "$NORMALIZED_FILE" \
            --smiles_col "Normalized_SMILES_combined" \
            --size 1024 \
            --radius 2

    # Check if features computation succeeded
    if [ $? -ne 0 ]; then
        echo "❌ Error: Features computation failed."
        exit 1
    fi
fi

# ============================================
# Step 6: Extract GNN embeddings (optional)
# ============================================
if [ -n "$CHECKPOINT_PATH" ]; then
    echo ""
    echo "=========================================="
    echo "Step 6: Extracting GNN embeddings"
    echo "=========================================="
    
    # Check if checkpoint file exists
    if [ ! -f "$CHECKPOINT_PATH" ]; then
        echo "❌ Error: Checkpoint file not found: $CHECKPOINT_PATH"
        echo "   Skipping GNN embedding extraction."
    else
        # Check if input file exists
        if [ ! -f "$NORMALIZED_FILE" ]; then
            echo "❌ Error: Step 2 output not found: $NORMALIZED_FILE"
            echo "   Please run Step 2 first."
            exit 1
        fi
        
        echo "🐍 Ensuring conda environment: $ENV_STEP2_3"
        module load mamba
        source activate "$ENV_STEP2_3"
        if [ $? -ne 0 ]; then
            echo "❌ Error: Failed to activate $ENV_STEP2_3 environment."
            exit 1
        fi
        
        # Parse layer names (comma-separated)
        IFS=',' read -ra LAYER_ARRAY <<< "$GNN_LAYERS"
        
        # Extract embeddings for each specified layer
        for LAYER_NAME in "${LAYER_ARRAY[@]}"; do
            # Trim whitespace
            LAYER_NAME=$(echo "$LAYER_NAME" | xargs)
            
            # Validate layer name
            if [[ ! "$LAYER_NAME" =~ ^(GNN|first_fc|second_fc)$ ]]; then
                echo "⚠️  Warning: Invalid layer name '$LAYER_NAME'. Skipping. Valid options: GNN, first_fc, second_fc"
                continue
            fi
            
            # Determine output file based on layer name
            case "$LAYER_NAME" in
                "GNN")
                    OUTPUT_FILE="$GNN_GNN_FEATURES_FILE"
                    ;;
                "first_fc")
                    OUTPUT_FILE="$GNN_FIRST_FC_FEATURES_FILE"
                    ;;
                "second_fc")
                    OUTPUT_FILE="$GNN_SECOND_FC_FEATURES_FILE"
                    ;;
            esac
            
            # Check if file already exists
            if [ -f "$OUTPUT_FILE" ]; then
                echo "⏭️  Skipping GNN-$LAYER_NAME: Output file already exists"
                echo "   Found: $(basename $OUTPUT_FILE)"
            else
                echo "Computing GNN-$LAYER_NAME embeddings..."
                python /scratch/work/masooda1/Multi_Modal_Contrastive/data/featurizer_GNN.py \
                        --input_file "$NORMALIZED_FILE" \
                        --output_file "$OUTPUT_FILE" \
                        --checkpoint_path "$CHECKPOINT_PATH" \
                        --smiles_col "Normalized_SMILES_combined" \
                        --layer_name "$LAYER_NAME" \
                        --batch_size 32
                
                # Check if extraction succeeded
                if [ $? -ne 0 ]; then
                    echo "❌ Error: GNN-$LAYER_NAME embedding extraction failed."
                    exit 1
                fi
            fi
        done
        
        echo "✅ GNN embedding extraction completed!"
    fi
else
    echo ""
    echo "ℹ️  Step 6: Skipping GNN embedding extraction (no checkpoint provided)"
fi

echo ""
echo "✅ DILIRank v2.0 processing pipeline completed successfully!"
echo "📁 Output files in: $OUTPUT_DIR"

