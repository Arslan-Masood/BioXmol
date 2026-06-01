import warnings
import os
import sys

# Suppress RDKit deprecation warnings
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*please use MorganGenerator.*')
warnings.filterwarnings('ignore', message='.*Skipped loading.*')
warnings.filterwarnings('ignore', message='.*missing a dependency.*')
warnings.filterwarnings('ignore', message='.*cannot import name.*')

# Suppress RDKit logger warnings
import logging
logging.getLogger('rdkit').setLevel(logging.ERROR)

import deepchem as dc
import numpy as np
import pandas as pd
import argparse
import torch
from tqdm import tqdm
from typing import List, Union, Optional
from transformers import AutoModel, AutoTokenizer


def compute_ECFP(smiles: Union[str, List[str]], size: int = 1024, radius: int = 2) -> np.ndarray:
    """
    Compute molecular features using CircularFingerprint from deepchem.
    
    Args:
        smiles: Single SMILES string or list of SMILES strings
        size: Size of the fingerprint (default: 1024)
        radius: Radius of the circular fingerprint (default: 2)
    
    Returns:
        numpy.ndarray: Feature array(s). If single SMILES, returns 1D array of shape (size,).
                      If list of SMILES, returns 2D array of shape (n_smiles, size).
    
    Example:
        >>> smiles = ['C1=CC=CC=C1']
        >>> features = compute_ECFP(smiles)
        >>> type(features[0])
        <class 'numpy.ndarray'>
        >>> features[0].shape
        (1024,)
    """
    # Convert single SMILES to list for consistent processing
    if isinstance(smiles, str):
        smiles = [smiles]
        single_input = True
    else:
        single_input = False
    
    # Create featurizer
    featurizer = dc.feat.CircularFingerprint(size=size, radius=radius)
    
    # Compute features
    features = featurizer.featurize(smiles)
    
    # Convert to numpy array if not already
    if not isinstance(features, np.ndarray):
        features = np.array(features)
    
    # Return single array if single input was provided
    if single_input:
        return features[0]
    
    return features


def compute_molformer_features(smiles: Union[str, List[str]], 
                               model_name: str = "ibm/MoLFormer-XL-both-10pct",
                               batch_size: int = 32,
                               device: Optional[str] = None) -> np.ndarray:
    """
    Compute molecular features using MolFormer transformer model.
    
    Args:
        smiles: Single SMILES string or list of SMILES strings
        model_name: Name of the MolFormer model (default: "ibm/MoLFormer-XL-both-10pct")
        batch_size: Batch size for processing (default: 32)
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        numpy.ndarray: Feature array(s). If single SMILES, returns 1D array.
                      If list of SMILES, returns 2D array of shape (n_smiles, feature_dim).
    """
    # Convert single SMILES to list for consistent processing
    if isinstance(smiles, str):
        smiles = [smiles]
        single_input = True
    else:
        single_input = False
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model and tokenizer
    print(f"Loading MolFormer model: {model_name}")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    
    # Get model short name for progress bar
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    
    print(f"  Using feature extraction: pooler_output")
    
    # Process in batches
    all_features = []
    for i in tqdm(range(0, len(smiles), batch_size), desc=f"Computing {model_short} features"):
        batch_smiles = smiles[i:i+batch_size]
        
        # Tokenize with padding
        inputs = tokenizer(batch_smiles, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get features using pooler_output
        with torch.no_grad():
            outputs = model(**inputs)
            batch_features = outputs.pooler_output.cpu().numpy()
        
        all_features.append(batch_features)
    
    # Concatenate all batches
    features = np.concatenate(all_features, axis=0)
    
    # Return single array if single input was provided
    if single_input:
        return features[0]
    
    return features


def compute_chemberta_features(smiles: Union[str, List[str]], 
                               model_name: str,
                               batch_size: int = 32,
                               device: Optional[str] = None) -> np.ndarray:
    """
    Compute molecular features using ChemBERTa transformer model (RoBERTa-based).
    
    Args:
        smiles: Single SMILES string or list of SMILES strings
        model_name: Name of the ChemBERTa model (e.g., "DeepChem/ChemBERTa-77M-MTR", 
                   "DeepChem/ChemBERTa-77M-MLM")
        batch_size: Batch size for processing (default: 32)
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        numpy.ndarray: Feature array(s). If single SMILES, returns 1D array.
                      If list of SMILES, returns 2D array of shape (n_smiles, feature_dim).
    """
    # Convert single SMILES to list for consistent processing
    if isinstance(smiles, str):
        smiles = [smiles]
        single_input = True
    else:
        single_input = False
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model and tokenizer
    print(f"Loading ChemBERTa model: {model_name}")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    
    # Get model short name for progress bar
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    
    print(f"  Using feature extraction: last hidden state (mean pooling)")
    
    # Process in batches
    all_features = []
    for i in tqdm(range(0, len(smiles), batch_size), desc=f"Computing {model_short} features"):
        batch_smiles = smiles[i:i+batch_size]
        
        # Tokenize with truncation and padding
        # ChemBERTa is RoBERTa-based, which doesn't use token_type_ids
        inputs = tokenizer(batch_smiles, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get features using mean pooling of last hidden state
        with torch.no_grad():
            outputs = model(**inputs)
            batch_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        all_features.append(batch_features)
    
    # Concatenate all batches
    features = np.concatenate(all_features, axis=0)
    
    # Return single array if single input was provided
    if single_input:
        return features[0]
    
    return features


def compute_transformer_features(smiles: Union[str, List[str]], 
                                 model_name: str,
                                 batch_size: int = 32,
                                 device: Optional[str] = None) -> np.ndarray:
    """
    Compute molecular features using transformer models.
    Routes to appropriate function based on model type.
    
    Args:
        smiles: Single SMILES string or list of SMILES strings
        model_name: Name of the transformer model
        batch_size: Batch size for processing (default: 32)
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        numpy.ndarray: Feature array(s). If single SMILES, returns 1D array.
                      If list of SMILES, returns 2D array of shape (n_smiles, feature_dim).
    """
    # Route to appropriate function based on model type
    if "ChemBERTa" in model_name:
        return compute_chemberta_features(smiles, model_name, batch_size, device)
    elif "MolFormer" in model_name:
        return compute_molformer_features(smiles, model_name, batch_size, device)
    else:
        # Default to MolFormer-style processing
        return compute_molformer_features(smiles, model_name, batch_size, device)


def compute_features_for_dataset(input_file: str, 
                                 smiles_col: str = "Normalized_SMILES_combined",
                                 size: int = 1024, radius: int = 2,
                                 batch_size: int = 32,
                                 device: Optional[str] = None):
    """
    Compute all molecular features (ECFP, MolFormer, ChemBERTa-MTR, ChemBERTa-MLM) 
    for SMILES in a dataset and save with metadata columns.
    
    Args:
        input_file: Path to input CSV file containing SMILES
        smiles_col: Name of the column containing SMILES strings
        size: Size of the fingerprint for ECFP (default: 1024)
        radius: Radius of the circular fingerprint for ECFP (default: 2)
        batch_size: Batch size for transformer models (default: 32)
        device: Device to use for transformer models ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        List[str]: List of paths to all generated feature files.
    """
    # Load data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} compounds from {input_file}")
    
    # Check required columns
    required_cols = ["LTKBID", "CompoundName", smiles_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")
    
    # Get SMILES list
    smiles_list = df[smiles_col].tolist()
    
    # Filter out None/NaN values for processing
    valid_indices = [i for i, s in enumerate(smiles_list) if pd.notna(s) and s is not None and str(s).strip() != '']
    valid_smiles = [smiles_list[i] for i in valid_indices]
    
    print(f"Computing features for {len(valid_smiles)} valid SMILES...")
    print(f"  Computing all feature types: ECFP, MolFormer, ChemBERTa-MTR, and ChemBERTa-MLM...")
    
    output_files = []
    
    # Define all models to compute
    models_to_compute = [
        ("ECFP", "ECFP", None),
        ("MolFormer", "Transformer", "ibm/MoLFormer-XL-both-10pct"),
        ("ChemBERTa-MTR", "Transformer", "DeepChem/ChemBERTa-77M-MTR"),
        ("ChemBERTa-MLM", "Transformer", "DeepChem/ChemBERTa-77M-MLM"),
    ]
    
    for model_display_name, model_type, model_name in models_to_compute:
        print("\n" + "="*50)
        print(f"Computing {model_display_name} features...")
        print("="*50)
        
        model_file = _compute_single_featurizer(
            input_file=input_file,
            output_file=None,  # Auto-generate
            smiles_col=smiles_col,
            featurizer_type=model_type,
            size=size,
            radius=radius,
            transformer_model_name=model_name,  # Only used for transformer models, None for ECFP
            batch_size=batch_size,
            device=device,
            df=df,
            smiles_list=smiles_list,
            valid_indices=valid_indices,
            valid_smiles=valid_smiles
        )
        output_files.append(model_file)
    
    print("\n" + "="*50)
    print("✅ All feature files generated successfully!")
    for i, (model_display_name, _, _) in enumerate(models_to_compute):
        print(f"  {model_display_name}: {output_files[i]}")
    print("="*50)
    
    return output_files


def _compute_single_featurizer(input_file: str, output_file: str, 
                               smiles_col: str, featurizer_type: str,
                               size: int, radius: int, transformer_model_name: Optional[str],
                               batch_size: int, device: Optional[str],
                               df: pd.DataFrame, smiles_list: list, 
                               valid_indices: list, valid_smiles: list) -> str:
    """
    Internal helper function to compute features for a single featurizer type.
    """
    if featurizer_type.upper() == "ECFP":
        print(f"  Fingerprint size: {size}, radius: {radius}")
        # Compute ECFP features one by one
        features_list = []
        for smiles in tqdm(valid_smiles, desc="Computing ECFP features"):
            try:
                features = compute_ECFP(smiles, size=size, radius=radius)
                features_list.append(features)
            except Exception as e:
                print(f"Warning: Failed to compute features for SMILES '{smiles}': {e}")
                features_list.append(None)
        
        # Create full feature array with None for invalid SMILES
        all_features = [None] * len(smiles_list)
        for idx, valid_idx in enumerate(valid_indices):
            all_features[valid_idx] = features_list[idx]
        
        # Get feature dimension
        if len(features_list) > 0 and features_list[0] is not None:
            feature_dim = len(features_list[0])
        else:
            feature_dim = size
        
        # Convert to numpy array (use zeros for invalid SMILES)
        feature_arrays = []
        for feat in all_features:
            if feat is not None:
                feature_arrays.append(feat)
            else:
                feature_arrays.append(np.zeros(feature_dim))
        
        features_array = np.array(feature_arrays)
        
    elif featurizer_type.upper() in ["MOLFORMER", "TRANSFORMER"]:
        if transformer_model_name is None:
            raise ValueError("transformer_model_name must be provided for transformer models")
        print(f"  Model: {transformer_model_name}, batch size: {batch_size}")
        # Compute transformer features in batches
        try:
            valid_features = compute_transformer_features(valid_smiles, model_name=transformer_model_name, 
                                                         batch_size=batch_size, device=device)
            feature_dim = valid_features.shape[1]
        except Exception as e:
            print(f"Error computing transformer features: {e}")
            raise
        
        # Create full feature array with zeros for invalid SMILES
        features_array = np.zeros((len(smiles_list), feature_dim))
        for idx, valid_idx in enumerate(valid_indices):
            features_array[valid_idx] = valid_features[idx]
        
        # For statistics, all valid SMILES were processed successfully
        all_features = None  # Not used for transformer stats
        successful = len(valid_smiles)
        
    else:
        raise ValueError(f"Unknown featurizer type: {featurizer_type}. Must be 'ECFP' or 'Transformer'")
    
    
    # Add feature columns
    feature_dim = features_array.shape[1]
    feature_cols = [f'feature_{i}' for i in range(feature_dim)]
    feature_df = pd.DataFrame(features_array, columns=feature_cols)
    output_df = pd.concat([df, feature_df], axis=1)
    
    # Generate output filename if not provided
    if output_file is None:
        input_dir = os.path.dirname(input_file)
        input_basename = os.path.basename(input_file)
        input_name = os.path.splitext(input_basename)[0]
        if featurizer_type.upper() == "ECFP":
            output_file = os.path.join(input_dir, f"{input_name}_ECFP_{size}_{radius}.csv")
        elif featurizer_type.upper() in ["MOLFORMER", "TRANSFORMER"]:
            if transformer_model_name is None:
                raise ValueError("transformer_model_name must be provided for transformer models")
            model_short = transformer_model_name.split("/")[-1] if "/" in transformer_model_name else transformer_model_name
            # Use specific names for known models
            if "ChemBERTa-77M-MTR" in transformer_model_name:
                output_file = os.path.join(input_dir, f"{input_name}_ChemBERTa-MTR.csv")
            elif "ChemBERTa-77M-MLM" in transformer_model_name:
                output_file = os.path.join(input_dir, f"{input_name}_ChemBERTa-MLM.csv")
            elif "MolFormer" in transformer_model_name:
                output_file = os.path.join(input_dir, f"{input_name}_MolFormer_{model_short}.csv")
            else:
                output_file = os.path.join(input_dir, f"{input_name}_{model_short}.csv")
    
    # Save features
    output_df.to_csv(output_file, index=False)
    print(f"\nFeatures saved to: {output_file}")
    print(f"  Shape: {output_df.shape}")
    
    # Statistics
    if featurizer_type.upper() == "ECFP":
        successful = sum(1 for f in all_features if f is not None)
        failed = len(all_features) - successful
    else:  # Transformer models (MolFormer, ChemBERTa, etc.)
        failed = len(smiles_list) - successful
    
    print(f"\nSummary:")
    print(f"  Successful: {successful}/{len(smiles_list)} ({successful/len(smiles_list)*100:.1f}%)")
    print(f"  Failed: {failed}/{len(smiles_list)} ({failed/len(smiles_list)*100:.1f}%)")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute all molecular features (ECFP, MolFormer, ChemBERTa-MTR, ChemBERTa-MLM) from SMILES")
    parser.add_argument("--input_file", "-i", type=str, 
                       default="/scratch/work/masooda1/datasets/downstream_datasets/DILIRank_v2/DILIrank_2.0_normalized.csv",
                       help="Path to input CSV file containing SMILES")
    parser.add_argument("--smiles_col", "-s", type=str, default="Normalized_SMILES_combined",
                       help="Name of the column containing SMILES strings (default: Normalized_SMILES_combined)")
    parser.add_argument("--size", type=int, default=1024,
                       help="Size of the fingerprint for ECFP (default: 1024)")
    parser.add_argument("--radius", type=int, default=2,
                       help="Radius of the circular fingerprint for ECFP (default: 2)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for transformer models (default: 32)")
    parser.add_argument("--device", type=str, default=None,
                       choices=["cuda", "cpu", None],
                       help="Device to use for transformer models ('cuda', 'cpu', or None for auto-detect)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    try:
        compute_features_for_dataset(
            args.input_file, 
            args.smiles_col,
            args.size,
            args.radius,
            args.batch_size,
            args.device
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
