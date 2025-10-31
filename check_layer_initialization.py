#!/usr/bin/env python3

import sys
import os
sys.path.append('/scratch/work/masooda1/Multi_Modal_Contrastive')

import torch
from mocop.model import LightningGGNN

def check_layer_initialization():
    checkpoint_path = "/scratch/work/masooda1/Multi_Modal_Contrastive/downstream/DILI_finetuning/extracted_molecular_encoders/molecular_encoder.ckpt"
    
    print("=" * 80)
    print("CHECKING LAYER INITIALIZATION")
    print("=" * 80)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Load model
    print("\nLoading model...")
    multimodal = LightningGGNN.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        n_edge=1,
        in_dim=75,
        n_conv=6,
        fc_dims=[1024, 128, 1],  # Added final layer for DILI (binary classification)
        p_dropout=0.1,
        freeze=False
    )
    
    print("Model loaded successfully!")
    
    # Get model state dict
    model_state_dict = multimodal.state_dict()
    checkpoint_state_dict = checkpoint['state_dict']
    
    print(f"\nModel has {len(model_state_dict)} parameters")
    print(f"Checkpoint has {len(checkpoint_state_dict)} parameters")
    
    # Check which layers are loaded vs randomly initialized
    print("\n" + "=" * 80)
    print("LAYER ANALYSIS")
    print("=" * 80)
    
    loaded_layers = []
    missing_layers = []
    extra_layers = []
    
    for name, param in model_state_dict.items():
        if name in checkpoint_state_dict:
            # Check if parameters are the same (loaded from checkpoint)
            if torch.equal(param, checkpoint_state_dict[name]):
                loaded_layers.append(name)
            else:
                print(f"⚠️  {name}: Present in both but different values (might be remapped)")
        else:
            missing_layers.append(name)
    
    for name in checkpoint_state_dict.keys():
        if name not in model_state_dict:
            extra_layers.append(name)
    
    print(f"\n✅ LOADED LAYERS ({len(loaded_layers)}):")
    for layer in sorted(loaded_layers):
        print(f"  - {layer}")
    
    print(f"\n🆕 RANDOMLY INITIALIZED LAYERS ({len(missing_layers)}):")
    for layer in sorted(missing_layers):
        print(f"  - {layer}")
    
    print(f"\n📦 EXTRA LAYERS IN CHECKPOINT ({len(extra_layers)}):")
    for layer in sorted(extra_layers):
        print(f"  - {layer}")
    
    # Detailed analysis by layer type
    print("\n" + "=" * 80)
    print("DETAILED LAYER ANALYSIS")
    print("=" * 80)
    
    conv_layers = [k for k in model_state_dict.keys() if 'conv_layers' in k]
    fc_layers = [k for k in model_state_dict.keys() if 'fc_layers' in k]
    
    print(f"\n🔗 CONVOLUTION LAYERS ({len(conv_layers)}):")
    for layer in sorted(conv_layers):
        status = "✅ Loaded" if layer in checkpoint_state_dict else "🆕 Random"
        print(f"  {status}: {layer}")
    
    print(f"\n🧠 FULLY CONNECTED LAYERS ({len(fc_layers)}):")
    for layer in sorted(fc_layers):
        status = "✅ Loaded" if layer in checkpoint_state_dict else "🆕 Random"
        print(f"  {status}: {layer}")
    
    # Check parameter shapes
    print(f"\n" + "=" * 80)
    print("PARAMETER SHAPES")
    print("=" * 80)
    
    for name, param in model_state_dict.items():
        if name in checkpoint_state_dict:
            checkpoint_shape = checkpoint_state_dict[name].shape
            model_shape = param.shape
            if checkpoint_shape == model_shape:
                print(f"✅ {name}: {model_shape} (loaded)")
            else:
                print(f"⚠️  {name}: {model_shape} vs {checkpoint_shape} (shape mismatch)")
        else:
            print(f"🆕 {name}: {param.shape} (random)")

if __name__ == "__main__":
    check_layer_initialization()
