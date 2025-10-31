"""
Utilities for loading and integrating pretrained VAE/AE encoders into multimodal models.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any
import os
import sys

# Add path to access VAE models
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))

try:
    from vae import JUMPVAE, VAEEncoder
except ImportError:
    print("Warning: Could not import VAE models. Make sure models/vae.py is available.")
    JUMPVAE = None
    VAEEncoder = None


class PretrainedEncoder(nn.Module):
    """
    Wrapper class for pretrained encoders that can be used in multimodal models.
    
    This class extracts the encoder part from a pretrained VAE/AE and provides
    a clean interface for use in downstream tasks.
    """
    
    def __init__(
        self, 
        encoder: nn.Module, 
        output_dim: int,
        freeze: bool = False,
        use_latent_layer: bool = True
    ):
        """
        Initialize pretrained encoder wrapper.
        
        Args:
            encoder: The pretrained encoder module
            output_dim: Expected output dimension
            freeze: Whether to freeze the encoder weights
            use_latent_layer: Whether to include the final latent projection layer
        """
        super().__init__()
        
        self.encoder = encoder
        self.output_dim = output_dim
        self.use_latent_layer = use_latent_layer
        
        # Freeze encoder if requested
        if freeze:
            self.freeze_encoder()
        
        print(f"PretrainedEncoder initialized with output_dim={output_dim}, frozen={freeze}")
    
    def freeze_encoder(self):
        """Freeze all parameters in the encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder weights frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze all parameters in the encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder weights unfrozen")
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass through the pretrained encoder.
        
        Args:
            x: Input tensor
            **kwargs: Additional arguments (for compatibility)
            
        Returns:
            Encoded features
        """
        # Ensure encoder is on same device as input
        if x.device != next(self.encoder.parameters()).device:
            self.encoder = self.encoder.to(x.device)
            
        if hasattr(self.encoder, 'encoder') and hasattr(self.encoder, 'fc_latent'):
            # This is a VAEEncoder - get features from the main encoder layers
            h = self.encoder.encoder(x)
            
            if self.use_latent_layer:
                # Use the latent projection layer (fc_latent for AE, fc_mu for VAE)
                if hasattr(self.encoder, 'fc_latent'):
                    # AE mode
                    return self.encoder.fc_latent(h)
                elif hasattr(self.encoder, 'fc_mu'):
                    # VAE mode - use mean as deterministic encoding
                    return self.encoder.fc_mu(h)
                else:
                    return h
            else:
                # Use features before latent projection
                return h
        else:
            # Generic encoder
            return self.encoder(x)


def load_pretrained_encoder(
    checkpoint_path: str,
    freeze: bool = False,
    device: Optional[torch.device] = None,
    load_weights: bool = True
) -> PretrainedEncoder:
    """
    Load a pretrained encoder from a VAE/AE checkpoint.
    
    Always uses the 128-dimensional latent layer representation (best practice).
    
    Args:
        checkpoint_path: Path to the checkpoint file (.ckpt)
        freeze: Whether to freeze the encoder weights
        device: Device to load the model on
        load_weights: If True, load pretrained weights. If False, only load architecture.
        
    Returns:
        PretrainedEncoder wrapper around the loaded encoder (128-dim output)
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract hyperparameters
        if 'hyper_parameters' in checkpoint:
            hparams = checkpoint['hyper_parameters']
        else:
            raise RuntimeError("Hyperparameters not found in checkpoint")
        
        # Create model instance
        if JUMPVAE is None:
            raise RuntimeError("VAE model class not available. Check imports.")
        
        model = JUMPVAE(**hparams)
        
        if load_weights:
            # Setting 2: Load pretrained weights
            model.load_state_dict(checkpoint['state_dict'])
            print(f"  - Loaded pretrained weights from checkpoint")
        else:
            # Setting 1: Only architecture, keep random initialization
            print(f"  - Using architecture only (random weights)")
        
        model.to(device)
        
        # Extract encoder
        encoder = model.encoder
        
        # Always use 128-dimensional latent representation
        output_dim = 128
        
        print(f"Successfully loaded pretrained encoder:")
        print(f"  - Model type: {hparams.get('model_type', 'unknown')}")
        print(f"  - Input dim: {hparams.get('input_dim', 'unknown')}")
        print(f"  - Latent dim: {hparams.get('latent_dim', 'unknown')}")
        print(f"  - Architecture: {hparams.get('encoder_hidden_dims', 'unknown')}")
        print(f"  - Output dim: {output_dim}")
        
        return PretrainedEncoder(
            encoder=encoder,
            output_dim=output_dim,
            freeze=freeze,
            use_latent_layer=True  # Always use latent layer (128-dim)
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")


def create_encoder_from_config(
    checkpoint_path: Optional[str] = None,
    encoder_config: Optional[Dict[str, Any]] = None,
    freeze: bool = False,
    device: Optional[torch.device] = None
) -> Optional[PretrainedEncoder]:
    """
    Create an encoder either from a checkpoint or from config.
    
    Always uses 128-dimensional latent layer representation.
    
    Args:
        checkpoint_path: Path to pretrained checkpoint (takes priority)
        encoder_config: Configuration for creating a new encoder
        freeze: Whether to freeze encoder weights
        device: Device to load on
        
    Returns:
        PretrainedEncoder if checkpoint_path provided, None otherwise
    """
    if checkpoint_path is not None:
        return load_pretrained_encoder(
            checkpoint_path=checkpoint_path,
            freeze=freeze,
            device=device
        )
    else:
        return None


def print_encoder_info(encoder: PretrainedEncoder):
    """Print information about a pretrained encoder (always 128-dimensional)."""
    print(f"\nEncoder Information:")
    print(f"  - Output dimension: {encoder.output_dim} (always 128-dim latent)")
    print(f"  - Frozen: {not next(encoder.encoder.parameters()).requires_grad}")
    print(f"  - Parameters: {sum(p.numel() for p in encoder.encoder.parameters()):,}")
    print(f"  - Trainable parameters: {sum(p.numel() for p in encoder.encoder.parameters() if p.requires_grad):,}")


# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing
    print("Pretrained encoder utilities loaded successfully")
