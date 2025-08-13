import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict, Any
import wandb


class VAEEncoder(nn.Module):
    """Encoder network for VAE/AE with configurable architecture."""
    
    def __init__(
        self,
        input_dim: int = 4000,  # JUMP CP features
        hidden_dims: List[int] = [1024, 512, 256],
        latent_dim: int = 128,
        dropout: float = 0.1,
        norm_type: Optional[str] = "batchnorm",  # "batchnorm", "layernorm", None
        model_type: str = "vae",  # "vae" or "ae"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model_type = model_type
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Normalization
            if norm_type == "batchnorm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm_type == "layernorm":
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        if self.model_type == "vae":
            # Mean and log variance layers for VAE
            self.fc_mu = nn.Linear(prev_dim, latent_dim)
            self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        else:  # AE mode
            # Single deterministic latent layer for AE
            self.fc_latent = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log variance (VAE) or latent and zeros (AE)."""
        h = self.encoder(x)
        
        if self.model_type == "vae":
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
        else:  # AE mode
            latent = self.fc_latent(h)
            # Return zeros for logvar to maintain interface compatibility
            zeros = torch.zeros_like(latent)
            return latent, zeros


class VAEDecoder(nn.Module):
    """Decoder network for VAE/AE with configurable architecture."""
    
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: List[int] = [256, 512, 1024],
        output_dim: int = 4000,
        dropout: float = 0.1,
        norm_type: Optional[str] = "batchnorm",
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build decoder layers (reverse of encoder)
        layers = []
        prev_dim = latent_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Normalization
            if norm_type == "batchnorm":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm_type == "layernorm":
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        
        # Final output layer (no activation for regression)
        self.fc_out = nn.Linear(prev_dim, output_dim)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass from latent to reconstruction."""
        h = self.decoder(z)
        return self.fc_out(h)


class JUMPVAE(pl.LightningModule):
    """Variational Autoencoder or Autoencoder for JUMP Cell Painting dataset."""
    
    def __init__(
        self,
        input_dim: int = 4000,
        encoder_hidden_dims: List[int] = [1024, 512, 256],
        decoder_hidden_dims: List[int] = [256, 512, 1024],
        latent_dim: int = 128,
        dropout: float = 0.1,
        norm_type: Optional[str] = "batchnorm",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        beta: float = 1.0,  # KL divergence weight
        model_type: str = "vae",  # "vae" or "ae"
        optimizer: str = "adamw",  # "adamw", "adam", "sgd"
        # Learning rate scheduler settings
        scheduler_type: str = "plateau",  # "plateau" or "cosine"
        T_max: int = 10,                # Used if scheduler_type == "cosine"
        eta_min: float = 0.0,           # Used if scheduler_type == "cosine"
        warmup_epochs: int = 0,         # Used if scheduler_type == "cosine_with_warmup"
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_type = model_type
        
        # Model components
        self.encoder = VAEEncoder(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            latent_dim=latent_dim,
            dropout=dropout,
            norm_type=norm_type,
            model_type=model_type,
        )
        
        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=input_dim,
            dropout=dropout,
            norm_type=norm_type,
        )
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta = beta if model_type == "vae" else 0.0  # No KL for AE
        
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE. For AE, just return mu."""
        if self.model_type == "vae":
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:  # AE mode
            return mu  # Deterministic latent code
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE/AE."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def loss_function(
        self, 
        x_recon: torch.Tensor, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE/AE loss (reconstruction + optional KL divergence)."""
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        if self.model_type == "vae":
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        else:  # AE mode
            # No KL divergence for standard autoencoder
            kl_loss = torch.tensor(0.0, device=x.device)
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch
        x_recon, mu, logvar = self.forward(x)
        
        total_loss, recon_loss, kl_loss = self.loss_function(x_recon, x, mu, logvar)
        
        # Calculate MAE for tracking
        mae = F.l1_loss(x_recon, x, reduction='mean')
        
        # Log metrics
        self.log('train/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/recon_loss', recon_loss, on_step=False, on_epoch=True)
        if self.model_type == "vae":
            self.log('train/kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('train/mae', mae, on_step=False, on_epoch=True)
        self.log('train/lr', self.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=False)
    
        return total_loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        x = batch
        x_recon, mu, logvar = self.forward(x)
        
        total_loss, recon_loss, kl_loss = self.loss_function(x_recon, x, mu, logvar)
        
        # Calculate MAE
        mae = F.l1_loss(x_recon, x, reduction='mean')
        
        # Log validation metrics
        self.log('val/total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/recon_loss', recon_loss, on_step=False, on_epoch=True)
        if self.model_type == "vae":
            self.log('val/kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('val/mae', mae, on_step=False, on_epoch=True)
        
        return {
            'val_total_loss': total_loss,
            'val_recon_loss': recon_loss,
            'val_kl_loss': kl_loss,
            'val_mae': mae,
        }
    
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step - similar to validation step but for testing."""
        x = batch
        x_recon, mu, logvar = self.forward(x)
        
        total_loss, recon_loss, kl_loss = self.loss_function(x_recon, x, mu, logvar)
        
        # Calculate MAE
        mae = F.l1_loss(x_recon, x, reduction='mean')
        
        # Log test metrics
        self.log('test/total_loss', total_loss, on_step=False, on_epoch=True)
        self.log('test/recon_loss', recon_loss, on_step=False, on_epoch=True)
        if self.model_type == "vae":
            self.log('test/kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('test/mae', mae, on_step=False, on_epoch=True)
        
        return {
            'test_total_loss': total_loss,
            'test_recon_loss': recon_loss,
            'test_kl_loss': kl_loss,
            'test_mae': mae,
        }
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        if self.hparams.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.hparams.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
            )
        
        elif self.hparams.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.hparams.optimizer}. Choose from: adamw, adam, sgd")
        
        # Select LR scheduler
        scheduler_cfg = {}
        if self.hparams.scheduler_type == "plateau":
            print("Using ReduceLROnPlateau scheduler")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=10,
                verbose=True,
            )
            scheduler_cfg = {
                "scheduler": self.scheduler,
                "monitor": "val/total_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        elif self.hparams.scheduler_type == "cosine":
            print("Using CosineAnnealingLR scheduler")
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.hparams.T_max,
                eta_min=self.hparams.eta_min,
            )
            scheduler_cfg = {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        elif self.hparams.scheduler_type == "cosine_with_warmup":
            print("Using CosineAnnealingWarmRestarts scheduler")
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.hparams.warmup_epochs,
                T_mult=1,
                eta_min=self.hparams.eta_min,
            )
            scheduler_cfg = {
                "scheduler": self.scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        else:
            raise ValueError(
                f"Unknown scheduler_type '{self.hparams.scheduler_type}'. Choose 'plateau' or 'cosine'."
            )
        
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": scheduler_cfg,
        }
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def generate(self, num_samples: int = 100) -> torch.Tensor:
        """Generate new samples from the learned distribution."""
        with torch.no_grad():
            if self.model_type == "vae":
                # Sample from learned distribution for VAE
                z = torch.randn(num_samples, self.hparams.latent_dim, device=self.device)
            else:
                # For AE, we need to sample from the empirical latent distribution
                # This is a placeholder - in practice you'd need actual data to sample from
                print("Warning: AE generation requires sampling from empirical latent distribution")
                z = torch.randn(num_samples, self.hparams.latent_dim, device=self.device)
            
            samples = self.decode(z)
        return samples


# Utility function to create model with different configurations
def create_vae_model(
    input_dim: int = 4000,
    architecture: str = "vanilla",  # "vanilla", "medium", "large"
    latent_dim: int = 128,
    dropout: float = 0.1,
    norm_type: str = "batchnorm",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    beta: float = 1.0,
    model_type: str = "vae",  # "vae" or "ae"
    **kwargs
) -> JUMPVAE:
    """Create VAE/AE model with predefined architectures."""
    
    if architecture == "vanilla":
        encoder_dims = [512, 256]
        decoder_dims = [256, 512]
    elif architecture == "medium":
        encoder_dims = [1024, 512, 256]
        decoder_dims = [256, 512, 1024]
    elif architecture == "large":
        encoder_dims = [2048, 1024, 512, 256]
        decoder_dims = [256, 512, 1024, 2048]
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from: vanilla, medium, large")
    
    return JUMPVAE(
        input_dim=input_dim,
        encoder_hidden_dims=encoder_dims,
        decoder_hidden_dims=decoder_dims,
        latent_dim=latent_dim,
        dropout=dropout,
        norm_type=norm_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta=beta,
        model_type=model_type,
        **kwargs
    ) 