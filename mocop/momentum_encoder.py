import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from model import CellLineTripleInputEncoderSoftHard


class CellLineTripleInputEncoderMomentum(CellLineTripleInputEncoderSoftHard):
    """
    CellLineTripleInputEncoder with momentum encoders for encoder_b and encoder_c.
    
    This class implements momentum encoders (like BYOL, MoCo, DINO) where:
    - encoder_a: Regular encoder (no momentum)
    - encoder_b_momentum: Slow-moving copy of encoder_b for stable targets
    - encoder_c_momentum: Slow-moving copy of encoder_c for stable targets
    
    The momentum encoders are used to compute YY (targets) while the main encoders
    compute XY (predictions), providing stable targets for contrastive learning.
    """
    
    def __init__(self, *args, momentum=0.999, center_momentum=0.9, use_centering=True, 
                 temperature_momentum_morph=None, temperature_momentum_genomic=None, update_teacher_encoders=False, **kwargs):
        """
        Initialize with momentum encoders and optional centering mechanism.
        
        Args:
            momentum (float): Momentum coefficient for EMA updates (0.999 recommended)
            center_momentum (float): Momentum coefficient for center updates (0.9 recommended)
            use_centering (bool): Whether to use DINO-style centering mechanism
            temperature_momentum_morph (float): Temperature for morphological momentum encoder
            temperature_momentum_genomic (float): Temperature for genomic momentum encoder
        """
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.center_momentum = center_momentum
        self.use_centering = use_centering
        self.update_teacher_encoders = update_teacher_encoders
        
        # Set separate momentum temperatures for each modality
        self.temperature_momentum_morph = temperature_momentum_morph if temperature_momentum_morph is not None else self.temperature_momentum
        self.temperature_momentum_genomic = temperature_momentum_genomic if temperature_momentum_genomic is not None else self.temperature_momentum
        
        print(f"Temperature main: {self.temperature_main}")
        print(f"Temperature momentum (morphological): {self.temperature_momentum_morph}")
        print(f"Temperature momentum (genomic): {self.temperature_momentum_genomic}")
        
        # Initialize momentum encoders and centers after parent initialization
        self._init_momentum_encoders()
        if self.use_centering:
            self._init_centers()
    
    def _init_momentum_encoders(self):
        """Initialize momentum encoders as copies of the main encoders."""
        import copy
        
        # Create momentum copies of encoder_b and encoder_c
        if self.encoder_b is not None:
            self.encoder_b_momentum = copy.deepcopy(self.encoder_b)
            # Disable gradients for momentum encoder
            for p in self.encoder_b_momentum.parameters():
                p.requires_grad = False
        
        if self.encoder_c is not None:
            self.encoder_c_momentum = copy.deepcopy(self.encoder_c)
            # Disable gradients for momentum encoder
            for p in self.encoder_c_momentum.parameters():
                p.requires_grad = False
        
        # Create momentum copies of embeddings
        if hasattr(self, 'cell_embedding'):
            self.cell_embedding_momentum = copy.deepcopy(self.cell_embedding)
            for p in self.cell_embedding_momentum.parameters():
                p.requires_grad = False
        
        if hasattr(self, 'dose_embedding'):
            self.dose_embedding_momentum = copy.deepcopy(self.dose_embedding)
            for p in self.dose_embedding_momentum.parameters():
                p.requires_grad = False
        
        if hasattr(self, 'time_embedding'):
            self.time_embedding_momentum = copy.deepcopy(self.time_embedding)
            for p in self.time_embedding_momentum.parameters():
                p.requires_grad = False
    
    def _init_centers(self):
        """Initialize center vectors for DINO-style centering mechanism."""
        # Centers for morphological and genomic momentum encoders
        self.register_buffer('center_morph', torch.zeros(128))  # 128-dim embeddings
        self.register_buffer('center_genomic', torch.zeros(128))  # 128-dim embeddings
    
    def _update_momentum_encoders(self):
        """Update momentum encoders using exponential moving average."""
        # Update encoder_b momentum
        if hasattr(self, 'encoder_b_momentum') and self.encoder_b is not None:
            for param_q, param_k in zip(self.encoder_b.parameters(), self.encoder_b_momentum.parameters()):
                param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data
        
        # Update encoder_c momentum
        if hasattr(self, 'encoder_c_momentum') and self.encoder_c is not None:
            for param_q, param_k in zip(self.encoder_c.parameters(), self.encoder_c_momentum.parameters()):
                param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data
        
        # Update embedding momentums
        if hasattr(self, 'cell_embedding_momentum'):
            for param_q, param_k in zip(self.cell_embedding.parameters(), self.cell_embedding_momentum.parameters()):
                param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data
        
        if hasattr(self, 'dose_embedding_momentum'):
            for param_q, param_k in zip(self.dose_embedding.parameters(), self.dose_embedding_momentum.parameters()):
                param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data
        
        if hasattr(self, 'time_embedding_momentum'):
            for param_q, param_k in zip(self.time_embedding.parameters(), self.time_embedding_momentum.parameters()):
                param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data
    
    def _update_centers(self, emb_b_momentum, emb_c_momentum):
        """Update center vectors using exponential moving average of momentum encoder outputs."""
        # Update morphological center
        if emb_b_momentum is not None:
            batch_center = emb_b_momentum.mean(dim=0)
            self.center_morph.data = self.center_momentum * self.center_morph.data + (1 - self.center_momentum) * batch_center
        
        # Update genomic center
        if emb_c_momentum is not None and len(emb_c_momentum) > 0:
            batch_center = emb_c_momentum.mean(dim=0)
            self.center_genomic.data = self.center_momentum * self.center_genomic.data + (1 - self.center_momentum) * batch_center
    
    def forward(self, x_a=None, x_b=None, x_c=None, cell_indices=None, doses=None, times=None, batch_indices=None, use_momentum=False):
        """
        Forward pass with optional momentum encoders.
        
        Args:
            use_momentum (bool): If True, use momentum encoders for encoder_b and encoder_c
        """
        # 1. Encode molecular features (always use main encoder_a)
        emb_a = None
        if x_a is not None:
            emb_a = F.normalize(self.encoder_a(x_a))  # [B, 128]

        # 2. Encode morphological features
        emb_b = None
        if x_b is not None:
            # Check for valid samples (not all -1s)
            b_valid = ~torch.all(x_b == -1, dim=1)
            
            if b_valid.any():
                if use_momentum and hasattr(self, 'encoder_b_momentum'):
                    # Process only valid samples through momentum encoder
                    emb_b = F.normalize(self.encoder_b_momentum(x_b[b_valid]))  # [n_valid, 128]
                else:
                    # Process only valid samples through main encoder
                    emb_b = F.normalize(self.encoder_b(x_b[b_valid]))  # [n_valid, 128]

        # 3. Encode genomic features
        emb_c = None
        if x_c is not None and len(x_c) > 0:
            # Select embeddings based on momentum flag
            if use_momentum:
                cell_emb = getattr(self, 'cell_embedding_momentum', self.cell_embedding)
                dose_emb = getattr(self, 'dose_embedding_momentum', self.dose_embedding)
                time_emb = getattr(self, 'time_embedding_momentum', self.time_embedding)
                encoder_c = getattr(self, 'encoder_c_momentum', self.encoder_c)
            else:
                cell_emb = self.cell_embedding
                dose_emb = self.dose_embedding
                time_emb = self.time_embedding
                encoder_c = self.encoder_c
            
            # Concatenate genomic features with conditional embeddings
            cell_embeddings = cell_emb(cell_indices.long())
            dose_embeddings = dose_emb(doses.long())
            time_embeddings = time_emb(times.long())
            
            x_c_combined = torch.cat([
                x_c, cell_embeddings, dose_embeddings, time_embeddings
            ], dim=-1)
               
            emb_c = F.normalize(encoder_c(x_c_combined))  # [total_conditions, 128]

        # 4. Calculate contrastive logits
        logits_ab = None
        if emb_b is not None:
            logits_ab = torch.matmul(emb_a, emb_b.T)  # [B, n_valid]
        
        logits_ac = None
        if emb_c is not None and len(emb_c) > 0:
            emb_a_expanded = emb_a[batch_indices]
            logits_ac = torch.matmul(emb_a_expanded, emb_c.T)  # [total_conditions, total_conditions]
        
        return logits_ab, logits_ac, emb_a, emb_b, emb_c
    
    def _compute_momentum_contrastive_losses(self, logits_ab_main, logits_ac_main, emb_b_main, emb_c_main,
                                           logits_ab_momentum, logits_ac_momentum, emb_b_momentum, emb_c_momentum,
                                           inputs, step_name):
        """Compute contrastive losses using momentum encoders for stable targets."""
        loss = 0
        n_valid_losses = 0
        logs = {}
                
        # 1. Morphological contrastive loss with momentum targets (reuse SoftHard)
        if logits_ab_main is not None and emb_b_momentum is not None:
            # Apply DINO-style centering to momentum embeddings before computing targets (if enabled)
            if self.use_centering:
                emb_b_momentum_centered = emb_b_momentum - self.center_morph
            else:
                emb_b_momentum_centered = emb_b_momentum
            

            b_valid = ~torch.all(inputs['x_b'] == -1, dim=1)
            # Reuse parent implementation with centered momentum embeddings
            morph_loss, morph_fwd, morph_bwd = super()._compute_morphological_contrastive_loss(
                logits_ab_main[b_valid],
                emb_b_momentum_centered,
                is_training=(step_name == "train")
            )
            loss += morph_loss
            n_valid_losses += 1
            # Use the same log keys as the parent class for consistency
            logs["morphological_loss"] = morph_loss.detach()
            logs["loss_ab"] = morph_fwd.detach()
            logs["loss_ba"] = morph_bwd.detach()
            
            # Note: Collapse detection for main encoders removed since momentum encoders
            # provide better collapse prevention and monitoring
        
        # 2. Genomic contrastive loss with momentum targets (reuse SoftHard)
        if logits_ac_main is not None and logits_ac_momentum is not None and logits_ac_main.size(0) > 0:
            # Apply DINO-style centering to momentum embeddings before computing targets (if enabled)
            if self.use_centering:
                emb_c_momentum_centered = emb_c_momentum - self.center_genomic
            else:
                emb_c_momentum_centered = emb_c_momentum
            
            # Reuse parent implementation with centered momentum embeddings
            batch_indices = inputs['batch_indices']
            genomic_loss, gen_fwd, gen_bwd = super()._compute_genomic_soft_contrastive_loss(
                logits_ac=logits_ac_main,
                emb_c=emb_c_momentum_centered,
                batch_indices=batch_indices,
                is_training=(step_name == "train")
            )
            loss += genomic_loss
            n_valid_losses += 1
            # Use the same log keys as the parent class for consistency
            logs["genomic_loss"] = genomic_loss.detach()
            logs["loss_ac"] = gen_fwd.detach()
            logs["loss_ca"] = gen_bwd.detach()
            
            # Note: Collapse detection for main encoders removed since momentum encoders
            # provide better collapse prevention and monitoring
        
        # Average loss over number of modalities
        loss = loss / max(n_valid_losses, 1)
        logs["n_valid_momentum_losses"] = n_valid_losses
        
        return loss, logs
    
    
    def _step(self, batch, batch_idx, step_name, dataloader_idx=None):
        """Override _step to implement momentum encoder training."""
        inputs = batch["inputs"]
        
        # Forward pass through main encoders
        logits_ab_main, logits_ac_main, emb_a_main, emb_b_main, emb_c_main = self.forward(**inputs, use_momentum=False)
        
        # Forward pass through momentum encoders (for targets)
        with torch.no_grad():
            logits_ab_momentum, logits_ac_momentum, _, emb_b_momentum, emb_c_momentum = self.forward(**inputs, use_momentum=True)
        
        # Compute momentum contrastive losses
        contrastive_loss, contrastive_logs = self._compute_momentum_contrastive_losses(
            logits_ab_main, logits_ac_main, emb_b_main, emb_c_main,
            logits_ab_momentum, logits_ac_momentum, emb_b_momentum, emb_c_momentum,
            inputs, step_name
        )
        
        # Update momentum encoders and centers after computing loss
        if step_name == "train":
            if self.update_teacher_encoders:   
                self._update_momentum_encoders()
            if self.use_centering:
                self._update_centers(emb_b_momentum, emb_c_momentum)
        
        # Compute accuracies using main encoder outputs
        accuracy_logs = self._compute_accuracies(logits_ab_main, logits_ac_main, inputs)
        
        # Combine all logs
        logs = {}
        logs.update(contrastive_logs)
        logs.update(accuracy_logs)
        logs["loss"] = contrastive_loss.detach()  # Keep as GPU tensor for sync_dist
        
        logs = {f"{step_name}/{k}": v for k, v in logs.items()}

        batch_dictionary = {
            "loss": contrastive_loss,
            "log": logs,
        }

        return batch_dictionary
