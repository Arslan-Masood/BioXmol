# Using Pretrained Encoders in Multimodal Contrastive Learning

This guide explains how to integrate pretrained Variational Autoencoders (VAEs) or Autoencoders (AEs) as encoders in the `CellLineTripleInputEncoder` framework for multimodal contrastive learning.

## Overview

Instead of randomly initializing the morphological (`encoder_b`) and genomic (`encoder_c`) encoders, you can now use pretrained encoders that have been trained on the respective modalities using autoencoding objectives. This approach provides several benefits:

1. **Better initialization**: Start with encoders that already understand the data modalities
2. **Faster convergence**: Reduced training time due to good initialization
3. **Better representations**: Leverage learned representations from autoencoding tasks
4. **Flexible fine-tuning**: Choose to freeze or fine-tune the pretrained encoders

## Quick Start

### 1. Train Pretrained Encoders

First, train separate VAE/AE models for your modalities:

```bash
# Train morphological (JUMP Cell Painting) autoencoder
python train_vae.py --config configs/train_ae_jump_cp_config.yaml

# Train genomic (LINCS L1000) autoencoder  
python train_vae.py --config configs/train_ae_genomic_config.yaml
```

### 2. Update Configuration

Update your multimodal training configuration to use the pretrained encoders:

```yaml
model:
  _target_: model.CellLineTripleInputEncoder
  
  # Original encoder configurations (used if no pretrained path provided)
  encoder_a:
    _target_: model.GatedGraphNeuralNetwork
    # ... molecular encoder config
  encoder_b:
    _target_: model.MultiLayerPerceptron
    # ... morphological encoder config (ignored if pretrained path provided)
  encoder_c:
    _target_: model.MultiLayerPerceptron
    # ... genomic encoder config (ignored if pretrained path provided)
  
  # Pretrained encoder paths
  pretrained_encoder_b_path: "/path/to/morphological_ae.ckpt"
  pretrained_encoder_c_path: "/path/to/genomic_ae.ckpt"
  
  # Freezing options
  freeze_encoder_b: false  # Allow fine-tuning
  freeze_encoder_c: false  # Allow fine-tuning
  
  # Note: latent layer usage is now always enabled by default (best practice)
```

### 3. Train Multimodal Model

```bash
python bin/train.py --config configs/jump_mocop_LINCS_with_pretrained.yml
```

## Configuration Options

### Pretrained Encoder Paths

```yaml
pretrained_encoder_b_path: "/path/to/checkpoint.ckpt"  # Path to morphological encoder
pretrained_encoder_c_path: "/path/to/checkpoint.ckpt"  # Path to genomic encoder
```

Set to `null` to use randomly initialized encoders instead.

### Freezing Options

```yaml
freeze_encoder_b: false  # Whether to freeze morphological encoder weights
freeze_encoder_c: false  # Whether to freeze genomic encoder weights
```

- `false`: Allow fine-tuning of pretrained weights (recommended for most cases)
- `true`: Freeze weights and use as fixed feature extractor

### Layer Selection

The system now automatically uses the latent projection layer (128-dimensional) from pretrained encoders, as this is the best practice for both VAE and AE models. This provides the most meaningful representations learned during pretraining.

## Training Strategies

### 1. Full Fine-tuning (Recommended)

Allow both encoders to be fine-tuned for the contrastive task:

```yaml
freeze_encoder_b: false
freeze_encoder_c: false
```

**Benefits**: Best performance, encoders adapt to contrastive objective
**Drawbacks**: Longer training time, requires more GPU memory

### 2. Feature Extraction

Freeze both encoders and only train the projection heads:

```yaml
freeze_encoder_b: true
freeze_encoder_c: true
```

**Benefits**: Faster training, lower memory usage, preserves pretrained features
**Drawbacks**: May not achieve optimal performance for contrastive task

### 3. Mixed Strategy

Freeze one encoder while fine-tuning the other:

```yaml
freeze_encoder_b: false  # Fine-tune morphological
freeze_encoder_c: true   # Freeze genomic
```

**Use cases**: When one modality has better pretrained representations than the other

## Architecture Compatibility

### Pretrained Model Requirements

The pretrained checkpoint must contain:
- A `VAEEncoder` or compatible encoder module
- Hyperparameters in the checkpoint
- Compatible input dimensions with your data

### Automatic Architecture Detection

The system automatically detects:
- Model type (VAE vs AE)
- Input/output dimensions
- Hidden layer configurations
- Latent dimension

### Dimension Handling

For genomic encoder (`encoder_c`):
- **Pretrained**: Uses genomic features directly (trained on them)
- **Random init**: Concatenates genomic features with cell line embeddings

## Example Configurations

### Configuration 1: Both Pretrained, Fine-tune Both

```yaml
pretrained_encoder_b_path: "/path/to/morphological_ae.ckpt"
pretrained_encoder_c_path: "/path/to/genomic_ae.ckpt"
freeze_encoder_b: false
freeze_encoder_c: false
use_latent_layer_b: true
use_latent_layer_c: true
```

### Configuration 2: Only Morphological Pretrained

```yaml
pretrained_encoder_b_path: "/path/to/morphological_ae.ckpt"
pretrained_encoder_c_path: null
freeze_encoder_b: false
freeze_encoder_c: false
use_latent_layer_b: true
use_latent_layer_c: true
```

### Configuration 3: Feature Extraction Mode

```yaml
pretrained_encoder_b_path: "/path/to/morphological_ae.ckpt"
pretrained_encoder_c_path: "/path/to/genomic_ae.ckpt"
freeze_encoder_b: true
freeze_encoder_c: true
use_latent_layer_b: true
use_latent_layer_c: true
```

## Troubleshooting

### Common Issues

1. **Checkpoint not found**
   ```
   FileNotFoundError: Checkpoint file not found: /path/to/checkpoint.ckpt
   ```
   **Solution**: Verify the checkpoint path exists and is accessible

2. **Dimension mismatch**
   ```
   RuntimeError: size mismatch, got input[X], expected input[Y]
   ```
   **Solution**: Ensure pretrained model was trained on compatible data dimensions

3. **Import errors**
   ```
   ImportError: Could not import VAE models
   ```
   **Solution**: Ensure `models/vae.py` is available and dependencies are installed

### Validation

Use the test script to validate your setup:

```bash
python test_pretrained_integration.py
```

This script will:
- Test encoder loading functionality
- Validate model configuration
- Show configuration examples
- Provide troubleshooting guidance

## Performance Considerations

### Memory Usage

- **Pretrained encoders**: Use existing memory footprint of the pretrained model
- **Frozen encoders**: Slightly lower memory usage (no gradient computation)
- **Fine-tuning**: Higher memory usage due to gradient computation

### Training Speed

- **Frozen encoders**: Faster forward pass, no backward pass through encoder
- **Fine-tuning**: Slower but typically converges faster due to good initialization

### Disk Space

- **Checkpoints**: Store full model state including encoder weights
- **Final models**: Include both pretrained and newly trained components

## Best Practices

1. **Start with fine-tuning**: Usually gives best performance
2. **Use latent layers**: Generally provides best representations
3. **Validate dimensions**: Ensure pretrained models match your data
4. **Monitor training**: Compare with random initialization baseline
5. **Save configurations**: Keep track of which pretrained models you used

## API Reference

### `load_pretrained_encoder()`

```python
def load_pretrained_encoder(
    checkpoint_path: str,
    freeze: bool = False,
    use_latent_layer: bool = True,
    device: Optional[torch.device] = None
) -> PretrainedEncoder
```

**Parameters:**
- `checkpoint_path`: Path to the `.ckpt` file
- `freeze`: Whether to freeze encoder weights
- `use_latent_layer`: Whether to use final latent projection
- `device`: Device to load model on

**Returns:** `PretrainedEncoder` wrapper object

### `CellLineTripleInputEncoder` New Parameters

```python
def __init__(
    self,
    # ... existing parameters ...
    pretrained_encoder_b_path: Optional[str] = None,
    pretrained_encoder_c_path: Optional[str] = None,
    freeze_encoder_b: bool = False,
    freeze_encoder_c: bool = False,
    use_latent_layer_b: bool = True,
    use_latent_layer_c: bool = True,
    # ... existing parameters ...
)
```

## Examples Repository

See the following files for complete examples:
- `configs/jump_mocop_LINCS_with_pretrained.yml`: Complete configuration example
- `test_pretrained_integration.py`: Test and demonstration script
- `mocop/pretrained_utils.py`: Implementation details
