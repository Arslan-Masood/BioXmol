
# VAE Model Evaluation Report

## Model Architecture
- Input Dimension: 4709
- Latent Dimension: 128
- Encoder Hidden Dims: [512, 256]
- Decoder Hidden Dims: [256, 512]
- Dropout: 0.1
- Normalization: batchnorm
- Beta (KL weight): 1.0

## Reconstruction Metrics

### Normalized Space (Training Space)
- Mean MAE: 0.414468 ± 0.455824
- Mean MSE: 0.765085 ± 2.980052
- Number of samples: 100

### Original Space (Interpretable Units)
- Mean MAE: 41.126389 ± 80.055878
- Mean MSE: 953548.187500 ± 5271770.500000

### Interpretation Notes
- **Normalized Space**: Reflects what the model actually optimized for during training
- **Original Space**: Shows reconstruction quality in original measurement units
- **Ratio (Orig/Norm)**: MAE ratio = 99.23, MSE ratio = 1246329.50

## Latent Space Analysis
- Best number of clusters: 2
- Best silhouette score: 0.8891
- PCA explained variance (PC1, PC2): 0.8062, 0.1277

## Feature Reconstruction Quality
- Features with excellent reconstruction (correlation > 0.9): 911 / 4707
- Features with good reconstruction (correlation > 0.8): 2253 / 4707
- Features with poor reconstruction (correlation < 0.5): 1098 / 4707
- Mean correlation in Original space: 0.6438

## Model Parameters
- Total parameters: 5,159,141
- Trainable parameters: 5,159,141

## Recommendations
- High error ratio suggests normalization is crucial for this dataset
- Consider improving model architecture or increasing training
- Latent space shows good clustering structure
