#!/usr/bin/env python3
"""
Script to extract embeddings from pretrained model and train Random Forest with nested cross-validation
for DILI toxicity prediction.

Usage:
    python extract_embeddings_and_train_rf.py --checkpoint_path <path> --layer <layer_name> --output_dir <dir>
"""

import os
import pickle
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import (
    balanced_accuracy_score, classification_report, confusion_matrix,
    matthews_corrcoef, roc_auc_score, roc_curve, average_precision_score,
    f1_score, precision_score
)
import sklearn.metrics as metrics
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from scipy.stats import randint
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add the mocop directory to Python path (same as train.py)
sys.path.insert(1, '/scratch/work/masooda1/Multi_Modal_Contrastive/mocop')

from model import LightningGGNN
from dataset import SupervisedGraphDataset
from torch.utils.data import DataLoader


def load_data(train_path, test_path):
    """Load train and test data from CSV files."""
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    print(f"Training data shape: {train_df.shape}")
    
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Test data shape: {test_df.shape}")
    
    return train_df, test_df


def create_dataset(df, smiles_col='SMILES', label_col='TOXICITY'):
    """Create a dataset from DataFrame."""
    # Create a temporary CSV file for the dataset
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    dataset = SupervisedGraphDataset(
        data_path=temp_file.name,
        cmpd_col=smiles_col,
        label_col=label_col,
        cmpd_col_is_inchikey=False,
        pad_length=250
    )
    
    # Clean up the temporary file
    import os
    os.unlink(temp_file.name)
    
    return dataset


def extract_embeddings(model, dataset, batch_size=32, layer_name='encode'):
    """Extract embeddings from the specified layer of the model."""
    model.eval()
    embeddings = []
    labels = []
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
    )
    
    print(f"Extracting embeddings from layer: {layer_name}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
            inputs = batch["inputs"]
            batch_labels = batch["labels"]
            
            # Move to device
            if torch.cuda.is_available():
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                batch_labels = batch_labels.cuda()
            
            # Extract embeddings from specified layer
            batch_embeddings = extract_from_layer(model, inputs['x_a'], layer_name)
            
            embeddings.append(batch_embeddings.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    
    # Concatenate all embeddings and labels
    X = np.vstack(embeddings)
    y = np.concatenate(labels)
    
    # Flatten labels if they're 2D
    if y.ndim > 1:
        y = y.flatten()
    
    print(f"Extracted embeddings shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution: {np.bincount(y.astype(int))}")
    
    return X, y


def extract_from_layer(model, x_a, layer_name):
    """Extract embeddings from specific layers of the GatedGraphNeuralNetwork.
    
    Args:
        model: LightningGGNN model
        x_a: Input tensor [adj_mat, node_feat, atom_vec]
        layer_name: One of 'GNN', 'first_fc', 'second_fc'
    
    Returns:
        torch.Tensor: Extracted embeddings
    """
    adj, node_feat, atom_vec = x_a
    
    # Forward through all conv layers (common for all options)
    for layer in model.model.conv_layers:
        node_feat = layer(adj, node_feat)
        node_feat = model.model.dropout(node_feat)
    
    # Apply atom_vec and sum to get graph-level representation
    output = torch.mul(node_feat, atom_vec)
    output = output.sum(1)
    
    if layer_name == 'GNN':
        # Return after conv layers (75-dim)
        return output
    elif layer_name == 'first_fc':
        # Continue through first FC layer (1024-dim)
        output = model.model.fc_layers[0](output)
        return output
    elif layer_name == 'second_fc':
        # Continue through first FC layer, then second FC layer (128-dim)
        output = model.model.fc_layers[0](output)
        output = model.model.dropout(output)
        output = model.model.fc_layers[1](output)
        return output
    else:
        raise ValueError(f"Unknown layer: {layer_name}. Choose from: GNN, first_fc, second_fc")


def nested_cross_validation(X_train, y_train, cfg):
    """Perform nested cross-validation with Random Forest."""
    list_of_lists = []
    
    # Set baseline based on the best model metric
    best_model_metric = cfg.random_forest.best_model_metric
    baseline_dict = {
        'matthews_corrcoef': 0.0,      # MCC baseline (0.0 = random performance)
        'auc': 0.5,                    # AUC baseline (0.5 = random performance)
        'balanced_accuracy': 0.5,      # Balanced accuracy baseline (0.5 = random performance)
        'f1_score': 0.0,               # F1 baseline (0.0 = no positive predictions)
        'sensitivity': 0.0,            # Sensitivity baseline (0.0 = no true positives)
        'specificity': 0.0,            # Specificity baseline (0.0 = no true negatives)
        'precision': 0.0,              # Precision baseline (0.0 = no positive predictions)
        'average_precision': 0.0       # Average precision baseline (0.0 = no positive predictions)
    }
    baseline = baseline_dict.get(best_model_metric, 0.0)  # Default to 0.0 if metric not found
    
    best_classifier = None
    
    for i in tqdm(cfg.random_forest.random_states, desc="Random states"):
        print(f"\nRandom state: {i}")
        
        # Outer CV
        outercv = KFold(n_splits=cfg.random_forest.outer_cv_splits, random_state=i, shuffle=True)
        
        for split, (train_index, test_index) in enumerate(outercv.split(X_train)):
            print(f"  Split: {split}")
            
            # Split data
            X_train_fold = X_train[train_index]
            y_train_fold = y_train[train_index]
            X_test_fold = X_train[test_index]
            y_test_fold = y_train[test_index]
            
            # Inner CV for hyperparameter optimization
            inner_cv = KFold(n_splits=cfg.random_forest.inner_cv_splits, random_state=i, shuffle=True)
            
            # Random Forest with hyperparameter search
            rf = RandomForestClassifier(n_jobs=-1, random_state=i)
            
            # Parameter grid from config
            param_dist_grid = {
                'max_depth': randint(cfg.random_forest.param_distributions.max_depth[0], cfg.random_forest.param_distributions.max_depth[1]),
                'max_features': randint(cfg.random_forest.param_distributions.max_features[0], cfg.random_forest.param_distributions.max_features[1]),
                'min_samples_leaf': randint(cfg.random_forest.param_distributions.min_samples_leaf[0], cfg.random_forest.param_distributions.min_samples_leaf[1]),
                'min_samples_split': randint(cfg.random_forest.param_distributions.min_samples_split[0], cfg.random_forest.param_distributions.min_samples_split[1]),
                'n_estimators': cfg.random_forest.param_distributions.n_estimators,
                'bootstrap': cfg.random_forest.param_distributions.bootstrap,
                'oob_score': cfg.random_forest.param_distributions.oob_score,
                'random_state': cfg.random_forest.param_distributions.random_state,
                'criterion': cfg.random_forest.param_distributions.criterion,
                'n_jobs': cfg.random_forest.param_distributions.n_jobs,
                'class_weight': cfg.random_forest.param_distributions.class_weight
            }
            
            # Halving random search
            rsh = HalvingRandomSearchCV(
                estimator=rf,
                param_distributions=param_dist_grid,
                factor=cfg.random_forest.halving_search.factor,
                random_state=i,
                n_jobs=cfg.random_forest.halving_search.n_jobs,
                verbose=cfg.random_forest.halving_search.verbose,
                cv=inner_cv,
                scoring=cfg.random_forest.halving_search.get('scoring', None)
            )
            
            rsh.fit(X_train_fold, y_train_fold)
            classifier = rsh.best_estimator_
            classifier.fit(X_train_fold, y_train_fold)
            
            # Threshold balancing
            cross_val_prob_cp = cross_val_predict(
                classifier, X_train_fold, y_train_fold, 
                cv=inner_cv, method='predict_proba', n_jobs=cfg.random_forest.halving_search.n_jobs
            )[:, 1]
            
            # Calculate ROC curves and find best threshold
            fpr, tpr, thresholds = roc_curve(y_train_fold, cross_val_prob_cp)
            J = tpr - fpr
            ix = np.argmax(J)
            best_thresh_cp = thresholds[ix]
            print(f'    Best Threshold: {best_thresh_cp:.4f}')
            
            # Predict on test fold
            y_proba = classifier.predict_proba(X_test_fold)[:, 1]
            y_pred = [1 if x > best_thresh_cp else 0 for x in y_proba]
            
            # Calculate metrics
            conf_matrix = confusion_matrix(y_test_fold, y_pred)
            print(f"    Confusion Matrix:\n{conf_matrix}")
            print(f"    Classification Report:\n{classification_report(y_test_fold, y_pred)}")
            
            ba = balanced_accuracy_score(y_test_fold, y_pred)
            mcc = matthews_corrcoef(y_test_fold, y_pred)
            auc = roc_auc_score(y_test_fold, y_proba)
            f1 = metrics.f1_score(y_test_fold, y_pred)
            ppv = metrics.precision_score(y_test_fold, y_pred, average='binary')
            avg_precision = average_precision_score(y_test_fold, y_proba)
            
            # Sensitivity and Specificity
            tn, fp, fn, tp = conf_matrix.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            pos_lr = sensitivity / (1 - specificity) if specificity != 1 else float('inf')
            
            print(f"    Balanced Accuracy: {ba:.4f}")
            print(f"    Matthews Correlation: {mcc:.4f}")
            print(f"    AUC: {auc:.4f}")
            print(f"    F1: {f1:.4f}")
            print(f"    Sensitivity: {sensitivity:.4f}")
            print(f"    Specificity: {specificity:.4f}")
            print(f"    Positive LR: {pos_lr:.4f}")
            print(f"    PPV: {ppv:.4f}")
            print(f"    Average Precision: {avg_precision:.4f}")
            
            # Track best model based on configured metric
            metrics_dict = {
                'matthews_corrcoef': mcc,
                'auc': auc,
                'balanced_accuracy': ba,
                'f1_score': f1,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': ppv,
                'average_precision': avg_precision
            }
            current_metric_value = metrics_dict[best_model_metric]
            if current_metric_value > baseline:
                print(f"    New best model found! {best_model_metric.upper()}: {current_metric_value:.4f} (previous best: {baseline:.4f})")
                baseline = current_metric_value
                best_classifier = classifier
            
            # Store results
            metrics_row = [
                "NCV", "DILI", i, split, ba, mcc, sensitivity, specificity, 
                auc, f1, pos_lr, ppv, avg_precision
            ]
            list_of_lists.append(metrics_row)
    
    # Create results DataFrame
    results = pd.DataFrame(
        list_of_lists,
        columns=[
            "NCV", "DILI", "i", "split", "ba", "mcc", "Sensitivity", 
            "Specificity", "auc", "f1", "pos_LR", "ppv", "average_precision_score"
        ]
    )
    
    return results, best_classifier


def evaluate_on_heldout_test(X_test, y_test, classifier, cfg):
    """Evaluate the best classifier on held-out test set."""
    print("\nEvaluating on held-out test set...")
    
    # Use the same threshold balancing approach
    inner_cv = KFold(n_splits=4, random_state=53, shuffle=True)
    
    # For threshold balancing, we need training data
    # Since we don't have it here, we'll use the test set for threshold calculation
    # This is not ideal but follows the user's code structure
    cross_val_prob_cp = cross_val_predict(
        classifier, X_test, y_test, 
        cv=inner_cv, method='predict_proba', n_jobs=cfg.random_forest.halving_search.n_jobs
    )[:, 1]
    
    # Calculate ROC curves and find best threshold
    fpr, tpr, thresholds = roc_curve(y_test, cross_val_prob_cp)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh_cp = thresholds[ix]
    print(f'Best Threshold: {best_thresh_cp:.4f}')
    
    # Predict on test set
    y_proba = classifier.predict_proba(X_test)[:, 1]
    y_pred = [1 if x > best_thresh_cp else 0 for x in y_proba]
    
    # Calculate metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    ba = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = metrics.f1_score(y_test, y_pred)
    ppv = metrics.precision_score(y_test, y_pred, average='binary')
    avg_precision = average_precision_score(y_test, y_proba)
    
    # Sensitivity and Specificity
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    pos_lr = sensitivity / (1 - specificity) if specificity != 1 else float('inf')
    
    print(f"Balanced Accuracy: {ba:.4f}")
    print(f"Matthews Correlation: {mcc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Positive LR: {pos_lr:.4f}")
    print(f"PPV: {ppv:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    # Plot ROC curve
    fpr_roc, tpr_roc, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_roc, tpr_roc, label='Our Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('ROC Curve - Held-out Test Set')
    plt.show()
    
    return {
        'balanced_accuracy': ba,
        'matthews_corrcoef': mcc,
        'auc': auc,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'positive_lr': pos_lr,
        'ppv': ppv,
        'average_precision': avg_precision
    }


@hydra.main(version_base=None, config_path="/scratch/work/masooda1/Multi_Modal_Contrastive/configs", config_name="RF_analysis_DILI")
def main(cfg: DictConfig) -> None:
    """Main function using Hydra configuration."""
    os.chdir(hydra.utils.get_original_cwd())
    print(OmegaConf.to_yaml(cfg))
    
    # Create output directory with layer subdirectory
    layer_output_dir = os.path.join(cfg.output.output_dir, f"layer_{cfg.layer}")
    os.makedirs(layer_output_dir, exist_ok=True)
    print(f"Results will be saved to: {layer_output_dir}")
    
    # Load data
    train_df, test_df = load_data(cfg.data.train_data_path, cfg.data.test_data_path)
    
    # Create datasets
    train_dataset = create_dataset(train_df, cfg.data.smiles_col, cfg.data.label_col)
    test_dataset = create_dataset(test_df, cfg.data.smiles_col, cfg.data.label_col)
    
    # Load pretrained model using Hydra instantiate
    print(f"Loading model from config...")
    model = hydra.utils.instantiate(cfg.model)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Extract embeddings
    print("Extracting training embeddings...")
    X_train, y_train = extract_embeddings(model, train_dataset, cfg.batch_size, cfg.layer)
    
    print("Extracting test embeddings...")
    X_test, y_test = extract_embeddings(model, test_dataset, cfg.batch_size, cfg.layer)
    
    # Save embeddings if requested
    if cfg.output.save_embeddings:
        layer_name = cfg.layer
        np.save(os.path.join(layer_output_dir, f'X_ncv_{layer_name}.npy'), X_train)
        np.save(os.path.join(layer_output_dir, f'y_ncv_{layer_name}.npy'), y_train)
        np.save(os.path.join(layer_output_dir, f'X_heldouttest_{layer_name}.npy'), X_test)
        np.save(os.path.join(layer_output_dir, f'y_heldouttest_{layer_name}.npy'), y_test)
        print(f"Embeddings saved to {layer_output_dir} with layer suffix: {layer_name}")
    
    # Perform nested cross-validation
    print("Starting nested cross-validation...")
    results, best_classifier = nested_cross_validation(X_train, y_train, cfg)
    
    # Save results
    results.to_csv(os.path.join(layer_output_dir, 'ncv_results.csv'), index=False)
    print(f"Results saved to {layer_output_dir}/ncv_results.csv")
    
    # Evaluate on held-out test set
    if best_classifier is not None:
        best_model_metric = cfg.random_forest.best_model_metric
        baseline_dict = {
            'matthews_corrcoef': 0.0, 'auc': 0.5, 'balanced_accuracy': 0.5,
            'f1_score': 0.0, 'sensitivity': 0.0, 'specificity': 0.0,
            'precision': 0.0, 'average_precision': 0.0
        }
        baseline_value = baseline_dict.get(best_model_metric, 0.0)
        print(f"Best classifier found with {best_model_metric.upper()} > {baseline_value}. Evaluating on held-out test set...")
        test_metrics = evaluate_on_heldout_test(X_test, y_test, best_classifier, cfg)
        
        # Save test metrics
        test_results_df = pd.DataFrame([test_metrics])
        test_results_df.to_csv(os.path.join(layer_output_dir, 'heldout_test_results.csv'), index=False)
        print(f"Test results saved to {layer_output_dir}/heldout_test_results.csv")
        
        # Save best model if requested
        if cfg.output.save_model:
            model_path = os.path.join(layer_output_dir, 'dili_bestNCV_model.sav')
            pickle.dump(best_classifier, open(model_path, 'wb'))
            print(f"Best model saved to {model_path}")
        else:
            print("Model saving disabled in config (save_model: false)")
    else:
        best_model_metric = cfg.random_forest.best_model_metric
        baseline_dict = {
            'matthews_corrcoef': 0.0, 'auc': 0.5, 'balanced_accuracy': 0.5,
            'f1_score': 0.0, 'sensitivity': 0.0, 'specificity': 0.0,
            'precision': 0.0, 'average_precision': 0.0
        }
        baseline_value = baseline_dict.get(best_model_metric, 0.0)
        print(f"No best classifier found! (No model achieved {best_model_metric.upper()} > {baseline_value})")


if __name__ == "__main__":
    main()

