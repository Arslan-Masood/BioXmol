from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score, f1_score,
                             precision_score, recall_score, balanced_accuracy_score,
                             confusion_matrix, matthews_corrcoef, roc_curve,
                             cohen_kappa_score)
from scipy.stats import spearmanr, kendalltau

            
def _validation_epoch_end(self, validation_step_outputs, is_regression=False):
    # Handle single dictionary case
    if isinstance(validation_step_outputs, dict):
        validation_step_outputs = [validation_step_outputs]
    
    # Handle empty case
    if not validation_step_outputs:
        return
        
    # Handle list of dictionaries case
    if not isinstance(validation_step_outputs[0], list):
        validation_step_outputs = [validation_step_outputs]

    for i, validation_step_output in enumerate(validation_step_outputs):
        # If single dictionary, wrap it in a list
        if isinstance(validation_step_output, dict):
            validation_step_output = [validation_step_output]
            
        all_logs = [o["log"] for o in validation_step_output]
        
        # Collect all unique keys across all batches
        all_keys = set()
        for log in all_logs:
            all_keys.update(log.keys())
        
        # For each key, collect values from batches that have that key
        aggregated_logs = {}
        for k in all_keys:
            values = [log[k].cpu().detach().item() if torch.is_tensor(log[k]) else log[k] for log in all_logs if k in log]
            if values:  # Only add if there are values
                mean_val = np.mean(values)
                aggregated_logs[k] = mean_val
        
        all_logs = aggregated_logs

        if (
            "outputs" in validation_step_output[0]
            and "labels" in validation_step_output[0]
        ):
            all_outputs = torch.cat([o["outputs"] for o in validation_step_output])
            all_labels = torch.cat([o["labels"] for o in validation_step_output])
            metric_func = _supervised_metric
            if is_regression:
                metric_func = _supervised_metric_regression
            metrics = metric_func(all_labels, all_outputs)
            metrics = {f"val/{k}": v for k, v in metrics.items()}
            all_logs.update(metrics)
            
            # Clean up large tensors immediately after computation
            del all_outputs, all_labels

        if i != 0:
            all_logs = {f"{k}_{i}": v for k, v in all_logs.items()}
        
        # Log metrics only if not in sanity check
        if not self.trainer.sanity_checking:
            # When using manual gathering, all ranks have identical gathered data
            # Use sync_dist=True with sync_dist_op="mean" to average (not sum) across ranks
            # Since all ranks have same values, mean of identical values = same value
            self.log_dict(all_logs, on_step=False, on_epoch=True)

def _validation_epoch_end_all_cell_lines(self, validation_step_outputs, is_regression=False):
    # Handle single dictionary case
    if isinstance(validation_step_outputs, dict):
        validation_step_outputs = [validation_step_outputs]
    
    # Handle empty case
    if not validation_step_outputs:
        return
        
    # Handle list of dictionaries case
    if not isinstance(validation_step_outputs[0], list):
        validation_step_outputs = [validation_step_outputs]

    for i, validation_step_output in enumerate(validation_step_outputs):
        # If single dictionary, wrap it in a list
        if isinstance(validation_step_output, dict):
            validation_step_output = [validation_step_output]
            
        # Extract logs and calculate means
        all_logs = [o["log"] for o in validation_step_output]
        
        # List of metrics we want to extract
        desired_metrics = [
            'val/loss_ab',
            'val/loss_ba', 
            'val/morphological_loss',
            'val/genomic_loss',
            'val/n_cell_lines',
            'val/loss',
            'val/n_valid_losses'
        ]
        
        # Filter and calculate means only for desired metrics
        filtered_logs = {}
        for metric in desired_metrics:
            # Check if metric exists in any of the logs, not just the first one
            values = [log[metric].cpu().detach().item() if torch.is_tensor(log[metric]) else log[metric] for log in all_logs if metric in log]
            if values:  # Only add if there are values
                filtered_logs[metric] = np.mean(values)

        if i != 0:
            filtered_logs = {f"{k}_{i}": v for k, v in filtered_logs.items()}
        
        # Log metrics only if not in sanity check
        if not self.trainer.sanity_checking:
            self.log_dict(filtered_logs, on_step=False, on_epoch=True, sync_dist=True)
    
def _supervised_metric(supervised_labels, supervised_outputs):
    metric = defaultdict(list)
    for labels, logits in zip(supervised_labels.T, supervised_outputs.T):
        mask = labels != -1
        masked_labels = torch.masked_select(labels, mask).cpu().detach().numpy()
        if len(masked_labels) == 0 or np.max(masked_labels) == np.min(masked_labels):
            # Handle edge cases where all labels are the same
            metric["auroc"].append(0.5)
            metric["auprc"].append(0.5)
            metric["f1"].append(0.0)
            metric["precision"].append(0.0)
            metric["recall"].append(0.0)
            metric["sensitivity"].append(0.0)
            metric["specificity"].append(0.0)
            metric["balanced_accuracy"].append(0.5)
            metric["average_precision"].append(0.5)
            metric["ECE"].append(0.0)
            metric["mcc"].append(0.0)
            metric["ppv"].append(0.0)
            metric["pos_lr"].append(0.0)
            metric["optimal_threshold"].append(0.5)
            continue

        masked_logits = torch.masked_select(logits, mask).cpu().detach().numpy()

        fpr, tpr, thresholds = roc_curve(masked_labels, masked_logits)
        J = tpr - fpr
        ix = np.argmax(J)
        try:
            optimal_thresh = thresholds[ix]
            # Handle infinity values
            if np.isinf(optimal_thresh) or np.isnan(optimal_thresh):
                optimal_thresh = 0.5
        except:
            optimal_thresh = 0.5

        # Convert probabilities to binary predictions using optimal threshold
        binary_preds = (masked_logits > optimal_thresh).astype(int)
        
        # Store optimal threshold for this task
        metric["optimal_threshold"].append(optimal_thresh)
        
        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(masked_labels, binary_preds).ravel()
        
        # Calculate all metrics
        metric["auroc"].append(roc_auc_score(masked_labels, masked_logits))
        metric["auprc"].append(average_precision_score(masked_labels, masked_logits))
        metric["f1"].append(f1_score(masked_labels, binary_preds, zero_division=0))
        metric["precision"].append(precision_score(masked_labels, binary_preds, zero_division=0))
        metric["recall"].append(recall_score(masked_labels, binary_preds, zero_division=0))
        
        # Sensitivity = Recall = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metric["sensitivity"].append(sensitivity)
        
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metric["specificity"].append(specificity)
        
        # Balanced Accuracy = (Sensitivity + Specificity) / 2
        metric["balanced_accuracy"].append(balanced_accuracy_score(masked_labels, binary_preds))
        
        # Average Precision Score (same as AUPRC)
        metric["average_precision"].append(average_precision_score(masked_labels, masked_logits))
        
        # Expected Calibration Error (ECE)
        ece = _compute_expected_calibration_error(masked_labels, masked_logits)
        metric["ECE"].append(ece)
        
        # Matthews Correlation Coefficient (MCC)
        mcc = matthews_corrcoef(masked_labels, binary_preds)
        metric["mcc"].append(mcc)
        
        # Cohen's Kappa (κ) - accounts for chance agreement
        cohen_kappa = cohen_kappa_score(masked_labels, binary_preds)
        metric["cohen_kappa"].append(cohen_kappa)
        
        # Positive Predictive Value (PPV) = Precision = TP / (TP + FP)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metric["ppv"].append(ppv)
        
        # Enrichment Factor = Precision / Prevalence
        # Shows how much better the model is than random selection
        prevalence = np.mean(masked_labels)  # Fraction of positive cases in dataset
        if prevalence > 0:
            enrichment_factor = ppv / prevalence
        else:
            enrichment_factor = 0.0  # Avoid division by zero
        metric["enrichment_factor"].append(enrichment_factor)
        
        # Ranking metrics for drug discovery prioritization
        # Spearman's ρ (rho) - rank correlation
        try:
            spearman_rho, _ = spearmanr(masked_labels, masked_logits)
            if np.isnan(spearman_rho):
                spearman_rho = 0.0
        except:
            spearman_rho = 0.0
        metric["spearman_rho"].append(spearman_rho)
        
        # Kendall's τ (tau) - rank correlation
        try:
            kendall_tau, _ = kendalltau(masked_labels, masked_logits)
            if np.isnan(kendall_tau):
                kendall_tau = 0.0
        except:
            kendall_tau = 0.0
        metric["kendall_tau"].append(kendall_tau)
        
        # Top-k ranking metrics (most relevant for drug discovery)
        # Spearman's ρ and Kendall's τ on top-k fractions
        
        # Top 10% ranking correlation
        n_top_10 = max(1, int(len(masked_labels) * 0.1))  # Top 10%
        top_10_indices = np.argsort(masked_logits)[-n_top_10:]  # Highest scores
        if len(top_10_indices) > 1:  # Need at least 2 points for correlation
            top_10_labels = masked_labels[top_10_indices]
            top_10_scores = masked_logits[top_10_indices]
            try:
                spearman_rho_top10, _ = spearmanr(top_10_labels, top_10_scores)
                if np.isnan(spearman_rho_top10):
                    spearman_rho_top10 = 0.0
            except:
                spearman_rho_top10 = 0.0
            try:
                kendall_tau_top10, _ = kendalltau(top_10_labels, top_10_scores)
                if np.isnan(kendall_tau_top10):
                    kendall_tau_top10 = 0.0
            except:
                kendall_tau_top10 = 0.0
        else:
            spearman_rho_top10 = 0.0
            kendall_tau_top10 = 0.0
        metric["spearman_rho_top10"].append(spearman_rho_top10)
        metric["kendall_tau_top10"].append(kendall_tau_top10)
        
        # Top 20% ranking correlation
        n_top_20 = max(1, int(len(masked_labels) * 0.2))  # Top 20%
        top_20_indices = np.argsort(masked_logits)[-n_top_20:]  # Highest scores
        if len(top_20_indices) > 1:  # Need at least 2 points for correlation
            top_20_labels = masked_labels[top_20_indices]
            top_20_scores = masked_logits[top_20_indices]
            try:
                spearman_rho_top20, _ = spearmanr(top_20_labels, top_20_scores)
                if np.isnan(spearman_rho_top20):
                    spearman_rho_top20 = 0.0
            except:
                spearman_rho_top20 = 0.0
            try:
                kendall_tau_top20, _ = kendalltau(top_20_labels, top_20_scores)
                if np.isnan(kendall_tau_top20):
                    kendall_tau_top20 = 0.0
            except:
                kendall_tau_top20 = 0.0
        else:
            spearman_rho_top20 = 0.0
            kendall_tau_top20 = 0.0
        metric["spearman_rho_top20"].append(spearman_rho_top20)
        metric["kendall_tau_top20"].append(kendall_tau_top20)
        
        # Top 50% ranking correlation
        n_top_50 = max(1, int(len(masked_labels) * 0.5))  # Top 50%
        top_50_indices = np.argsort(masked_logits)[-n_top_50:]  # Highest scores
        if len(top_50_indices) > 1:  # Need at least 2 points for correlation
            top_50_labels = masked_labels[top_50_indices]
            top_50_scores = masked_logits[top_50_indices]
            try:
                spearman_rho_top50, _ = spearmanr(top_50_labels, top_50_scores)
                if np.isnan(spearman_rho_top50):
                    spearman_rho_top50 = 0.0
            except:
                spearman_rho_top50 = 0.0
            try:
                kendall_tau_top50, _ = kendalltau(top_50_labels, top_50_scores)
                if np.isnan(kendall_tau_top50):
                    kendall_tau_top50 = 0.0
            except:
                kendall_tau_top50 = 0.0
        else:
            spearman_rho_top50 = 0.0
            kendall_tau_top50 = 0.0
        metric["spearman_rho_top50"].append(spearman_rho_top50)
        metric["kendall_tau_top50"].append(kendall_tau_top50)
        
        # Positive Likelihood Ratio = Sensitivity / (1 - Specificity)
        if specificity != 1:
            pos_lr = sensitivity / (1 - specificity)
            # Handle infinity values
            if np.isinf(pos_lr) or np.isnan(pos_lr):
                pos_lr = 1.0  # Default to 1.0 for infinite values
        else:
            pos_lr = 1.0  # Default to 1.0 instead of infinity
        metric["pos_lr"].append(pos_lr)
        
        # Decision-making metrics for drug discovery
        # Recall@Precision: For finding compounds we're confident are safe (non-toxic)
        recall_at_precision_75 = _calculate_recall_at_precision(masked_labels, masked_logits, 0.75)
        recall_at_precision_80 = _calculate_recall_at_precision(masked_labels, masked_logits, 0.80)
        recall_at_precision_85 = _calculate_recall_at_precision(masked_labels, masked_logits, 0.85)
        recall_at_precision_90 = _calculate_recall_at_precision(masked_labels, masked_logits, 0.90)
        recall_at_precision_95 = _calculate_recall_at_precision(masked_labels, masked_logits, 0.95)
        
        metric["recall_at_precision_75"].append(recall_at_precision_75)
        metric["recall_at_precision_80"].append(recall_at_precision_80)
        metric["recall_at_precision_85"].append(recall_at_precision_85)
        metric["recall_at_precision_90"].append(recall_at_precision_90)
        metric["recall_at_precision_95"].append(recall_at_precision_95)
        
        # TNR@Recall: For filtering out compounds we're confident are toxic
        tnr_at_recall_75 = _calculate_tnr_at_recall(masked_labels, masked_logits, 0.75)
        tnr_at_recall_80 = _calculate_tnr_at_recall(masked_labels, masked_logits, 0.80)
        tnr_at_recall_85 = _calculate_tnr_at_recall(masked_labels, masked_logits, 0.85)
        tnr_at_recall_90 = _calculate_tnr_at_recall(masked_labels, masked_logits, 0.90)
        tnr_at_recall_95 = _calculate_tnr_at_recall(masked_labels, masked_logits, 0.95)
        
        metric["tnr_at_recall_75"].append(tnr_at_recall_75)
        metric["tnr_at_recall_80"].append(tnr_at_recall_80)
        metric["tnr_at_recall_85"].append(tnr_at_recall_85)
        metric["tnr_at_recall_90"].append(tnr_at_recall_90)
        metric["tnr_at_recall_95"].append(tnr_at_recall_95)

    metric_mean = {f"{k}_mean": np.mean(v) for k, v in metric.items()}
    return metric_mean


def _compute_expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration error calculation
    
    Returns:
        ECE value
    """
    try:
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence in this bin
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    except:
        # Return 0 if calculation fails
        return 0.0


def _supervised_metric_regression(supervised_labels, supervised_outputs):
    metric = defaultdict(list)
    for labels, logits in zip(supervised_labels.T, supervised_outputs.T):
        labels = labels.cpu().detach().numpy()
        logits = logits.cpu().detach().numpy()
        metric["mae"].append(mean_absolute_error(labels, logits))
        metric["mse"].append(mean_squared_error(labels, logits))
    metric = {k: np.mean(v) for k, v in metric.items()}
    return metric


def _calculate_recall_at_precision(y_true, y_scores, min_precision):
    """
    Calculate recall at fixed precision threshold.
    
    Args:
        y_true: Binary labels (0=non-toxic, 1=toxic)
        y_scores: Prediction scores/probabilities
        min_precision: Minimum precision threshold
        
    Returns:
        Recall achieved at the given precision threshold
    """
    try:
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Find the threshold that achieves at least min_precision
        valid_indices = precision >= min_precision
        if valid_indices.any():
            # Take the threshold that gives maximum recall among valid precisions
            max_recall_idx = np.argmax(recall[valid_indices])
            actual_idx = np.where(valid_indices)[0][max_recall_idx]
            return recall[actual_idx]
        else:
            return 0.0
    except:
        return 0.0


def _calculate_tnr_at_recall(y_true, y_scores, min_recall):
    """
    Calculate True Negative Rate (TNR) at fixed recall threshold.
    
    Args:
        y_true: Binary labels (0=non-toxic, 1=toxic)
        y_scores: Prediction scores/probabilities
        min_recall: Minimum recall threshold
        
    Returns:
        TNR achieved at the given recall threshold
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        tnr = 1 - fpr  # True Negative Rate = 1 - False Positive Rate
        
        # Find the threshold that achieves at least min_recall
        valid_indices = tpr >= min_recall
        if valid_indices.any():
            # Take the threshold that gives maximum TNR among valid recalls
            max_tnr_idx = np.argmax(tnr[valid_indices])
            actual_idx = np.where(valid_indices)[0][max_tnr_idx]
            return tnr[actual_idx]
        else:
            return 0.0
    except:
        return 0.0
