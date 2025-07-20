"""
Evaluation module for XAI4LtR framework.

This module contains evaluation metrics for selective classification:
- Accuracy metrics (overall, accepted)
- Calibration metrics (ECE, NLL, Brier Score)
- Discrimination metrics (AUROC, AUPR)
- Risk-Coverage metrics (AURC)
- F1-Score for rejection task
"""

import numpy as np
from sklearn.metrics import (accuracy_score, roc_curve, auc, precision_recall_curve, 
                            confusion_matrix, brier_score_loss, log_loss, f1_score, roc_auc_score)


def calculate_ece(model_predictions, rejection_scores, true_labels, num_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    
    Args:
        model_predictions: Model predictions
        rejection_scores: Rejection/confidence scores
        true_labels: True labels
        num_bins: Number of bins for ECE calculation
        
    Returns:
        ece: Expected Calibration Error
    """
    if len(rejection_scores) == 0:
        return 0.0

    bins = np.linspace(0., 1., num_bins + 1)
    ece = 0.0
    total_samples = len(true_labels)

    for i in range(num_bins):
        lower_bound = bins[i]
        upper_bound = bins[i+1]
        # Handle potential NaNs from previous calculations before comparison
        mask = (np.nan_to_num(rejection_scores, nan=-np.inf) >= lower_bound) & (np.nan_to_num(rejection_scores, nan=-np.inf) < upper_bound)
        if i == num_bins - 1:  # Include 1.0 in last bin
            mask = (np.nan_to_num(rejection_scores, nan=-np.inf) >= lower_bound) & (np.nan_to_num(rejection_scores, nan=-np.inf) <= upper_bound)

        bin_samples_indices = np.where(mask)[0]
        bin_count = len(bin_samples_indices)

        if bin_count > 0:
            bin_accuracy = accuracy_score(true_labels[bin_samples_indices], model_predictions[bin_samples_indices])
            bin_rejection_score_mean = np.mean(rejection_scores[bin_samples_indices])
            ece += (bin_count / total_samples) * np.abs(bin_accuracy - bin_rejection_score_mean)
    
    return ece


def calculate_metrics(model_predictions, rejection_scores, true_labels, rejection_threshold, verbose=True):
    """
    Calculate various selective classification metrics.
    `rejection_scores` is unified score, higher score means accepted.
    
    Args:
        model_predictions: Model predictions
        rejection_scores: Rejection scores
        true_labels: True labels
        rejection_threshold: Rejection threshold
        verbose: Whether to print metrics
        
    Returns:
        Dictionary containing all calculated metrics
    """
    accepted_indices = rejection_scores >= rejection_threshold
    rejected_indices = rejection_scores < rejection_threshold

    num_total = len(true_labels)
    num_accepted = np.sum(accepted_indices)
    num_rejected = np.sum(rejected_indices)

    # Coverage
    coverage = num_accepted / num_total
    rejection_rate = num_rejected / num_total

    # Accuracy on Accepted Cases (Risk)
    if num_accepted > 0:
        accepted_predictions = model_predictions[accepted_indices]
        accepted_true_labels = true_labels[accepted_indices]
        accuracy_accepted = accuracy_score(accepted_true_labels, accepted_predictions)
        risk = 1.0 - accuracy_accepted
        # NLL and Brier Score on accepted samples
        all_possible_labels = np.unique(true_labels)  # Get all unique labels from original true_labels
        nll_accepted = log_loss(accepted_true_labels, rejection_scores[accepted_indices], labels=all_possible_labels)
        brier_accepted = brier_score_loss(accepted_true_labels, rejection_scores[accepted_indices])
    else:
        accuracy_accepted = 0.0 
        risk = 1.0
        nll_accepted = np.nan
        brier_accepted = np.nan

    # Overall accuracy (for comparison)
    overall_accuracy = accuracy_score(true_labels, model_predictions)

    if verbose:
        print(f"\n--- Evaluation Results (Threshold={rejection_threshold:.4f}) ---")
        print(f"Overall accuracy (All samples): {overall_accuracy:.4f}")
        print(f"Coverage: {coverage:.4f} ({num_accepted} samples accepted)")
        print(f"Rejection rate: {rejection_rate:.4f} ({num_rejected} samples rejected)")
        print(f"Accuracy on accepted cases: {accuracy_accepted:.4f}")
        print(f"Risk on accepted cases: {risk:.4f}")

    # Calibration metrics (ECE - Expected Calibration Error)
    ece = calculate_ece(model_predictions, rejection_scores, true_labels)
    if verbose:
        print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
        print(f"Negative Log-Likelihood (NLL) on accepted samples: {nll_accepted:.4f}")
        print(f"Brier Score on accepted samples: {brier_accepted:.4f}")

    # AUROC and AUPR calculations
    is_correct = (model_predictions == true_labels).astype(int)
    
    if len(np.unique(true_labels)) > 1: 
        fpr, tpr, roc_thresholds = roc_curve(is_correct, rejection_scores)
        auroc = auc(fpr, tpr)
        if verbose:
            print(f"AUROC (Rejection score as score for correctness): {auroc:.4f}")
    else:
        auroc = np.nan

    if len(np.unique(is_correct)) > 1:
        precision_correct, recall_correct, _ = precision_recall_curve(is_correct, rejection_scores)
        aupr_correct = auc(recall_correct, precision_correct)
        if verbose:
            print(f"AUPR (Rejection score as score for correctness): {aupr_correct:.4f}")
    else:
        aupr_correct = np.nan

    # AURC calculation
    sorted_indices = np.argsort(rejection_scores)
    sorted_predictions = model_predictions[sorted_indices]
    sorted_labels = true_labels[sorted_indices]

    risks = []
    coverages = []
    for i_idx in range(num_total):
        current_accepted_preds = sorted_predictions[i_idx:]
        current_accepted_labels = sorted_labels[i_idx:]
        current_coverage = (num_total - i_idx) / num_total

        if (num_total - i_idx) == 0:
            current_risk = 1.0 
        else:
            num_correct = np.sum(current_accepted_preds == current_accepted_labels)
            current_risk = 1.0 - (num_correct / (num_total - i_idx)) 

        coverages.append(current_coverage)
        risks.append(current_risk)

    coverages = coverages[::-1]
    risks = risks[::-1]
    aurc = np.trapz(risks, coverages)
    
    if verbose:
        print(f"Area under Risk-Coverage curve (AURC): {aurc:.4f}")

    # F1-Score for rejection task: ability to correctly identify rejected samples
    is_rejected = (rejection_scores < rejection_threshold).astype(int)
    is_incorrect = (model_predictions != true_labels).astype(int)
    
    if len(np.unique(is_incorrect)) > 1:
        f1_rejection = f1_score(is_incorrect, is_rejected)
        if verbose:
            print(f"F1-Score (Rejection Task - Identifying Incorrect Predictions): {f1_rejection:.4f}")
    else:
        f1_rejection = np.nan

    return {
        'overall_accuracy': overall_accuracy,
        'accuracy_on_accepted': accuracy_accepted,
        'coverage': coverage,
        'rejection_rate': rejection_rate,
        'risk': risk,
        'ece': ece,
        'nll_accepted': nll_accepted,
        'brier_accepted': brier_accepted,
        'auroc_correct_incorrect': auroc,
        'aupr_correct_incorrect': aupr_correct,
        'aurc': aurc,
        'f1_rejection': f1_rejection
    }


def calculate_calibration_metrics(model_predictions, rejection_scores, true_labels, num_bins=10):
    """
    Calculate detailed calibration metrics.
    
    Args:
        model_predictions: Model predictions
        rejection_scores: Rejection/confidence scores
        true_labels: True labels
        num_bins: Number of bins for calibration analysis
        
    Returns:
        Dictionary containing calibration metrics
    """
    if len(rejection_scores) == 0:
        return {
            'ece': 0.0,
            'mce': 0.0,
            'reliability_diagram': None
        }

    bins = np.linspace(0., 1., num_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    max_calibration_error = 0.0

    for i in range(num_bins):
        lower_bound = bins[i]
        upper_bound = bins[i+1]
        mask = (rejection_scores >= lower_bound) & (rejection_scores < upper_bound)
        if i == num_bins - 1:
            mask = (rejection_scores >= lower_bound) & (rejection_scores <= upper_bound)

        bin_samples_indices = np.where(mask)[0]
        bin_count = len(bin_samples_indices)

        if bin_count > 0:
            bin_accuracy = accuracy_score(true_labels[bin_samples_indices], model_predictions[bin_samples_indices])
            bin_confidence = np.mean(rejection_scores[bin_samples_indices])
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
            
            # Calculate Maximum Calibration Error (MCE)
            calibration_error = np.abs(bin_accuracy - bin_confidence)
            max_calibration_error = max(max_calibration_error, calibration_error)
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
            bin_counts.append(0)

    # Filter out empty bins
    valid_bins_mask = ~np.isnan(bin_accuracies)
    bin_accuracies = np.array(bin_accuracies)[valid_bins_mask]
    bin_confidences = np.array(bin_confidences)[valid_bins_mask]
    bin_counts = np.array(bin_counts)[valid_bins_mask]

    # Calculate ECE
    ece = calculate_ece(model_predictions, rejection_scores, true_labels, num_bins)

    reliability_diagram = {
        'accuracies': bin_accuracies,
        'confidences': bin_confidences,
        'counts': bin_counts
    }

    return {
        'ece': ece,
        'mce': max_calibration_error,
        'reliability_diagram': reliability_diagram
    }


def calculate_discrimination_metrics(model_predictions, rejection_scores, true_labels):
    """
    Calculate discrimination metrics (AUROC, AUPR).
    
    Args:
        model_predictions: Model predictions
        rejection_scores: Rejection/confidence scores
        true_labels: True labels
        
    Returns:
        Dictionary containing discrimination metrics
    """
    is_correct = (model_predictions == true_labels).astype(int)
    
    metrics = {}
    
    # AUROC
    if len(np.unique(true_labels)) > 1 and len(np.unique(is_correct)) > 1:
        fpr, tpr, _ = roc_curve(is_correct, rejection_scores)
        metrics['auroc'] = auc(fpr, tpr)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
    else:
        metrics['auroc'] = np.nan
        metrics['fpr'] = None
        metrics['tpr'] = None

    # AUPR
    if len(np.unique(is_correct)) > 1:
        precision, recall, _ = precision_recall_curve(is_correct, rejection_scores)
        metrics['aupr'] = auc(recall, precision)
        metrics['precision'] = precision
        metrics['recall'] = recall
    else:
        metrics['aupr'] = np.nan
        metrics['precision'] = None
        metrics['recall'] = None

    return metrics


def calculate_risk_coverage_metrics(model_predictions, rejection_scores, true_labels):
    """
    Calculate risk-coverage metrics (AURC).
    
    Args:
        model_predictions: Model predictions
        rejection_scores: Rejection/confidence scores
        true_labels: True labels
        
    Returns:
        Dictionary containing risk-coverage metrics
    """
    num_total = len(true_labels)
    
    # Sort samples by score in increasing order to simulate increasing rejection
    sorted_indices = np.argsort(rejection_scores)
    sorted_predictions = model_predictions[sorted_indices]
    sorted_labels = true_labels[sorted_indices]

    risks = []
    coverages = []
    
    for i_idx in range(num_total):
        current_accepted_preds = sorted_predictions[i_idx:]
        current_accepted_labels = sorted_labels[i_idx:]
        current_coverage = (num_total - i_idx) / num_total

        if (num_total - i_idx) == 0:
            current_risk = 1.0 
        else:
            num_correct = np.sum(current_accepted_preds == current_accepted_labels)
            current_risk = 1.0 - (num_correct / (num_total - i_idx)) 

        coverages.append(current_coverage)
        risks.append(current_risk)

    # Reverse lists so coverage goes from 0 to 1
    coverages = np.array(coverages[::-1])
    risks = np.array(risks[::-1])
    
    aurc = np.trapz(risks, coverages)

    return {
        'aurc': aurc,
        'coverages': coverages,
        'risks': risks
    } 