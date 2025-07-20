"""
Rejection module for XAI4LtR framework.

This module contains the main functions for:
- Computing rejection scores and predictions
- Finding optimal rejection thresholds
- Categorizing rejected cases
"""

import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.special import softmax

from .models import BaseClassifier
from .calibration import get_calibrator
from .ood_detection import calculate_single_model_odin_score, calculate_energy_score
from .utils import extract_features, adjust_confidence_with_training_dynamics


def get_rejection_scores_and_predictions(cfg, data_loader, ensemble_models_dir, 
                                        final_overall_learning_metrics=None, train_dataset=None,
                                        calibration_method='temperature_scaling',
                                        ood_detection_method='none',
                                        combine_ood_with_disagreement=False):
    """
    Compute rejection scores and ensemble predictions for samples.
    Optionally integrates Monte Carlo Dropout (MCDO), calibration methods,
    and OOD detection methods (ODIN, Energy Score).
    Applies training dynamics adjustment if cfg.ENABLE_TRAINING_DYNAMICS is True.

    Args:
        cfg: Configuration object
        data_loader: Data loader for inference
        ensemble_models_dir: Directory containing ensemble models
        final_overall_learning_metrics: Training dynamics metrics (optional)
        train_dataset: Training dataset for dynamics adjustment (optional)
        calibration_method: Calibration method to use
        ood_detection_method: OOD detection method to use
        combine_ood_with_disagreement: Whether to combine OOD with disagreement

    Returns:
        Tuple containing:
        - all_predictions: Final ensemble predictions
        - final_rejection_scores: Final rejection scores (higher = accepted)
        - all_labels: True labels
        - all_original_indices: Original global indices
        - all_ensemble_individual_probs_stacked: Individual model probabilities
        - all_odin_scores_raw: Raw ODIN scores (None if not applied)
        - all_energy_scores_raw: Raw Energy scores (None if not applied)
    """
    
    all_predictions = []
    all_labels = []
    all_original_indices = []
    
    all_calibrated_confidences = []  # Confidence scores after calibration, before disagreement/OOD
    
    all_individual_run_probs_across_batches_if_mcdo_enabled = []  # For disagreement calculation
    all_odin_scores_across_batches = []  # Average ODIN scores for each sample
    all_energy_scores_across_batches = []  # Energy scores for each sample

    # Load all ensemble models
    loaded_models = []
    for i in range(cfg.NUM_ENSEMBLE_MODELS):
        model = BaseClassifier(
            num_classes=2, 
            dropout_rate=(cfg.MCDO_DROPOUT_RATE if cfg.MCDO_ENABLE else 0.0)
        ).to(cfg.DEVICE)
        model.load_state_dict(torch.load(os.path.join(ensemble_models_dir, f'best_model_ensemble_{i}.pth')))
        model.eval()  # Always in eval mode for inference when not using MCDO, but will enable dropout if MCDO_ENABLE
        loaded_models.append(model)
    
    # Clear CUDA cache before starting inference/calibration
    if cfg.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Initialize calibrator
    calibrator = None
    if calibration_method == 'temperature_scaling':
        calibrator = get_calibrator('temperature_scaling')
        calibrator.to(cfg.DEVICE)
        print("Using Temperature Scaling for calibration.")
    elif calibration_method == 'isotonic_regression':
        calibrator = get_calibrator('isotonic_regression')
        print("Using Isotonic Regression for calibration.")
    elif calibration_method == 'beta_calibration':
        calibrator = get_calibrator('beta_calibration')
        print("Using Beta Calibration for calibration.")
    else:
        print("No post-hoc calibration used (or invalid method).")

    print(f"Calibrating calibrator ({calibration_method}) on ensemble average logits/probabilities from current data_loader...")
    
    # Collect all logits/confidences/labels from current data_loader for calibration
    data_loader_ensemble_raw_outputs = []  # Logits or average probabilities
    data_loader_labels_for_calibration = []

    with torch.no_grad():
        for inputs_batch_cal, labels_batch_cal, _ in tqdm(data_loader, desc="Collecting Data for Calibration"):
            inputs_batch_cal = inputs_batch_cal.to(cfg.DEVICE)
            
            ensemble_logits_batch_cal = []
            if cfg.MCDO_ENABLE:
                for model_idx, model in enumerate(loaded_models):
                    model.train()  # Enable dropout for MCDO during inference for avg_logits for calibration
                    model_logits_runs = []
                    for _ in range(cfg.MCDO_NUM_RUNS):
                        model_logits_runs.append(model(inputs_batch_cal))
                    ensemble_logits_batch_cal.append(torch.stack(model_logits_runs).mean(dim=0))
            else:
                for model_idx, model in enumerate(loaded_models):
                    model.eval()
                    ensemble_logits_batch_cal.append(model(inputs_batch_cal))
            
            avg_logits_batch_cal = torch.stack(ensemble_logits_batch_cal).mean(dim=0)
            
            data_loader_ensemble_raw_outputs.append(avg_logits_batch_cal.cpu())  # Move to CPU
            data_loader_labels_for_calibration.append(labels_batch_cal.cpu())

    # Clear CUDA cache after collecting data for calibration
    if cfg.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    if data_loader_ensemble_raw_outputs: 
        data_loader_ensemble_raw_outputs_all = torch.cat(data_loader_ensemble_raw_outputs)
        data_loader_labels_for_calibration_all = torch.cat(data_loader_labels_for_calibration)

        if calibration_method == 'temperature_scaling':
            calibrator.calibrate(data_loader_ensemble_raw_outputs_all.to(cfg.DEVICE),
                                 data_loader_labels_for_calibration_all.to(cfg.DEVICE), cfg.DEVICE)
        elif calibration_method in ['isotonic_regression', 'beta_calibration']:
            probs_for_calibration = softmax(data_loader_ensemble_raw_outputs_all.numpy(), axis=1)
            confidences_for_calibration = np.max(probs_for_calibration, axis=1)
            calibrator.calibrate(confidences_for_calibration, data_loader_labels_for_calibration_all.numpy())
    else:
        print("Warning: No data in data_loader for calibration. Skipping post-hoc calibration.")

    # Clear CUDA cache after calibration
    if cfg.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Extract features from current data_loader (val/test set) for training dynamics adjustment
    print("Extracting features from current data loader for confidence adjustment...")
    loaded_models[0].eval()  # Ensure model is in eval mode when extracting features
    all_extracted_features, extracted_original_indices = extract_features(
        loaded_models[0], data_loader, cfg.DEVICE 
    )

    # Clear CUDA cache after feature extraction
    if cfg.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    print("Creating predictions and rejection scores...")
    for inputs, labels, original_indices_batch in tqdm(data_loader, desc="Prediction and Score Calculation"):
        inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)

        ensemble_logits_per_model = [] 
        current_batch_ensemble_run_probs = []  # For all_ensemble_individual_probs_stacked

        # Calculate ODIN/Energy for each sample/model
        current_batch_odin_scores_individual_models = []
        current_batch_energy_scores_individual_models = []

        for model_idx, model in enumerate(loaded_models):
            # Ensure model is in correct mode (train for MCDO, eval otherwise)
            if cfg.MCDO_ENABLE:
                model.train()  # Enable dropout during inference for MCDO
            else:
                model.eval()

            with torch.no_grad():  # Use no_grad for main inference
                if cfg.MCDO_ENABLE:
                    model_logits_runs = []
                    for _ in range(cfg.MCDO_NUM_RUNS):
                        model_logits_runs.append(model(inputs).cpu())  # Move logits to CPU immediately
                    
                    avg_logits_for_model = torch.stack(model_logits_runs).mean(dim=0).to(cfg.DEVICE)
                    ensemble_logits_per_model.append(avg_logits_for_model)

                    # Store all MCDO runs' probabilities for disagreement calculation
                    probs_runs = softmax(torch.stack([l.to(cfg.DEVICE) for l in model_logits_runs]).detach().cpu().numpy(), axis=2)
                    current_batch_ensemble_run_probs.append(np.transpose(probs_runs, (1, 0, 2)))
                else:  # MCDO is disabled, just one forward pass per model
                    logits = model(inputs)
                    ensemble_logits_per_model.append(logits)
                    
                    # Store single run probabilities for disagreement calculation
                    probs_individual = softmax(logits.detach().cpu().numpy(), axis=1)
                    current_batch_ensemble_run_probs.append(probs_individual[:, np.newaxis, :])

            # Calculate ODIN Score for each model
            if ood_detection_method == 'odin':
                odin_score_batch = calculate_single_model_odin_score(
                    model, inputs.clone().detach(), cfg.ODIN_TEMP, cfg.ODIN_EPSILON, cfg.DEVICE
                )
                current_batch_odin_scores_individual_models.append(odin_score_batch)
            
            # Calculate Energy Score for each model's logits
            if ood_detection_method == 'energy':
                with torch.no_grad():
                    if cfg.MCDO_ENABLE:
                        energy_score_batch = -torch.logsumexp(avg_logits_for_model, dim=1).detach().cpu().numpy()
                    else:  # If MCDO is disabled, use the single pass logits
                        energy_score_batch = -torch.logsumexp(logits, dim=1).detach().cpu().numpy()
                current_batch_energy_scores_individual_models.append(energy_score_batch)

        # Clear CUDA cache after processing each model within the batch loop
        if cfg.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        # Average ODIN/Energy scores across ensemble models for the current batch
        if current_batch_odin_scores_individual_models:
            all_odin_scores_across_batches.append(np.mean(np.stack(current_batch_odin_scores_individual_models, axis=1), axis=1))
        if current_batch_energy_scores_individual_models:
            all_energy_scores_across_batches.append(np.mean(np.stack(current_batch_energy_scores_individual_models, axis=1), axis=1))

        # Concatenate individual run probs for current batch across models
        if current_batch_ensemble_run_probs:
            all_individual_run_probs_across_batches_if_mcdo_enabled.append(np.concatenate(current_batch_ensemble_run_probs, axis=1))
        
        # Average logits from ensemble for this batch
        avg_ensemble_logits = torch.stack(ensemble_logits_per_model).mean(dim=0)

        # Apply Calibrator
        if calibrator:
            if calibration_method == 'temperature_scaling':
                calibrated_logits = calibrator.forward(avg_ensemble_logits)
                calibrated_probs = softmax(calibrated_logits.detach().cpu().numpy(), axis=1)
            elif calibration_method in ['isotonic_regression', 'beta_calibration']:
                initial_probs = softmax(avg_ensemble_logits.detach().cpu().numpy(), axis=1)
                initial_confidences = np.max(initial_probs, axis=1)
                
                calibrated_confidences_vals = calibrator.predict_proba(initial_confidences)
                calibrated_probs = initial_probs.copy()
                for k_idx in range(len(calibrated_probs)):
                    predicted_class = np.argmax(calibrated_probs[k_idx])
                    if calibrated_probs[k_idx][predicted_class] > 0:
                        scaling_factor = calibrated_confidences_vals[k_idx] / calibrated_probs[k_idx][predicted_class]
                        calibrated_probs[k_idx, :] *= scaling_factor
                        calibrated_probs[k_idx, :] = np.maximum(0, calibrated_probs[k_idx, :])
                        calibrated_probs[k_idx, :] /= (np.sum(calibrated_probs[k_idx, :]) + 1e-9)
                    else:  # Fallback for edge case
                        calibrated_probs[k_idx, predicted_class] = calibrated_confidences_vals[k_idx]
                        other_class_indices = [j for j in range(calibrated_probs.shape[1]) if j != predicted_class]
                        if len(other_class_indices) > 0:
                            total_other_prob = 1.0 - calibrated_confidences_vals[k_idx]
                            if np.sum(calibrated_probs[k_idx, other_class_indices]) > 0:
                                calibrated_probs[k_idx, other_class_indices] *= (total_other_prob / np.sum(calibrated_probs[k_idx, other_class_indices]))
                            else:
                                calibrated_probs[k_idx, other_class_indices] = total_other_prob / len(other_class_indices)
        else:  # No post-hoc calibration
            calibrated_probs = softmax(avg_ensemble_logits.detach().cpu().numpy(), axis=1)

        # This is the confidence after calibration, BEFORE disagreement/OOD penalty
        current_calibrated_confidences = np.max(calibrated_probs, axis=1)
        all_calibrated_confidences.extend(current_calibrated_confidences)

        # Store predictions and true labels
        predictions = np.argmax(calibrated_probs, axis=1)
        all_predictions.extend(predictions)
        all_labels.extend(labels.cpu().numpy())
        all_original_indices.extend(original_indices_batch.cpu().numpy())
    
    # After iterating through all batches, stack probabilities for disagreement
    if all_individual_run_probs_across_batches_if_mcdo_enabled:
        all_ensemble_individual_probs_stacked = np.concatenate(all_individual_run_probs_across_batches_if_mcdo_enabled, axis=0)
    else:
        all_ensemble_individual_probs_stacked = np.array([])

    # Consolidate ODIN and Energy scores
    all_odin_scores_raw = np.concatenate(all_odin_scores_across_batches, axis=0) if all_odin_scores_across_batches else None
    all_energy_scores_raw = np.concatenate(all_energy_scores_across_batches, axis=0) if all_energy_scores_across_batches else None

    # Convert lists to numpy arrays
    all_calibrated_confidences = np.array(all_calibrated_confidences)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_original_indices = np.array(all_original_indices)

    # Calculate Disagreement Penalty
    disagreement_penalties = np.zeros_like(all_calibrated_confidences)
    if all_ensemble_individual_probs_stacked.size > 0:
        num_samples = all_ensemble_individual_probs_stacked.shape[0]
        for i_idx in range(num_samples):
            current_sample_individual_probs = all_ensemble_individual_probs_stacked[i_idx, :, :] 
            ensemble_predicted_class = all_predictions[i_idx] 
            
            if (current_sample_individual_probs.size > 0 and 
                ensemble_predicted_class < current_sample_individual_probs.shape[1] and 
                ensemble_predicted_class >= 0):
                variance_disagreement = np.var(current_sample_individual_probs[:, ensemble_predicted_class])
            else:
                variance_disagreement = 0.0 
            
            disagreement_penalties[i_idx] = variance_disagreement * cfg.DISAGREEMENT_PENALTY_FACTOR

    # Calculate Final Rejection Score based on Method
    final_rejection_scores = np.copy(all_calibrated_confidences)

    if ood_detection_method == 'none':
        final_rejection_scores = all_calibrated_confidences * (1.0 - disagreement_penalties)
        final_rejection_scores = np.clip(final_rejection_scores, 0.0, 1.0)
    elif ood_detection_method == 'odin':
        if all_odin_scores_raw is None or all_odin_scores_raw.size == 0:
            print("Warning: ODIN requested but no ODIN scores. Using regular confidence scores.")
            final_rejection_scores = all_calibrated_confidences * (1.0 - disagreement_penalties)
        else:
            final_rejection_scores = all_odin_scores_raw
            if combine_ood_with_disagreement:
                final_rejection_scores = final_rejection_scores * (1.0 - disagreement_penalties)
            final_rejection_scores = np.clip(final_rejection_scores, 0.0, 1.0)
    elif ood_detection_method == 'energy':
        if all_energy_scores_raw is None or all_energy_scores_raw.size == 0:
            print("Warning: Energy Score requested but no Energy scores. Using regular confidence scores.")
            final_rejection_scores = all_calibrated_confidences * (1.0 - disagreement_penalties)
        else:
            if all_energy_scores_raw.size > 0:
                min_e, max_e = np.min(all_energy_scores_raw), np.max(all_energy_scores_raw)
                if (max_e - min_e) > 0:
                    normalized_energy_scores = (all_energy_scores_raw - min_e) / (max_e - min_e)
                else: 
                    normalized_energy_scores = np.full_like(all_energy_scores_raw, 0.5) 
            else:
                normalized_energy_scores = np.array([])

            final_rejection_scores = normalized_energy_scores
            if combine_ood_with_disagreement:
                final_rejection_scores = final_rejection_scores * (1.0 - disagreement_penalties)
            final_rejection_scores = np.clip(final_rejection_scores, 0.0, 1.0)

    # Apply Training Dynamics Adjustment if flag ENABLE_TRAINING_DYNAMICS is True
    if cfg.ENABLE_TRAINING_DYNAMICS and final_overall_learning_metrics is not None and train_dataset is not None:
        # 'all_extracted_features' already computed at the beginning of function for current data_loader
        loaded_models[0].eval()  # Ensure eval mode for feature extraction
        from torch.utils.data import DataLoader
        train_features_for_adjustment, train_global_indices_for_adjustment = extract_features(
            loaded_models[0], DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2), cfg.DEVICE
        )

        final_rejection_scores = adjust_confidence_with_training_dynamics(
            cfg, all_extracted_features, final_rejection_scores, 
            train_features_for_adjustment, train_global_indices_for_adjustment, final_overall_learning_metrics
        )
    elif cfg.ENABLE_TRAINING_DYNAMICS:
        print("Warning: ENABLE_TRAINING_DYNAMICS enabled but final_overall_learning_metrics or train_dataset not provided.")

    # Clear CUDA cache at the very end of the function
    if cfg.DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    return (all_predictions, final_rejection_scores, all_labels,
            all_original_indices, all_ensemble_individual_probs_stacked,
            all_odin_scores_raw, all_energy_scores_raw)


def find_optimal_rejection_threshold(rejection_scores, original_model_predictions, true_labels, cfg):
    """
    Find optimal confidence threshold on validation set
    to meet target accuracy on accepted cases, target rejection rate, and optimize ECE.
    
    Args:
        rejection_scores: Rejection scores (higher = accepted)
        original_model_predictions: Model predictions
        true_labels: True labels
        cfg: Configuration object
        
    Returns:
        best_threshold: Optimal rejection threshold
        results_df: DataFrame with threshold evaluation results
    """
    from .evaluation import calculate_ece
    
    thresholds = np.linspace(0.0, 1.0, 1000) 
    best_threshold = 0.0
    min_deviation = float('inf')

    print("\n--- Finding Optimal Rejection Threshold on Validation Set ---")
    results = []
    # Handle potential NaNs in rejection_scores before thresholding
    rejection_scores_clean = np.nan_to_num(rejection_scores, nan=-np.inf)  # Treat NaN as low score (rejected)

    for threshold in tqdm(thresholds, desc="Evaluating thresholds"):
        # `rejection_scores` is unified score, higher score means accepted
        accepted_indices = rejection_scores_clean >= threshold 
        
        num_total = len(true_labels)
        num_accepted = np.sum(accepted_indices)
        
        current_coverage = num_accepted / num_total
        current_rejection_rate = 1.0 - current_coverage

        current_accuracy_on_accepted = 0.0
        current_ece_on_accepted = 0.0

        if num_accepted > 0:
            accepted_predictions = original_model_predictions[accepted_indices]
            accepted_true_labels = true_labels[accepted_indices]
            accepted_rejection_scores = rejection_scores[accepted_indices]  # Use original scores for ECE calc

            from sklearn.metrics import accuracy_score
            current_accuracy_on_accepted = accuracy_score(accepted_true_labels, accepted_predictions)
            current_ece_on_accepted = calculate_ece(accepted_predictions, accepted_rejection_scores, accepted_true_labels)

        # Calculate deviation from targets using configurable weights
        accuracy_deviation = max(0, cfg.TARGET_ACCEPTED_ACCURACY - current_accuracy_on_accepted) * cfg.ACCURACY_DEVIATION_WEIGHT
        rejection_deviation = abs(current_rejection_rate - cfg.TARGET_REJECTION_RATE) * cfg.REJECTION_RATE_DEVIATION_WEIGHT
        # Minimize ECE on accepted set (lower ECE is better)
        ece_deviation = current_ece_on_accepted * cfg.ECE_DEVIATION_WEIGHT

        deviation = accuracy_deviation + rejection_deviation + ece_deviation

        results.append({
            'threshold': threshold,
            'accuracy_on_accepted': current_accuracy_on_accepted,
            'rejection_rate': current_rejection_rate,
            'ece_on_accepted': current_ece_on_accepted,
            'deviation': deviation
        })

    results_df = pd.DataFrame(results)
    # Filter thresholds that actually provide some coverage
    results_df = results_df[results_df['rejection_rate'] < 1.0]

    if not results_df.empty:
        # Find row with minimum total deviation
        best_row_idx = results_df['deviation'].idxmin()
        best_threshold_info = results_df.loc[best_row_idx]
        best_threshold = best_threshold_info['threshold']
        min_deviation = best_threshold_info['deviation']

        print(f"Found optimal threshold: {best_threshold:.4f}")
        print(f"  Accuracy on accepted cases: {best_threshold_info['accuracy_on_accepted']:.4f}")
        print(f"  Rejection rate: {best_threshold_info['rejection_rate']:.4f}")
        print(f"  ECE on accepted cases: {best_threshold_info['ece_on_accepted']:.4f}")
        print(f"  Total deviation: {best_threshold_info['deviation']:.4f}")
    else:
        print("Could not find suitable threshold, defaulting to 0.5. Please check your data and configuration targets.")
        best_threshold = 0.5

    return best_threshold, results_df


def categorize_rejected_cases(rejection_scores, model_predictions, true_labels, original_indices, 
                             rejection_threshold, all_ensemble_individual_probs_stacked, 
                             all_odin_scores=None, all_energy_scores=None, ood_detection_method='none'):
    """
    Categorize rejected cases.
    - 'Failure Rejection': Cases rejected due to low rejection score AND model prediction is incorrect.
    - 'Potential OOD Rejected Cases': Cases rejected due to low rejection score AND identified as potential OOD.
    - 'Unknown/Ambiguous': Cases rejected due to low rejection score, correct prediction, and not identified as potential OOD.
    
    Args:
        rejection_scores: Rejection scores
        model_predictions: Model predictions
        true_labels: True labels
        original_indices: Original global indices
        rejection_threshold: Rejection threshold
        all_ensemble_individual_probs_stacked: Individual ensemble probabilities
        all_odin_scores: ODIN scores (optional)
        all_energy_scores: Energy scores (optional)
        ood_detection_method: OOD detection method used
        
    Returns:
        Dictionary containing categorized rejected cases
    """
    from .config import Config
    
    # Create default config if not provided
    cfg = Config()
    
    rejected_mask = rejection_scores < rejection_threshold
    
    rejected_indices = original_indices[rejected_mask]
    rejected_model_preds = model_predictions[rejected_mask]
    rejected_true_labels = true_labels[rejected_mask]
    rejected_rejection_scores = rejection_scores[rejected_mask]

    failure_rejection_indices = []
    unknown_ambiguous_indices = [] 
    potential_ood_indices = [] 
    
    # Create mapping from original index (global) to flat position in original_indices array
    original_to_flat_pos_map = {original_idx: flat_pos for flat_pos, original_idx in enumerate(original_indices)}

    print(f"\n--- Categorizing Rejected Cases ({len(rejected_indices)} rejected) ---")

    # If energy scores are used for OOD classification, calculate the threshold now
    energy_ood_threshold = None
    if ood_detection_method == 'energy' and all_energy_scores is not None and all_energy_scores.size > 0:
        energy_ood_threshold = np.percentile(all_energy_scores, cfg.ENERGY_CLASSIFY_PERCENTILE_THRESHOLD)

    for i_idx, rejected_original_idx in enumerate(rejected_indices):
        current_pred = rejected_model_preds[i_idx]
        current_true = rejected_true_labels[i_idx]
        current_rejection_score = rejected_rejection_scores[i_idx]

        if rejected_original_idx not in original_to_flat_pos_map:
            print(f"Warning: Original index {rejected_original_idx} not found. Skipping categorization for this sample.")
            continue 

        flat_pos_in_full_data = original_to_flat_pos_map[rejected_original_idx]
        
        is_potential_ood = False

        if ood_detection_method == 'odin' and all_odin_scores is not None:
            if all_odin_scores[flat_pos_in_full_data] < cfg.ODIN_CLASSIFY_THRESHOLD:
                is_potential_ood = True
        elif ood_detection_method == 'energy' and all_energy_scores is not None and energy_ood_threshold is not None:
            if all_energy_scores[flat_pos_in_full_data] < energy_ood_threshold:
                is_potential_ood = True
        else:  # Default OOD classification using confidence and ensemble variance
            if all_ensemble_individual_probs_stacked.size > 0:
                individual_probs_for_this_sample = all_ensemble_individual_probs_stacked[flat_pos_in_full_data, :, :] 
                ensemble_predicted_class = current_pred
                variance_disagreement = 0.0
                if (individual_probs_for_this_sample.size > 0 and 
                    ensemble_predicted_class < individual_probs_for_this_sample.shape[1] and 
                    ensemble_predicted_class >= 0):
                    variance_disagreement = np.var(individual_probs_for_this_sample[:, ensemble_predicted_class])
                else:
                    if individual_probs_for_this_sample.size > 0:
                        variance_disagreement = np.max(np.var(individual_probs_for_this_sample, axis=0))
                    else:
                        variance_disagreement = 0.0 

                if current_rejection_score < cfg.OOD_CONFIDENCE_THRESHOLD and variance_disagreement > cfg.OOD_VARIANCE_THRESHOLD:
                    is_potential_ood = True

        # Categorize
        if current_pred != current_true:
            failure_rejection_indices.append(rejected_original_idx)
        elif is_potential_ood:
            potential_ood_indices.append(rejected_original_idx)
        else:  # Correct prediction but rejected due to general uncertainty/disagreement in distribution
            unknown_ambiguous_indices.append(rejected_original_idx)

    return {
        'rejected_data_df': pd.DataFrame({
            'original_idx': rejected_indices,
            'model_pred': rejected_model_preds,
            'true_label': rejected_true_labels,
            'rejection_score': rejected_rejection_scores
        }),
        'failure_rejection_indices': failure_rejection_indices,
        'unknown_ambiguous_indices': unknown_ambiguous_indices, 
        'potential_ood_indices': potential_ood_indices 
    } 