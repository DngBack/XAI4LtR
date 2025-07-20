"""
Out-of-Distribution (OOD) Detection module for XAI4LtR framework.

This module contains methods for detecting out-of-distribution samples:
- ODIN (Out-of-Distribution Detector for Neural Networks)
- Energy Score
"""

import numpy as np
import torch
import torch.nn.functional as F


def calculate_single_model_odin_score(model, inputs, temp, epsilon, device):
    """
    Calculate ODIN score for input batch on a single model.
    Returns ODIN score (numpy array), higher score means more in-distribution.
    
    Args:
        model: Neural network model
        inputs: Input batch
        temp: Temperature parameter for ODIN
        epsilon: Perturbation magnitude for ODIN
        device: Device to run on
        
    Returns:
        odin_score: ODIN scores for the batch
    """
    # Ensure inputs can compute gradients
    inputs.requires_grad_(True)
    
    # Set model to evaluation mode
    model.eval() 
    
    # Forward pass to get logits
    outputs = model(inputs)
    
    # Apply temperature to logits
    temp_outputs = outputs / temp
    
    # Get predicted class for perturbation
    pred_class = temp_outputs.argmax(dim=1)
    
    # Calculate loss (negative log-likelihood) for predicted class.
    # Goal is to maximize probability of this predicted class by perturbing input.
    loss = F.cross_entropy(temp_outputs, pred_class)
    
    # Calculate gradient of loss with respect to input
    # create_graph=False to avoid building graph for subsequent backward passes
    grad = torch.autograd.grad(loss, inputs, create_graph=False)[0] 
    
    # Create perturbed input
    perturbed_inputs = inputs - epsilon * torch.sign(grad)
    
    # Pass perturbed input through model again
    with torch.no_grad():  # No need for gradients for this step
        perturbed_outputs = model(perturbed_inputs)
        
    # ODIN score is maximum probability of perturbed output after applying temperature
    odin_probs = F.softmax(perturbed_outputs / temp, dim=1)
    odin_score = torch.max(odin_probs, dim=1)[0]
    
    inputs.requires_grad_(False)  # Reset input gradient requirement
    
    return odin_score.cpu().numpy()


def calculate_energy_score(logits):
    """
    Calculate Energy Score from logits.
    Lower energy scores indicate potential OOD samples.
    
    Args:
        logits: Model logits
        
    Returns:
        energy_scores: Energy scores
    """
    # Energy score is negative logsumexp of logits
    energy_scores = -torch.logsumexp(logits, dim=1)
    return energy_scores.detach().cpu().numpy()


def calculate_odin_scores_ensemble(ensemble_models, inputs, temp, epsilon, device):
    """
    Calculate ODIN scores for ensemble of models.
    
    Args:
        ensemble_models: List of ensemble models
        inputs: Input batch
        temp: Temperature parameter
        epsilon: Perturbation magnitude
        device: Device to run on
        
    Returns:
        avg_odin_scores: Average ODIN scores across ensemble
    """
    odin_scores = []
    
    for model in ensemble_models:
        model.eval()
        odin_score = calculate_single_model_odin_score(model, inputs, temp, epsilon, device)
        odin_scores.append(odin_score)
    
    # Average ODIN scores across ensemble
    avg_odin_scores = np.mean(np.stack(odin_scores, axis=1), axis=1)
    return avg_odin_scores


def calculate_energy_scores_ensemble(ensemble_models, inputs, device):
    """
    Calculate Energy scores for ensemble of models.
    
    Args:
        ensemble_models: List of ensemble models
        inputs: Input batch
        device: Device to run on
        
    Returns:
        avg_energy_scores: Average Energy scores across ensemble
    """
    energy_scores = []
    
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            logits = model(inputs)
            energy_score = calculate_energy_score(logits)
            energy_scores.append(energy_score)
    
    # Average Energy scores across ensemble
    avg_energy_scores = np.mean(np.stack(energy_scores, axis=1), axis=1)
    return avg_energy_scores


def normalize_energy_scores(energy_scores):
    """
    Normalize energy scores to [0, 1] range.
    
    Args:
        energy_scores: Raw energy scores
        
    Returns:
        normalized_scores: Normalized energy scores
    """
    if len(energy_scores) == 0:
        return energy_scores
    
    min_e, max_e = np.min(energy_scores), np.max(energy_scores)
    if (max_e - min_e) > 0:
        normalized_scores = (energy_scores - min_e) / (max_e - min_e)
    else: 
        normalized_scores = np.full_like(energy_scores, 0.5) 
    
    return normalized_scores


def detect_ood_samples(confidence_scores, ensemble_variances, odin_scores=None, 
                      energy_scores=None, cfg=None):
    """
    Detect potential OOD samples using multiple criteria.
    
    Args:
        confidence_scores: Model confidence scores
        ensemble_variances: Ensemble prediction variances
        odin_scores: ODIN scores (optional)
        energy_scores: Energy scores (optional)
        cfg: Configuration object
        
    Returns:
        ood_mask: Boolean mask indicating potential OOD samples
    """
    if cfg is None:
        # Default thresholds
        confidence_threshold = 0.65
        variance_threshold = 0.08
        odin_threshold = 0.8
    else:
        confidence_threshold = cfg.OOD_CONFIDENCE_THRESHOLD
        variance_threshold = cfg.OOD_VARIANCE_THRESHOLD
        odin_threshold = cfg.ODIN_CLASSIFY_THRESHOLD
    
    # Initialize OOD mask
    ood_mask = np.zeros(len(confidence_scores), dtype=bool)
    
    # Method 1: Low confidence and high variance
    confidence_ood = confidence_scores < confidence_threshold
    variance_ood = ensemble_variances > variance_threshold
    ood_mask |= (confidence_ood & variance_ood)
    
    # Method 2: ODIN scores
    if odin_scores is not None:
        odin_ood = odin_scores < odin_threshold
        ood_mask |= odin_ood
    
    # Method 3: Energy scores
    if energy_scores is not None and cfg is not None:
        energy_threshold = np.percentile(energy_scores, cfg.ENERGY_CLASSIFY_PERCENTILE_THRESHOLD)
        energy_ood = energy_scores < energy_threshold
        ood_mask |= energy_ood
    
    return ood_mask 