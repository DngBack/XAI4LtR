"""
Utility functions for XAI4LtR framework.

This module contains utility functions for:
- Feature extraction
- Training dynamics adjustment
- Random seed setting
- Data loading helpers
"""

import os
import random
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader


def set_seed(seed):
    """
    Set random seed for reproducibility across different libraries.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_features(model, data_loader, device):
    """
    Extract features from model's feature_extractor for all samples in data_loader.
    Returns features as numpy array and corresponding original global indices.
    
    Args:
        model: Model to extract features from
        data_loader: Data loader containing samples
        device: Device to run on
        
    Returns:
        all_features: Extracted features
        all_indices: Corresponding global indices
    """
    model.eval()  # Ensure model is in eval mode for normal feature extraction
    all_features = []
    all_indices = []
    
    with torch.no_grad():
        for inputs, _, global_indices_batch in tqdm(data_loader, desc="Extracting Features"):
            inputs = inputs.to(device)
            features = model.get_features(inputs)  # Use the new get_features method
            all_features.append(features.cpu().numpy())
            all_indices.extend(global_indices_batch.cpu().numpy())
    
    return np.vstack(all_features), np.array(all_indices)


def adjust_confidence_with_training_dynamics(cfg, test_features, current_scores,
                                           train_features, train_global_indices, 
                                           final_overall_learning_metrics):
    """
    Adjust confidence/rejection scores of test set based on similarity with training samples 
    and their learning dynamics.
    Lower scores for test samples similar to 'hard' training samples (e.g., learned late, inconsistent).
    
    Args:
        cfg: Configuration object
        test_features: Features of test samples
        current_scores: Current confidence/rejection scores
        train_features: Features of training samples
        train_global_indices: Global indices of training samples
        final_overall_learning_metrics: Learning metrics for training samples
        
    Returns:
        adjusted_scores: Adjusted confidence/rejection scores
    """
    print("Adjusting test set scores with training dynamics...")
    adjusted_scores = np.copy(current_scores)

    # Create mapping from global_idx to learning metrics dictionary for efficient lookup
    train_global_idx_to_metrics = {idx: metrics for idx, metrics in final_overall_learning_metrics.items()}

    # Calculate cosine similarity between test and training features
    if len(test_features) == 0 or len(train_features) == 0:
        print("Skipping training dynamics adjustment: No test or training features.")
        return adjusted_scores

    similarities = cosine_similarity(test_features, train_features)
    
    for i in tqdm(range(len(test_features)), desc="Applying training dynamics adjustment"):
        # Find most similar training sample (by index in train_features array)
        most_similar_train_idx_in_features_array = np.argmax(similarities[i])
        # Get original global index of most similar training sample
        most_similar_train_global_idx = train_global_indices[most_similar_train_idx_in_features_array]
        
        # Check if learning metrics exist for this global index
        if most_similar_train_global_idx in train_global_idx_to_metrics:
            sample_metrics = train_global_idx_to_metrics[most_similar_train_global_idx]
            
            # Use 'mean_first_correct_epoch' as proxy for 'learning lateness' or 'difficulty'.
            # Higher 'mean_first_epoch' indicates a harder-to-learn sample.
            difficulty_value = sample_metrics.get('mean_first_correct_epoch', cfg.NUM_EPOCHS_PER_MODEL)
            
            # Normalize difficulty between 0 and 1 (0 = easy, 1 = hard).
            # If a sample was learned late (higher epoch), it's harder, so normalized difficulty closer to 1.
            if cfg.NUM_EPOCHS_PER_MODEL > 0:
                normalized_difficulty = difficulty_value / cfg.NUM_EPOCHS_PER_MODEL
            else:
                normalized_difficulty = 0.0  # Default if no epochs defined
        else:
            # If index not found, default to no penalty (neutral difficulty)
            normalized_difficulty = 0.0 

        # Reduce score based on difficulty and adjustable penalty factor
        # Higher difficulty leads to greater score reduction
        adjustment_factor = 1.0 - (cfg.TRAINING_DYNAMICS_CONF_PENALTY * normalized_difficulty)
        
        adjusted_scores[i] *= adjustment_factor
        adjusted_scores[i] = max(0.0, adjusted_scores[i])  # Ensure score is non-negative

    return adjusted_scores


def load_ensemble_models(cfg, model_dir=None):
    """
    Load all ensemble models from saved checkpoints.
    
    Args:
        cfg: Configuration object
        model_dir: Directory containing model checkpoints (optional)
        
    Returns:
        loaded_models: List of loaded ensemble models
    """
    from .models import BaseClassifier
    
    if model_dir is None:
        model_dir = cfg.MODEL_SAVE_DIR
    
    loaded_models = []
    for i in range(cfg.NUM_ENSEMBLE_MODELS):
        model = BaseClassifier(
            num_classes=2, 
            dropout_rate=(cfg.MCDO_DROPOUT_RATE if cfg.MCDO_ENABLE else 0.0)
        ).to(cfg.DEVICE)
        
        model_path = os.path.join(model_dir, f'best_model_ensemble_{i}.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
            model.eval()
            loaded_models.append(model)
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    return loaded_models


def compute_ensemble_disagreement(ensemble_probs):
    """
    Compute ensemble disagreement based on prediction variances.
    
    Args:
        ensemble_probs: Array of shape (num_samples, num_models, num_classes)
        
    Returns:
        disagreement_scores: Disagreement scores for each sample
    """
    if len(ensemble_probs.shape) != 3:
        raise ValueError("ensemble_probs should have shape (num_samples, num_models, num_classes)")
    
    # Compute variance across ensemble members for each sample and class
    variances = np.var(ensemble_probs, axis=1)  # Shape: (num_samples, num_classes)
    
    # Take maximum variance across classes as disagreement measure
    disagreement_scores = np.max(variances, axis=1)
    
    return disagreement_scores


def normalize_scores(scores, method='minmax'):
    """
    Normalize scores to [0, 1] range.
    
    Args:
        scores: Input scores
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        normalized_scores: Normalized scores
    """
    if method == 'minmax':
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.full_like(scores, 0.5)
    elif method == 'zscore':
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        if std_score > 0:
            normalized_scores = (scores - mean_score) / std_score
            # Clip to reasonable range and scale to [0, 1]
            normalized_scores = np.clip(normalized_scores, -3, 3)
            normalized_scores = (normalized_scores + 3) / 6
        else:
            normalized_scores = np.full_like(scores, 0.5)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized_scores


def create_save_directories(cfg):
    """
    Create necessary save directories.
    
    Args:
        cfg: Configuration object
    """
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(cfg.XAI_SAVE_DIR, exist_ok=True)
    print(f"Created save directories:")
    print(f"  Models: {cfg.MODEL_SAVE_DIR}")
    print(f"  XAI: {cfg.XAI_SAVE_DIR}")


def get_device_info():
    """
    Get information about available devices.
    
    Returns:
        device_info: Dictionary containing device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    
    if torch.cuda.is_available():
        device_info['cuda_device_name'] = torch.cuda.get_device_name(0)
        device_info['cuda_memory_total'] = torch.cuda.get_device_properties(0).total_memory
    
    return device_info


def print_system_info(cfg):
    """
    Print system information and configuration.
    
    Args:
        cfg: Configuration object
    """
    device_info = get_device_info()
    
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Device: {device_info['current_device']}")
    print(f"CUDA Available: {device_info['cuda_available']}")
    
    if device_info['cuda_available']:
        print(f"CUDA Device: {device_info['cuda_device_name']}")
        print(f"CUDA Memory: {device_info['cuda_memory_total'] / 1e9:.1f} GB")
    
    print(f"Random Seed: {cfg.RANDOM_SEED}")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print(f"Ensemble Size: {cfg.NUM_ENSEMBLE_MODELS}")
    print(f"Epochs per Model: {cfg.NUM_EPOCHS_PER_MODEL}")
    print("=" * 50) 