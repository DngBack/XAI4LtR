"""
Training module for XAI4LtR framework.

This module contains training functions for individual models and ensemble training
with training dynamics tracking for selective classification.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from scipy.special import softmax

from .models import BaseClassifier, LabelSmoothingLoss


def train_model(model, train_loader, val_loader, epochs, lr, device, model_idx, cfg):
    """
    Train a single instance of the base classifier and track detailed training dynamics
    (predicted labels and confidence for each sample at each epoch).
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        model_idx: Index of the model in ensemble
        cfg: Configuration object
        
    Returns:
        sample_learning_metrics: Dictionary containing learning metrics for each sample
    """
    
    if cfg.LABEL_SMOOTHING_ENABLE:
        criterion = LabelSmoothingLoss(classes=2, epsilon=cfg.LABEL_SMOOTHING_EPSILON).to(device)
        print(f"Enabled Label Smoothing with epsilon: {cfg.LABEL_SMOOTHING_EPSILON}")
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    model.to(device)
    best_val_accuracy = 0.0
    
    # Dictionary to store detailed prediction history for each training sample across epochs
    # Key: global_idx of sample, Value: List of dicts, each dict (epoch, is_correct, predicted_label, confidence)
    training_prediction_details = {global_idx: [] for global_idx in train_loader.dataset.global_indices}

    print(f"\n--- Training Ensemble Model {model_idx + 1} ---")
    
    for epoch in range(epochs):
        model.train()  # Ensure model is in train mode (dropout active if enabled)
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels, global_indices_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            # Extract outputs before converting to numpy
            probs = softmax(outputs.detach().cpu().numpy(), axis=1) 
            predicted_labels = np.argmax(probs, axis=1)
            confidences_batch = np.max(probs, axis=1)

            total_samples += labels.size(0)
            correct_in_batch = (predicted_labels == labels.cpu().numpy()).sum().item()
            correct_predictions += correct_in_batch

            # Update training dynamics: save predicted label and confidence for each sample at this epoch
            for i, global_idx_tensor in enumerate(global_indices_batch):
                global_idx = global_idx_tensor.item()
                is_correct_prediction = (predicted_labels[i] == labels[i].item())
                training_prediction_details[global_idx].append({
                    'epoch': epoch,
                    'is_correct': is_correct_prediction,
                    'predicted_label': predicted_labels[i],
                    'confidence': confidences_batch[i]
                })
        
        # Clear CUDA cache after each epoch to free up memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples

        # Validation phase
        model.eval()  # Set model to eval mode for validation
        val_correct_predictions = 0
        val_total_samples = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels, _ in val_loader:  # Don't need global_indices in val_loader for validation step
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()

        val_accuracy = val_correct_predictions / val_total_samples
        val_loss /= val_total_samples

        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        scheduler.step()

        # Save best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(cfg.MODEL_SAVE_DIR, f'best_model_ensemble_{model_idx}.pth'))
            print(f"Saved best model {model_idx + 1} with Val Acc: {best_val_accuracy:.4f}")

    print(f"Completed training Model {model_idx + 1}. Best Val Acc: {best_val_accuracy:.4f}")
    
    # After all epochs for this model, compute 'learning metrics' for each sample
    sample_learning_metrics = {}
    for global_idx, history in training_prediction_details.items():
        correct_epochs_history = [h for h in history if h['is_correct']]
        
        avg_correct_confidence = np.mean([h['confidence'] for h in correct_epochs_history]) if correct_epochs_history else 0.0
        
        # Learning lateness: first epoch where it was correct and remained correct for all subsequent epochs
        first_correct_epoch = epochs  # Default to 'never really learned' (max epochs)
        for k_idx in range(len(history)): 
            if history[k_idx]['is_correct']:
                # Check if it remained correct until the end
                if all(h_sub['is_correct'] for h_sub in history[k_idx:]):
                    first_correct_epoch = history[k_idx]['epoch']
                    break
        
        # Consistency: ratio of correct predictions across all epochs for this model
        consistency = len(correct_epochs_history) / epochs if epochs > 0 else 0.0

        sample_learning_metrics[global_idx] = {
            'avg_correct_confidence': avg_correct_confidence,
            'first_correct_epoch': first_correct_epoch,
            'consistency': consistency
        }
    
    return sample_learning_metrics


def train_ensemble(cfg, train_loader, val_loader):
    """
    Train ensemble of NUM_ENSEMBLE_MODELS models and return training dynamics.
    
    Args:
        cfg: Configuration object
        train_loader: Training data loader
        val_loader: Validation data loader
        
    Returns:
        final_overall_learning_metrics: Aggregated learning metrics across all ensemble members
    """
    
    print(f"🔥 Training ensemble of {cfg.NUM_ENSEMBLE_MODELS} models...")
    
    # Initialize dictionary to store learning metrics from all models
    overall_learning_metrics = {}
    
    for i in range(cfg.NUM_ENSEMBLE_MODELS):
        print(f"\n🤖 Training model {i+1}/{cfg.NUM_ENSEMBLE_MODELS}")
        
        # Set different seed for diversity
        from .utils import set_seed
        set_seed(cfg.RANDOM_SEED + i)
        
        # Create new model for each ensemble member
        model = BaseClassifier(
            num_classes=2, 
            dropout_rate=(cfg.MCDO_DROPOUT_RATE if cfg.MCDO_ENABLE else 0.0)
        )
        
        # Train model and collect learning metrics
        individual_learning_metrics = train_model(
            model, train_loader, val_loader, 
            cfg.NUM_EPOCHS_PER_MODEL, cfg.LEARNING_RATE, 
            cfg.DEVICE, i, cfg
        )
        
        # Merge learning metrics from this model into overall metrics  
        for global_idx, metrics in individual_learning_metrics.items():
            if global_idx not in overall_learning_metrics:
                overall_learning_metrics[global_idx] = {
                    'avg_correct_confidence_list': [],
                    'first_correct_epoch_list': [],
                    'consistency_list': []
                }
            
            overall_learning_metrics[global_idx]['avg_correct_confidence_list'].append(metrics['avg_correct_confidence'])
            overall_learning_metrics[global_idx]['first_correct_epoch_list'].append(metrics['first_correct_epoch'])
            overall_learning_metrics[global_idx]['consistency_list'].append(metrics['consistency'])
        
        # Clear CUDA cache after each model's training
        if cfg.DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Aggregate learning metrics across all ensemble members
    final_overall_learning_metrics = {}
    for global_idx, all_model_metrics in overall_learning_metrics.items():
        # Calculate average learning metrics across ensemble members with correct names
        avg_conf_list = all_model_metrics['avg_correct_confidence_list']
        epoch_list = all_model_metrics['first_correct_epoch_list']
        consistency_list = all_model_metrics['consistency_list']
        
        final_overall_learning_metrics[global_idx] = {
            'mean_avg_correct_confidence': np.mean(avg_conf_list) if avg_conf_list else 0.0,
            'mean_first_correct_epoch': np.mean(epoch_list) if epoch_list else cfg.NUM_EPOCHS_PER_MODEL,
            'mean_consistency': np.mean(consistency_list) if consistency_list else 0.0
        }
    
    print(f"✅ Ensemble training completed!")
    print(f"📊 Training dynamics tracked for {len(final_overall_learning_metrics)} samples")
    
    return final_overall_learning_metrics


def train_gating_network(cfg, frozen_ensemble_models, gating_net, train_loader, val_loader, epochs=50, lr=1e-3):
    """
    Train the gating network to select the best ensemble members for each input.
    
    Args:
        cfg: Configuration object
        frozen_ensemble_models: List of frozen ensemble models
        gating_net: Gating network to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        best_val_accuracy: Best validation accuracy achieved
    """
    
    print("Training Gating Network...")
    
    # Freeze ensemble models
    for model in frozen_ensemble_models:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gating_net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    gating_net.to(cfg.DEVICE)
    best_val_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training phase
        gating_net.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels, _ in tqdm(train_loader, desc=f"Gating Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            # Get ensemble predictions
            ensemble_logits = []
            with torch.no_grad():
                for model in frozen_ensemble_models:
                    if cfg.MCDO_ENABLE:
                        model.train()  # Enable dropout for MCDO
                        mcd_logits = [model(inputs) for _ in range(cfg.MCDO_NUM_RUNS)]
                        avg_logits = torch.mean(torch.stack(mcd_logits), dim=0)
                        ensemble_logits.append(avg_logits)
                        model.eval()
                    else:
                        ensemble_logits.append(model(inputs))
            
            # Compute gating features and weights
            gating_features = compute_gating_features(ensemble_logits)
            weights = torch.softmax(gating_net(gating_features), dim=1)
            
            # Weighted ensemble prediction
            stacked_logits = torch.stack(ensemble_logits, dim=1)
            weighted_logits = torch.bmm(weights.unsqueeze(1), stacked_logits).squeeze(1)
            
            # Compute loss
            loss = criterion(weighted_logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(weighted_logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation phase
        gating_net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                
                # Get ensemble predictions
                ensemble_logits = []
                for model in frozen_ensemble_models:
                    if cfg.MCDO_ENABLE:
                        model.train()
                        mcd_logits = [model(inputs) for _ in range(cfg.MCDO_NUM_RUNS)]
                        avg_logits = torch.mean(torch.stack(mcd_logits), dim=0)
                        ensemble_logits.append(avg_logits)
                        model.eval()
                    else:
                        ensemble_logits.append(model(inputs))
                
                # Compute gating features and weights
                gating_features = compute_gating_features(ensemble_logits)
                weights = torch.softmax(gating_net(gating_features), dim=1)
                
                # Weighted ensemble prediction
                stacked_logits = torch.stack(ensemble_logits, dim=1)
                weighted_logits = torch.bmm(weights.unsqueeze(1), stacked_logits).squeeze(1)
                
                # Compute loss
                loss = criterion(weighted_logits, labels)
                val_loss += loss.item()
                _, predicted = torch.max(weighted_logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        
        scheduler.step()
        
        # Save best gating network
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(gating_net.state_dict(), os.path.join(cfg.MODEL_SAVE_DIR, 'best_gating_network.pth'))
            print(f"Saved best gating network with Val Acc: {best_val_accuracy:.4f}")
    
    print(f"Gating network training completed. Best Val Acc: {best_val_accuracy:.4f}")
    return best_val_accuracy


def compute_gating_features(ensemble_logits):
    """
    Compute features for the gating network from ensemble logits.
    
    Args:
        ensemble_logits: List of logits from ensemble models
        
    Returns:
        gating_features: Concatenated features for gating network
    """
    import torch.nn.functional as F
    
    features = []
    
    for logits in ensemble_logits:
        # Extract features from logits
        probs = F.softmax(logits, dim=1)
        confidence = torch.max(probs, dim=1)[0]
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        
        # Compute margin (difference between top two probabilities)
        sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        
        # Concatenate features
        model_features = torch.stack([confidence, entropy, margin], dim=1)
        features.append(model_features)
    
    # Concatenate all model features
    gating_features = torch.cat(features, dim=1)
    return gating_features 