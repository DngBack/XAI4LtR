"""
Models module for XAI4LtR framework.

This module contains the neural network models and loss functions
used in the selective classification system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BaseClassifier(nn.Module):
    """
    A basic CNN classifier using pre-trained ResNet model.
    Includes a clear feature extractor for easy connection to Grad-CAM
    and for extracting features for similarity-based analysis.
    Integrates Dropout layers for Monte Carlo Dropout (MCDO).
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.0):
        super(BaseClassifier, self).__init__()
        self.dropout_rate = dropout_rate
        
        # Load pre-trained ResNet18 model
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Define target layer for Grad-CAM (last convolutional layer)
        self.target_layer = self.model.layer4[-1] 
        
        # Define feature extractor (everything before the final FC layer)
        feature_extractor_layers = list(self.model.children())[:-1]
        
        # Add dropout layer if dropout_rate is positive
        if self.dropout_rate > 0:
            # Find index of AdaptiveAvgPool2d layer to insert dropout after it
            try:
                avgpool_idx = [i for i, layer in enumerate(feature_extractor_layers) 
                              if isinstance(layer, nn.AdaptiveAvgPool2d)][0]
                feature_extractor_layers.insert(avgpool_idx + 1, nn.Dropout(p=self.dropout_rate))
                print(f"Added Dropout layer with rate {self.dropout_rate} to feature extractor.")
            except IndexError:
                print("Warning: AdaptiveAvgPool2d layer not found. Adding Dropout after all convolutional layers.")
                feature_extractor_layers.append(nn.Dropout(p=self.dropout_rate))

        self.feature_extractor = nn.Sequential(*feature_extractor_layers)
        
        # Replace the final fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Forward pass through feature extractor
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)  # Flatten features
        # Forward pass through final classification layer
        output = self.model.fc(features)
        return output

    def get_features(self, x):
        """
        Extract features from the layer before the final classification layer.
        """
        # Note: Dropout layers in feature_extractor will automatically turn off if model.eval()
        # or activate if model.train() and dropout_rate > 0
        with torch.no_grad(): 
            features = self.feature_extractor(x)
            features = torch.flatten(features, 1) 
        return features


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for Baseline A.2.3.
    
    This loss function helps improve model calibration by preventing
    the model from becoming overconfident in its predictions.
    """
    
    def __init__(self, classes, epsilon=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - epsilon
        self.epsilon = epsilon
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.epsilon / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class GatingNetwork(nn.Module):
    """
    Gating Network for dynamic ensemble selection.
    
    This network learns to select the best ensemble members for each input sample
    based on their individual predictions and uncertainty estimates.
    """
    
    def __init__(self, input_dim, num_models, hidden_dim=64):
        super(GatingNetwork, self).__init__()
        self.num_models = num_models
        
        self.gating_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_models)
        )
        
    def forward(self, x):
        return self.gating_net(x)


def compute_gating_features(ensemble_logits):
    """
    Compute features for the gating network from ensemble logits.
    
    Args:
        ensemble_logits: List of logits from ensemble models
        
    Returns:
        gating_features: Concatenated features for gating network
    """
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