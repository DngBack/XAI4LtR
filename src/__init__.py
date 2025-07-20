"""
XAI4LtR: Explainable Learning to Reject Framework

A comprehensive framework for implementing selective classification systems
with explainable AI methods for medical image analysis and other domains.

This package provides:
- Base classifiers and ensemble models
- Multiple calibration methods (Temperature Scaling, Isotonic Regression, Beta Calibration)
- OOD detection methods (ODIN, Energy Score)
- Training dynamics analysis
- XAI visualization with Grad-CAM++
- Comprehensive evaluation metrics
- Modular architecture for different datasets
"""

__version__ = "1.0.0"
__author__ = "XAI4LtR Team"

from .config import Config
from .models import BaseClassifier, LabelSmoothingLoss
from .data import CTScanDataset, prepare_datasets
from .training import train_model, train_ensemble
from .calibration import TemperatureScaler, IsotonicCalibrator, BetaCalibrator
from .ood_detection import calculate_single_model_odin_score
from .rejection import get_rejection_scores_and_predictions, find_optimal_rejection_threshold
from .evaluation import calculate_metrics, calculate_ece
from .visualization import visualize_ensemble_grad_cam, visualize_xai_examples
from .utils import set_seed, extract_features, adjust_confidence_with_training_dynamics

__all__ = [
    'Config',
    'BaseClassifier', 
    'LabelSmoothingLoss',
    'CTScanDataset',
    'prepare_datasets',
    'train_model',
    'train_ensemble',
    'TemperatureScaler',
    'IsotonicCalibrator', 
    'BetaCalibrator',
    'calculate_single_model_odin_score',
    'get_rejection_scores_and_predictions',
    'find_optimal_rejection_threshold',
    'calculate_metrics',
    'calculate_ece',
    'visualize_ensemble_grad_cam',
    'visualize_xai_examples',
    'set_seed',
    'extract_features',
    'adjust_confidence_with_training_dynamics'
]
