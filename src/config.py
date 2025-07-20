"""
Configuration module for XAI4LtR framework.

This module contains the Config class that manages all hyperparameters,
paths, and settings for the selective classification system.
"""

import os
import torch


class Config:
    """
    Configuration and hyperparameters for the entire selective classification pipeline.
    These parameters can be adjusted to fit different datasets and requirements.
    """
    
    def __init__(self, data_dir=None, model_save_dir=None, xai_save_dir=None):
        # Data paths
        self.DATA_DIR = data_dir or '/kaggle/input/covidqu/Infection Segmentation Data/Infection Segmentation Data/Train'
        self.COVID_DIR = os.path.join(self.DATA_DIR, 'COVID-19/images')
        self.NON_COVID_DIR = os.path.join(self.DATA_DIR, 'Normal/images')

        # Model and Training
        self.IMAGE_SIZE = (224, 224)  # Standard size for many pre-trained CNNs
        self.BATCH_SIZE = 16  # Reduced batch size to save GPU memory
        self.NUM_EPOCHS_PER_MODEL = 4  # Increased epochs for better training
        self.LEARNING_RATE = 1e-4
        self.NUM_ENSEMBLE_MODELS = 5  # Number of models in ensemble

        # Monte Carlo Dropout (MCDO) Configuration
        self.MCDO_ENABLE = False  # Enable/disable Monte Carlo Dropout (Baseline A.1)
        self.MCDO_DROPOUT_RATE = 0.5  # Dropout rate for MCDO
        self.MCDO_NUM_RUNS = 10  # Number of forward passes for MCDO uncertainty estimation

        # Label Smoothing Configuration (Baseline A.2.3)
        self.LABEL_SMOOTHING_ENABLE = False  # Enable/disable Label Smoothing
        self.LABEL_SMOOTHING_EPSILON = 0.1  # Epsilon parameter for Label Smoothing

        # Training Dynamics Configuration (Baseline B.3)
        self.ENABLE_TRAINING_DYNAMICS = False  # Flag to enable/disable confidence adjustment by training dynamics
        self.TRAINING_DYNAMICS_CONF_PENALTY = 0.1  # Degree to which "hard-to-learn" examples during training reduce confidence of similar test examples
        
        # ODIN/Energy Score Parameters (Baseline B.1.x, B.2.x)
        self.ODIN_TEMP = 1000.0  # Temperature for ODIN. Higher temperature usually works better.
        self.ODIN_EPSILON = 0.001  # Magnitude of perturbation for ODIN (can be adjusted based on dataset)
        self.ENERGY_CLASSIFY_PERCENTILE_THRESHOLD = 20  # Samples with energy scores in bottom X% are considered potential OOD
        
        # Rejection targets
        self.TARGET_ACCEPTED_ACCURACY = 0.99  # 99.5% accuracy on accepted cases
        self.TARGET_REJECTION_RATE = 0.07  # Reject approximately 10% of cases

        # Adjustable parameters for selective classification (experiment with these!)
        self.DISAGREEMENT_PENALTY_FACTOR = 5.0  # Degree to which ensemble disagreement reduces confidence
        
        # Weights for rejection threshold optimization objective (importance of each factor)
        # These weights allow tuning the trade-off between accepted accuracy, rejection rate, and ECE
        self.ACCURACY_DEVIATION_WEIGHT = 2.5  # High weight to strongly enforce TARGET_ACCEPTED_ACCURACY
        self.REJECTION_RATE_DEVIATION_WEIGHT = 1.0  # Standard weight for rejection rate deviation
        self.ECE_DEVIATION_WEIGHT = 5.0  # Weight for ECE in rejection threshold optimization (adjust this, higher values mean better calibrated accepted set)

        # OOD detection thresholds (can be manually adjusted in categorize_rejected_cases)
        # These are thresholds used to CATEGORIZE rejected cases, NOT to make initial rejection decisions
        self.OOD_CONFIDENCE_THRESHOLD = 0.65  # Samples below this confidence
        self.OOD_VARIANCE_THRESHOLD = 0.08  # And above this variance are potential OOD (used for older baselines)
        self.ODIN_CLASSIFY_THRESHOLD = 0.8  # Samples with ODIN score < this threshold are considered potential OOD
        
        # Enable/disable Weighted Logits by Confidence (ECE)
        self.USE_WEIGHTED_ENSEMBLE = True 
        
        # Enable/disable Dynamic Ensemble Selection (select best model for each sample)
        # Note: Set NUM_ENSEMBLE_MODELS = 3 for 2/3 selection logic to work correctly
        self.USE_DYNAMIC_SELECTION = True
        self.DYNAMIC_SELECTION_COUNT = 2  # Select 2 best models

        # Small constant to avoid division by zero when computing weights from ECE
        self.EPSILON_ECE = 1e-8

        # Other settings
        self.RANDOM_SEED = 42
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Save directories
        self.MODEL_SAVE_DIR = model_save_dir or '/kaggle/working/models'
        self.XAI_SAVE_DIR = xai_save_dir or '/kaggle/working/xai_visualizations'
        
        # Create directories
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(self.XAI_SAVE_DIR, exist_ok=True)

    def update_data_paths(self, data_dir, covid_dir=None, non_covid_dir=None):
        """Update data paths for different datasets."""
        self.DATA_DIR = data_dir
        if covid_dir:
            self.COVID_DIR = covid_dir
        else:
            self.COVID_DIR = os.path.join(self.DATA_DIR, 'COVID-19/images')
        
        if non_covid_dir:
            self.NON_COVID_DIR = non_covid_dir
        else:
            self.NON_COVID_DIR = os.path.join(self.DATA_DIR, 'Normal/images')

    def update_save_paths(self, model_save_dir, xai_save_dir):
        """Update save directories."""
        self.MODEL_SAVE_DIR = model_save_dir
        self.XAI_SAVE_DIR = xai_save_dir
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(self.XAI_SAVE_DIR, exist_ok=True)

    def enable_mcdo(self, dropout_rate=0.5, num_runs=10):
        """Enable Monte Carlo Dropout with specified parameters."""
        self.MCDO_ENABLE = True
        self.MCDO_DROPOUT_RATE = dropout_rate
        self.MCDO_NUM_RUNS = num_runs

    def enable_label_smoothing(self, epsilon=0.1):
        """Enable Label Smoothing with specified epsilon."""
        self.LABEL_SMOOTHING_ENABLE = True
        self.LABEL_SMOOTHING_EPSILON = epsilon

    def enable_training_dynamics(self, penalty=0.1):
        """Enable Training Dynamics adjustment with specified penalty."""
        self.ENABLE_TRAINING_DYNAMICS = True
        self.TRAINING_DYNAMICS_CONF_PENALTY = penalty

    def set_rejection_targets(self, target_accuracy=0.99, target_rejection_rate=0.07):
        """Set rejection targets."""
        self.TARGET_ACCEPTED_ACCURACY = target_accuracy
        self.TARGET_REJECTION_RATE = target_rejection_rate

    def set_optimization_weights(self, accuracy_weight=2.5, rejection_weight=1.0, ece_weight=5.0):
        """Set optimization weights for threshold finding."""
        self.ACCURACY_DEVIATION_WEIGHT = accuracy_weight
        self.REJECTION_RATE_DEVIATION_WEIGHT = rejection_weight
        self.ECE_DEVIATION_WEIGHT = ece_weight 