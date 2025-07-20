"""
Calibration module for XAI4LtR framework.

This module contains calibration methods to improve confidence estimation:
- Temperature Scaling
- Isotonic Regression
- Beta Calibration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize


class TemperatureScaler(nn.Module):
    """
    Learn a single temperature scalar parameter to calibrate probabilities.
    Based on Guo et al. "On Calibration of Modern Neural Networks" (ICML 2017).
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, logits_to_calibrate, labels_for_calibration, device):
        """
        Adjust temperature parameter using pre-computed logits and labels.
        """
        # Ensure logits and labels are on the correct device
        logits_all = logits_to_calibrate.to(device)
        labels_all = labels_for_calibration.to(device)

        nll_criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50, line_search_fn='strong_wolfe')

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits_all), labels_all)
            loss.backward()
            return loss

        optimizer.step(eval)
        print(f"Temperature calibrator calibrated. Optimal T: {self.temperature.item():.4f}")


class IsotonicCalibrator:
    """
    Calibrate using Isotonic Regression.
    """
    
    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds="clip")

    def calibrate(self, confidences_to_calibrate, labels_for_calibration):
        """
        Calibrate the isotonic regression model.
        
        Args:
            confidences_to_calibrate: 1D array of confidence scores
            labels_for_calibration: 1D array of binary labels (0 or 1)
        """
        self.ir.fit(confidences_to_calibrate, labels_for_calibration)
        print("Isotonic Regression calibrated.")

    def predict_proba(self, confidences_to_transform):
        """
        Transform confidence scores using calibrated isotonic regression.
        
        Args:
            confidences_to_transform: Confidence scores to transform
            
        Returns:
            Calibrated confidence scores
        """
        return self.ir.transform(confidences_to_transform)


class BetaCalibrator:
    """
    Calibrate using Beta Calibration.
    Alpha and beta parameters of Beta distribution are optimized.
    """
    
    def __init__(self):
        self.alpha = None
        self.beta = None
    
    def _objective_function(self, params, confidences, labels):
        """
        Objective function for beta calibration optimization.
        
        Args:
            params: [alpha, beta] parameters
            confidences: Confidence scores
            labels: Binary labels
            
        Returns:
            Negative log-likelihood
        """
        alpha, beta = params
        # Clip confidences to avoid log(0) or log(1) issues
        conf_clamped = np.clip(confidences, 1e-10, 1 - 1e-10)
        
        # Apply transformation: logit(p_calibrated) = sigmoid(alpha * logit(p_original) + beta)
        logit_original = np.log(conf_clamped / (1 - conf_clamped))
        calibrated_confidences = 1.0 / (1.0 + np.exp(- (alpha * logit_original + beta)))
        
        # Clamp again to avoid log(0) or log(1)
        calibrated_confidences = np.clip(calibrated_confidences, 1e-10, 1 - 1e-10)
        
        # Negative Log-Likelihood as objective
        nll = -np.mean(labels * np.log(calibrated_confidences) + (1 - labels) * np.log(1 - calibrated_confidences))
        return nll

    def calibrate(self, confidences_to_calibrate, labels_for_calibration):
        """
        Calibrate the beta calibration model.
        
        Args:
            confidences_to_calibrate: Confidence scores for calibration
            labels_for_calibration: Binary labels for calibration
        """
        # Initial guess for alpha and beta
        initial_params = [1.0, 0.0]  # alpha=1.0, beta=0.0 means no change (identity)
        
        # Perform optimization using L-BFGS-B (bounded to avoid extreme values)
        result = minimize(self._objective_function, initial_params, 
                          args=(confidences_to_calibrate, labels_for_calibration), 
                          method='L-BFGS-B', 
                          bounds=[(0.01, None), (None, None)])  # alpha must be positive
        
        self.alpha, self.beta = result.x
        print(f"Beta Calibration calibrated. Alpha: {self.alpha:.4f}, Beta: {self.beta:.4f}")

    def predict_proba(self, confidences_to_transform):
        """
        Transform confidence scores using calibrated beta calibration.
        
        Args:
            confidences_to_transform: Confidence scores to transform
            
        Returns:
            Calibrated confidence scores
        """
        if self.alpha is None or self.beta is None:
            raise ValueError("BetaCalibrator not calibrated. Please call .calibrate() first.")
        
        conf_clamped = np.clip(confidences_to_transform, 1e-10, 1 - 1e-10)
        logit_original = np.log(conf_clamped / (1 - conf_clamped))
        calibrated_conf = 1.0 / (1.0 + np.exp(- (self.alpha * logit_original + self.beta)))
        return np.clip(calibrated_conf, 0.0, 1.0)


def get_calibrator(calibration_method):
    """
    Factory function to get the appropriate calibrator.
    
    Args:
        calibration_method: String specifying calibration method
        
    Returns:
        Calibrator instance
    """
    if calibration_method == 'temperature_scaling':
        return TemperatureScaler()
    elif calibration_method == 'isotonic_regression':
        return IsotonicCalibrator()
    elif calibration_method == 'beta_calibration':
        return BetaCalibrator()
    else:
        raise ValueError(f"Unknown calibration method: {calibration_method}") 