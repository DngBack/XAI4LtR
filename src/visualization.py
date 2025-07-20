"""
Visualization module for XAI4LtR framework.

This module contains visualization functions for:
- XAI visualizations with Grad-CAM++
- Performance curves (calibration, ROC, PR, risk-coverage)
- Comparative plots across baselines
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    print("Warning: pytorch-grad-cam not available. XAI visualizations will be disabled.")
    GRADCAM_AVAILABLE = False


def preprocess_for_xai(image_path, cfg):
    """
    Load image and apply necessary transformations for XAI methods.
    
    Args:
        image_path: Path to image file
        cfg: Configuration object
        
    Returns:
        original_img_for_display: Original image for overlay (numpy array float 0-1)
        input_tensor: Transformed image for model input (tensor, normalized)
    """
    from torchvision import transforms
    
    img = Image.open(image_path).convert('RGB')
    
    # Original image for heatmap overlay (numpy array float 0-1)
    original_img_for_display = np.array(img.resize(cfg.IMAGE_SIZE)) / 255.0

    # Transformed image for model input (tensor, normalized)
    transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    input_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    return original_img_for_display, input_tensor


def visualize_ensemble_grad_cam(ensemble_models, image_path, original_img_for_display, input_tensor, 
                               true_label, ensemble_pred, ensemble_rejection_score, rejection_status, 
                               case_type, original_idx, save_dir, cam_method=None):
    """
    Create and display average Grad-CAM++ heatmap for ensemble.
    
    Args:
        ensemble_models: List of ensemble models
        image_path: Path to image file
        original_img_for_display: Original image for overlay
        input_tensor: Input tensor for models
        true_label: True label
        ensemble_pred: Ensemble prediction
        ensemble_rejection_score: Rejection score
        rejection_status: Status description
        case_type: Type of case being visualized
        original_idx: Original index
        save_dir: Directory to save visualization
        cam_method: CAM method to use (default: GradCAMPlusPlus)
    """
    if not GRADCAM_AVAILABLE:
        print("Grad-CAM not available. Skipping XAI visualization.")
        return
    
    if cam_method is None:
        cam_method = GradCAMPlusPlus
    
    print(f"Creating average Grad-CAM++ for Ensemble for {case_type} case (Original index: {original_idx})...")
    
    all_grayscale_cams = []
    for model in ensemble_models:
        model.eval()  # Ensure model is in eval mode
        cam_instance = cam_method(model=model, target_layers=[model.target_layer])
        targets = [ClassifierOutputTarget(ensemble_pred)]  # Target is class predicted by ensemble
        
        grayscale_cam = cam_instance(input_tensor=input_tensor, targets=targets)
        all_grayscale_cams.append(grayscale_cam[0, :])  # Store only heatmap (remove batch dimension)

    # Average the heatmaps
    if len(all_grayscale_cams) > 0:
        averaged_grayscale_cam = np.mean(all_grayscale_cams, axis=0)
    else:
        averaged_grayscale_cam = np.zeros_like(original_img_for_display[:,:,0])  # Fallback to black if no CAMs

    cam_image = show_cam_on_image(original_img_for_display, averaged_grayscale_cam, use_rgb=True)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img_for_display)
    plt.title(f"Original Image\nTrue: {true_label}, Ensemble Prediction: {ensemble_pred}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cam_image)
    plt.title(f"Ensemble Grad-CAM++ (Target: Class {ensemble_pred})\nRejection Score: {ensemble_rejection_score:.2f}, {rejection_status}")
    plt.axis('off')

    sanitized_case_type = case_type.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
    plt.suptitle(f"{sanitized_case_type} Case (Index: {original_idx}) - Ensemble Grad-CAM++ Explanation")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot and close it
    filepath = os.path.join(save_dir, f'ensemble_grad_cam_idx_{original_idx}_{sanitized_case_type}.png')
    plt.savefig(filepath)
    plt.show()
    plt.close()  # Close plot to avoid memory issues
    print(f"Ensemble Grad-CAM++ explanation for index {original_idx} saved to: {filepath}")
    print("Explanation: Average heatmap shows consensus regions for ensemble prediction.")


def visualize_xai_examples(cfg, ensemble_models, test_dataset, test_results, rejection_threshold):
    """
    Select representative cases and create XAI visualizations for them using Grad-CAM++.
    
    Args:
        cfg: Configuration object
        ensemble_models: List of ensemble models
        test_dataset: Test dataset
        test_results: Test results dictionary
        rejection_threshold: Rejection threshold
        
    Returns:
        generated_xai_image_paths: List of paths to generated XAI images
    """
    print("\n--- Creating XAI Visualizations for Selected Cases ---")

    # Ensure XAI save directory exists
    os.makedirs(cfg.XAI_SAVE_DIR, exist_ok=True)

    test_model_predictions = test_results['model_preds']
    test_rejection_scores = test_results['rejection_scores']
    test_true_labels = test_results['true_labels']
    test_original_indices = test_results['original_indices']
    rejected_categories = test_results['rejected_categories']

    # Filter different case types
    accepted_mask = test_rejection_scores >= rejection_threshold
    
    # Accepted and Correct
    accepted_correct_indices = np.where((accepted_mask) & (test_model_predictions == test_true_labels))[0]
    
    # Map from original dataset index to position in test_results flat arrays
    test_original_idx_to_pos = {idx: i for i, idx in enumerate(test_original_indices)}

    # List to store paths of generated XAI images
    generated_xai_image_paths = []

    # Different case examples
    cases_to_visualize = [
        ("Accepted_Correct", accepted_correct_indices, "Accepted & Correct"),
        ("Failure_Rejected", rejected_categories['failure_rejection_indices'], "Failure Rejected"), 
        ("Unknown_Ambiguous", rejected_categories['unknown_ambiguous_indices'], "Unknown/Ambiguous"),
        ("Potential_OOD", rejected_categories['potential_ood_indices'], "Potential OOD")
    ]

    for case_name, case_indices, case_description in cases_to_visualize:
        print(f"\n--- Example: {case_description} ---")
        if len(case_indices) > 0:
            # Get first case
            if case_name == "Accepted_Correct":
                sample_idx_in_test_data = case_indices[0]
                original_idx = test_original_indices[sample_idx_in_test_data]
            else:
                original_idx = case_indices[0]
                sample_idx_in_test_data = test_original_idx_to_pos[original_idx]
            
            image_path = test_dataset.image_paths[test_dataset.original_indices_map[original_idx]]
            true_label = test_true_labels[sample_idx_in_test_data]
            model_pred = test_model_predictions[sample_idx_in_test_data]
            rejection_score = test_rejection_scores[sample_idx_in_test_data]
            
            original_img_for_display, input_tensor = preprocess_for_xai(image_path, cfg)
            
            print(f"Selected {case_description} (Original index: {original_idx})")
            
            filepath_case = os.path.join(cfg.XAI_SAVE_DIR, f'ensemble_grad_cam_idx_{original_idx}_{case_name}.png')
            
            rejection_status = "Accepted" if case_name == "Accepted_Correct" else f"Rejected ({case_description.split(' ')[-1]})"
            
            visualize_ensemble_grad_cam(ensemble_models, image_path, original_img_for_display, input_tensor, 
                                      true_label, model_pred, rejection_score, rejection_status, 
                                      case_description, original_idx, cfg.XAI_SAVE_DIR)
            generated_xai_image_paths.append(filepath_case)
        else:
            print(f"No {case_description} cases found for XAI visualization.")
        
    print(f"\nAll XAI images saved to: {cfg.XAI_SAVE_DIR}")
    return generated_xai_image_paths


def plot_calibration_curve(model_predictions, rejection_scores, true_labels, num_bins=10, save_path=None):
    """
    Plot calibration curve (Reliability Diagram) to evaluate calibration ability.
    
    Args:
        model_predictions: Model predictions
        rejection_scores: Rejection/confidence scores
        true_labels: True labels
        num_bins: Number of bins
        save_path: Path to save plot
    """
    if len(rejection_scores) == 0:
        print("No score data to plot calibration curve.")
        return

    bins = np.linspace(0., 1., num_bins + 1)
    bin_accuracies = []
    bin_rejection_scores = []
    bin_counts = []

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
            bin_rejection_score_mean = np.mean(rejection_scores[bin_samples_indices])
            bin_accuracies.append(bin_accuracy)
            bin_rejection_scores.append(bin_rejection_score_mean)
            bin_counts.append(bin_count)
        else:
            bin_accuracies.append(np.nan)
            bin_rejection_scores.append(np.nan)
            bin_counts.append(0)

    # Filter empty bins
    valid_bins_mask = ~np.isnan(bin_accuracies)
    bin_accuracies = np.array(bin_accuracies)[valid_bins_mask]
    bin_rejection_scores = np.array(bin_rejection_scores)[valid_bins_mask]

    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.plot(bin_rejection_scores, bin_accuracies, marker='o', linestyle='-', color='blue', label='Model')
    
    # Draw bars
    for i_idx in range(len(bin_rejection_scores)):
        plt.plot([bin_rejection_scores[i_idx], bin_rejection_scores[i_idx]], 
                 [bin_rejection_scores[i_idx], bin_accuracies[i_idx]],
                 color='red' if bin_accuracies[i_idx] < bin_rejection_scores[i_idx] else 'green', 
                 linestyle='-', linewidth=2)

    plt.xlabel("Average Score")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.grid(True)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    if save_path:
        plt.savefig(save_path)
        print(f"Calibration curve saved to: {save_path}")
    plt.show()
    plt.close()


def plot_roc_curve(model_predictions, rejection_scores, true_labels, save_path=None):
    """
    Plot ROC curve (Receiver Operating Characteristic).
    
    Args:
        model_predictions: Model predictions
        rejection_scores: Rejection/confidence scores
        true_labels: True labels
        save_path: Path to save plot
    """
    if len(rejection_scores) == 0:
        print("No score data to plot ROC curve.")
        return

    is_correct = (model_predictions == true_labels).astype(int)
    
    if len(np.unique(is_correct)) < 2:
        print("Insufficient variation in correct/incorrect predictions to plot ROC curve.")
        return

    fpr, tpr, thresholds = roc_curve(is_correct, rejection_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to: {save_path}")
    plt.show()
    plt.close()


def plot_pr_curve(model_predictions, rejection_scores, true_labels, save_path=None):
    """
    Plot PR curve (Precision-Recall).
    
    Args:
        model_predictions: Model predictions
        rejection_scores: Rejection/confidence scores
        true_labels: True labels
        save_path: Path to save plot
    """
    if len(rejection_scores) == 0:
        print("No score data to plot PR curve.")
        return

    is_correct = (model_predictions == true_labels).astype(int)

    if len(np.unique(is_correct)) < 2:
        print("Insufficient variation in correct/incorrect predictions to plot PR curve.")
        return

    precision, recall, thresholds = precision_recall_curve(is_correct, rejection_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    if save_path:
        plt.savefig(save_path)
        print(f"PR curve saved to: {save_path}")
    plt.show()
    plt.close()


def plot_risk_coverage_curve(model_predictions, rejection_scores, true_labels, save_path=None):
    """
    Plot Risk-Coverage curve.
    
    Args:
        model_predictions: Model predictions
        rejection_scores: Rejection/confidence scores
        true_labels: True labels
        save_path: Path to save plot
    """
    if len(rejection_scores) == 0:
        print("No score data to plot Risk-Coverage curve.")
        return

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
    coverages = np.array(coverages[::-1]) * 100 
    risks = np.array(risks[::-1]) * 100       

    plt.figure(figsize=(8, 6))
    plt.plot(coverages, risks, lw=2, color='blue', label='Risk-Coverage')
    plt.xlabel("Coverage (%)")
    plt.ylabel("Risk (%)")
    plt.title("Risk-Coverage Curve")
    plt.grid(True)
    plt.legend()
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    if save_path:
        plt.savefig(save_path)
        print(f"Risk-Coverage curve saved to: {save_path}")
    plt.show()
    plt.close()


def plot_all_calibration_curves(all_results, save_dir, num_bins=10):
    """
    Plot calibration curves for all baselines on a single figure.
    
    Args:
        all_results: List of result dictionaries from run_baseline
        save_dir: Directory to save plot
        num_bins: Number of bins for calibration curves
    """
    plt.figure(figsize=(20, 10))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

    for result in all_results:
        try:
            config_name = result['metrics']['Config Name']
            preds = result['raw_data']['predictions']
            rejection_scores = result['raw_data']['rejection_scores']
            labels = result['raw_data']['true_labels']
        except KeyError as e:
            print(f"Error accessing data for result: {e}")
            continue

        if not all(isinstance(arr, np.ndarray) for arr in [preds, rejection_scores, labels]):
            print(f"Error: Input data for {config_name} must be np.ndarray.")
            continue

        if len(rejection_scores) == 0:
            print(f"No confidence data to plot calibration curve for {config_name}.")
            continue

        bins = np.linspace(0., 1., num_bins + 1)
        bin_accuracies = []
        bin_rejection_scores = []

        for i in range(num_bins):
            lower_bound = bins[i]
            upper_bound = bins[i + 1]
            mask = (rejection_scores >= lower_bound) & (rejection_scores < upper_bound)
            if i == num_bins - 1:
                mask = (rejection_scores >= lower_bound) & (rejection_scores <= upper_bound)

            bin_indices = np.where(mask)[0]
            if len(bin_indices) > 0:
                bin_accuracy = accuracy_score(labels[bin_indices], preds[bin_indices])
                bin_rejection_score_mean = np.mean(rejection_scores[bin_indices])
                bin_accuracies.append(bin_accuracy)
                bin_rejection_scores.append(bin_rejection_score_mean)
            else:
                bin_accuracies.append(np.nan)
                bin_rejection_scores.append(np.nan)

        valid_bins_mask = ~np.isnan(bin_accuracies)
        bin_accuracies = np.array(bin_accuracies)[valid_bins_mask]
        bin_rejection_scores = np.array(bin_rejection_scores)[valid_bins_mask]

        plt.plot(bin_rejection_scores, bin_accuracies, marker='o', linestyle='-', label=config_name)

    plt.xlabel("Average Score")
    plt.ylabel("Accuracy")
    plt.title("Calibration Curves for All Baselines")
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_reliability_diagrams.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Combined calibration curves saved to: {save_path}")
    plt.show()
    plt.close()


def plot_all_roc_curves(all_results, save_dir):
    """
    Plot ROC curves for all baselines on a single figure.
    
    Args:
        all_results: List of result dictionaries
        save_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    for result in all_results:
        config_name = result['metrics']['Config Name']
        preds = result['raw_data']['predictions']
        rejection_scores = result['raw_data']['rejection_scores']
        labels = result['raw_data']['true_labels']

        if len(rejection_scores) == 0:
            print(f"No score data to plot ROC curve for {config_name}.")
            continue
        
        is_correct = (preds == labels).astype(int)
        if len(np.unique(is_correct)) < 2:
            print(f"Insufficient correct/incorrect variation to plot ROC curve for {config_name}.")
            continue

        fpr, tpr, _ = roc_curve(is_correct, rejection_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{config_name} (AUC = {roc_auc:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Baselines')
    plt.legend(loc="lower right", bbox_to_anchor=(1, 0))
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_roc_curves.png')
    plt.savefig(save_path)
    print(f"Combined ROC curves saved to: {save_path}")
    plt.show()
    plt.close()


def plot_all_risk_coverage_curves(all_results, save_dir):
    """
    Plot Risk-Coverage curves for all baselines on a single figure.
    
    Args:
        all_results: List of result dictionaries
        save_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 8))

    for result in all_results:
        config_name = result['metrics']['Config Name']
        rejection_scores = result['raw_data']['rejection_scores']
        preds = result['raw_data']['predictions']
        labels = result['raw_data']['true_labels']

        num_total = len(labels)
        if num_total == 0:
            print(f"No data to plot Risk-Coverage curve for {config_name}.")
            continue

        # Sort samples by score in increasing order to simulate increasing rejection
        sorted_indices = np.argsort(rejection_scores)
        sorted_preds = preds[sorted_indices]
        sorted_labels = labels[sorted_indices]

        risks = []
        coverages = []
        for i_idx in range(num_total):
            current_accepted_preds = sorted_preds[i_idx:]
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
        coverages = np.array(coverages[::-1]) * 100 
        risks = np.array(risks[::-1]) * 100       

        plt.plot(coverages, risks, lw=2, label=f'{config_name} (AURC = {result["metrics"]["AURC"]:.4f})')

    plt.xlabel("Coverage (%)")
    plt.ylabel("Risk (%)")
    plt.title("Risk-Coverage Curves for All Baselines")
    plt.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'all_risk_coverage_curves.png')
    plt.savefig(save_path)
    print(f"Combined Risk-Coverage curves saved to: {save_path}")
    plt.show()
    plt.close() 