#!/usr/bin/env python3
"""
Main execution script for XAI4LtR framework.

This script demonstrates how to use the XAI4LtR framework to run different baselines
for selective classification with explainable AI methods.

Usage:
    python main.py --data_dir /path/to/data --output_dir /path/to/output
"""

import os
import argparse
import pandas as pd
from datetime import datetime

from src import (
    Config, BaseClassifier, prepare_datasets, train_ensemble,
    get_rejection_scores_and_predictions, find_optimal_rejection_threshold,
    calculate_metrics, categorize_rejected_cases, visualize_xai_examples,
    plot_all_calibration_curves, plot_all_roc_curves, plot_all_risk_coverage_curves,
    set_seed, print_system_info, create_save_directories
)


def run_baseline(config_name, mcdo_enable, label_smoothing_enable, 
                 calibration_method, final_overall_learning_metrics, 
                 ood_detection_method='none', combine_ood_with_disagreement=False,
                 enable_training_dynamics=False, cfg=None, train_loader=None, 
                 val_loader=None, test_loader=None, train_dataset=None):
    """
    Run a specific baseline configuration.
    
    Args:
        config_name: Name of the baseline configuration
        mcdo_enable: Whether to enable Monte Carlo Dropout
        label_smoothing_enable: Whether to enable Label Smoothing
        calibration_method: Calibration method to use
        final_overall_learning_metrics: Training dynamics metrics
        ood_detection_method: OOD detection method
        combine_ood_with_disagreement: Whether to combine OOD with disagreement
        enable_training_dynamics: Whether to enable training dynamics adjustment
        cfg: Configuration object
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        train_dataset: Training dataset
        
    Returns:
        Dictionary containing results and metrics
    """
    print(f"\n{'='*20}")
    print(f"Starting Baseline: {config_name}")
    print(f"{'='*20}")

    # Reset Config to default state before each run
    if cfg is None:
        cfg = Config()
    set_seed(cfg.RANDOM_SEED)

    # Configure flags for current baseline
    cfg.MCDO_ENABLE = mcdo_enable
    cfg.LABEL_SMOOTHING_ENABLE = label_smoothing_enable
    cfg.ENABLE_TRAINING_DYNAMICS = enable_training_dynamics

    # Get ensemble predictions and rejection scores for validation set
    print(f"\nGetting predictions and rejection scores for Validation Set "
          f"(Using calibration: {calibration_method}, OOD: {ood_detection_method}, "
          f"Combine disagreement: {combine_ood_with_disagreement})...")
    
    val_model_predictions, val_rejection_scores, val_true_labels, val_original_indices, _, _, _ = (
        get_rejection_scores_and_predictions(cfg, val_loader, cfg.MODEL_SAVE_DIR, 
                                            final_overall_learning_metrics, train_dataset, 
                                            calibration_method=calibration_method,
                                            ood_detection_method=ood_detection_method,
                                            combine_ood_with_disagreement=combine_ood_with_disagreement))

    # Find optimal rejection threshold on validation set
    best_rejection_threshold, _ = find_optimal_rejection_threshold(
        val_rejection_scores, val_model_predictions, val_true_labels, cfg
    )
    print(f"Final rejection threshold selected: {best_rejection_threshold:.4f}")

    # Evaluate on test set using learned threshold
    print(f"\n--- Evaluation on Test Set (Using calibration: {calibration_method}, "
          f"OOD: {ood_detection_method}, Combine disagreement: {combine_ood_with_disagreement}) ---")
    
    (test_model_predictions, test_rejection_scores, test_true_labels, test_original_indices,
     all_ensemble_individual_probs_test, test_odin_scores, test_energy_scores) = (
        get_rejection_scores_and_predictions(cfg, test_loader, cfg.MODEL_SAVE_DIR, 
                                            final_overall_learning_metrics, train_dataset, 
                                            calibration_method=calibration_method,
                                            ood_detection_method=ood_detection_method,
                                            combine_ood_with_disagreement=combine_ood_with_disagreement))

    test_metrics = calculate_metrics(test_model_predictions, test_rejection_scores, 
                                   test_true_labels, best_rejection_threshold, verbose=True)

    # Categorize rejected cases
    rejected_categories_info = categorize_rejected_cases(
        test_rejection_scores, test_model_predictions, test_true_labels, test_original_indices,
        best_rejection_threshold, all_ensemble_individual_probs_test, 
        all_odin_scores=test_odin_scores, all_energy_scores=test_energy_scores,
        ood_detection_method=ood_detection_method
    )

    print("\n--- Summary of Rejected Case Categorization ---")
    print(f"Failure Rejection Cases: {len(rejected_categories_info['failure_rejection_indices'])}")
    print(f"Unknown/Ambiguous Rejection Cases: {len(rejected_categories_info['unknown_ambiguous_indices'])}") 
    print(f"Potential OOD Rejection Cases: {len(rejected_categories_info['potential_ood_indices'])}") 

    return {
        'metrics': {
            'Config Name': config_name,
            'Overall Accuracy': test_metrics['overall_accuracy'],
            'Accuracy on Accepted': test_metrics['accuracy_on_accepted'],
            'Coverage': test_metrics['coverage'],
            'Rejection Rate': test_metrics['rejection_rate'],
            'Risk': test_metrics['risk'],
            'ECE': test_metrics['ece'],
            'NLL Accepted': test_metrics['nll_accepted'],
            'Brier Accepted': test_metrics['brier_accepted'],
            'AUROC': test_metrics['auroc_correct_incorrect'],
            'AUPR': test_metrics['aupr_correct_incorrect'],
            'AURC': test_metrics['aurc'],
            'F1 Rejection': test_metrics['f1_rejection'],
            'Failure Rejected Count': len(rejected_categories_info['failure_rejection_indices']),
            'Unknown/Ambiguous Rejected Count': len(rejected_categories_info['unknown_ambiguous_indices']),
            'Potential OOD Rejected Count': len(rejected_categories_info['potential_ood_indices'])
        },
        'raw_data': {
            'predictions': test_model_predictions,
            'rejection_scores': test_rejection_scores,
            'true_labels': test_true_labels
        }
    }


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='XAI4LtR: Explainable Learning to Reject Framework')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to output directory')
    parser.add_argument('--covid_dir', type=str, help='Path to COVID images directory')
    parser.add_argument('--non_covid_dir', type=str, help='Path to non-COVID images directory')
    parser.add_argument('--baselines', nargs='+', default=['all'], 
                       help='Baselines to run (default: all)')
    parser.add_argument('--skip_training', action='store_true', 
                       help='Skip ensemble training (use pre-trained models)')
    parser.add_argument('--skip_xai', action='store_true', 
                       help='Skip XAI visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"xai4ltr_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize configuration
    cfg = Config(
        data_dir=args.data_dir,
        model_save_dir=os.path.join(output_dir, 'models'),
        xai_save_dir=os.path.join(output_dir, 'xai_visualizations')
    )
    
    # Update data paths if provided
    if args.covid_dir:
        cfg.COVID_DIR = args.covid_dir
    if args.non_covid_dir:
        cfg.NON_COVID_DIR = args.non_covid_dir
    
    # Create save directories
    create_save_directories(cfg)
    
    # Print system information
    print_system_info(cfg)
    
    # Prepare datasets
    print("\n--- Preparing Datasets ---")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = prepare_datasets(cfg)
    
    # Train ensemble (unless skipped)
    final_overall_learning_metrics = None
    if not args.skip_training:
        print("\n--- Training Ensemble Models ---")
        final_overall_learning_metrics = train_ensemble(cfg, train_loader, val_loader)
    else:
        print("\n--- Skipping Ensemble Training (using pre-trained models) ---")
    
    # Define baseline configurations
    baseline_configs = {
        'Baseline 0': {
            'mcdo_enable': False,
            'label_smoothing_enable': False,
            'calibration_method': 'temperature_scaling',
            'ood_detection_method': 'none',
            'combine_ood_with_disagreement': False,
            'enable_training_dynamics': False
        },
        'Baseline A.1': {
            'mcdo_enable': True,
            'label_smoothing_enable': False,
            'calibration_method': 'temperature_scaling',
            'ood_detection_method': 'none',
            'combine_ood_with_disagreement': False,
            'enable_training_dynamics': False
        },
        'Baseline A.2.1': {
            'mcdo_enable': False,
            'label_smoothing_enable': False,
            'calibration_method': 'isotonic_regression',
            'ood_detection_method': 'none',
            'combine_ood_with_disagreement': False,
            'enable_training_dynamics': False
        },
        'Baseline A.2.2': {
            'mcdo_enable': False,
            'label_smoothing_enable': False,
            'calibration_method': 'beta_calibration',
            'ood_detection_method': 'none',
            'combine_ood_with_disagreement': False,
            'enable_training_dynamics': False
        },
        'Baseline A.2.3': {
            'mcdo_enable': False,
            'label_smoothing_enable': True,
            'calibration_method': 'temperature_scaling',
            'ood_detection_method': 'none',
            'combine_ood_with_disagreement': False,
            'enable_training_dynamics': False
        },
        'Baseline B.1.1': {
            'mcdo_enable': False,
            'label_smoothing_enable': False,
            'calibration_method': 'temperature_scaling',
            'ood_detection_method': 'odin',
            'combine_ood_with_disagreement': False,
            'enable_training_dynamics': False
        },
        'Baseline B.1.2': {
            'mcdo_enable': False,
            'label_smoothing_enable': False,
            'calibration_method': 'temperature_scaling',
            'ood_detection_method': 'odin',
            'combine_ood_with_disagreement': True,
            'enable_training_dynamics': False
        },
        'Baseline B.2.1': {
            'mcdo_enable': False,
            'label_smoothing_enable': False,
            'calibration_method': 'temperature_scaling',
            'ood_detection_method': 'energy',
            'combine_ood_with_disagreement': False,
            'enable_training_dynamics': False
        },
        'Baseline B.2.2': {
            'mcdo_enable': False,
            'label_smoothing_enable': False,
            'calibration_method': 'temperature_scaling',
            'ood_detection_method': 'energy',
            'combine_ood_with_disagreement': True,
            'enable_training_dynamics': False
        },
        'Baseline B.3': {
            'mcdo_enable': False,
            'label_smoothing_enable': False,
            'calibration_method': 'temperature_scaling',
            'ood_detection_method': 'none',
            'combine_ood_with_disagreement': False,
            'enable_training_dynamics': True
        }
    }
    
    # Determine which baselines to run
    if 'all' in args.baselines:
        baselines_to_run = list(baseline_configs.keys())
    else:
        baselines_to_run = [b for b in args.baselines if b in baseline_configs]
    
    # Run selected baselines
    all_results = []
    for baseline_name in baselines_to_run:
        if baseline_name in baseline_configs:
            config = baseline_configs[baseline_name]
            result = run_baseline(
                config_name=baseline_name,
                mcdo_enable=config['mcdo_enable'],
                label_smoothing_enable=config['label_smoothing_enable'],
                calibration_method=config['calibration_method'],
                final_overall_learning_metrics=final_overall_learning_metrics,
                ood_detection_method=config['ood_detection_method'],
                combine_ood_with_disagreement=config['combine_ood_with_disagreement'],
                enable_training_dynamics=config['enable_training_dynamics'],
                cfg=cfg, train_loader=train_loader, val_loader=val_loader, 
                test_loader=test_loader, train_dataset=train_dataset
            )
            all_results.append(result)
        else:
            print(f"Warning: Unknown baseline '{baseline_name}'")
    
    # Create comparative visualizations
    print("\n--- Creating Comparative Visualizations ---")
    plot_all_calibration_curves(all_results, cfg.XAI_SAVE_DIR)
    plot_all_roc_curves(all_results, cfg.XAI_SAVE_DIR)
    plot_all_risk_coverage_curves(all_results, cfg.XAI_SAVE_DIR)
    
    # Create XAI visualizations (if not skipped)
    if not args.skip_xai and len(all_results) > 0:
        print("\n--- Creating XAI Visualizations ---")
        # Use the first baseline's results for XAI visualization
        first_result = all_results[0]
        test_results = {
            'model_preds': first_result['raw_data']['predictions'],
            'rejection_scores': first_result['raw_data']['rejection_scores'],
            'true_labels': first_result['raw_data']['true_labels'],
            'original_indices': range(len(first_result['raw_data']['predictions'])),
            'rejected_categories': {
                'failure_rejection_indices': [],
                'unknown_ambiguous_indices': [],
                'potential_ood_indices': []
            }
        }
        
        # Load ensemble models for XAI
        from src.utils import load_ensemble_models
        ensemble_models = load_ensemble_models(cfg)
        
        # Find threshold for XAI visualization
        _, val_rejection_scores, _, _ = get_rejection_scores_and_predictions(
            cfg, val_loader, cfg.MODEL_SAVE_DIR, final_overall_learning_metrics, train_dataset
        )
        threshold, _ = find_optimal_rejection_threshold(
            val_rejection_scores, first_result['raw_data']['predictions'], 
            first_result['raw_data']['true_labels'], cfg
        )
        
        visualize_xai_examples(cfg, ensemble_models, test_dataset, test_results, threshold)
    
    # Save results summary
    print("\n--- Saving Results Summary ---")
    results_df = pd.DataFrame([result['metrics'] for result in all_results])
    results_path = os.path.join(output_dir, 'results_summary.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results summary saved to: {results_path}")
    
    # Print final summary
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETED")
    print("="*50)
    print(f"Output directory: {output_dir}")
    print(f"Baselines run: {len(all_results)}")
    print(f"Results saved to: {results_path}")
    print("="*50)


if __name__ == "__main__":
    main() 