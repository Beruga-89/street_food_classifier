#!/usr/bin/env python3
"""
Evaluation script for Street Food Classifier.

This script evaluates saved models and creates comprehensive analysis reports.

Example usage:
    python scripts/evaluate.py --model models/best_f1_model.pth
    python scripts/evaluate.py --model_folder models/ --compare_all
    python scripts/evaluate.py --saved_results results.json --visualize
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.street_food_classifier import StreetFoodClassifier
from src.evaluation import StandaloneModelEvaluator, EvaluationWorkflow
from config import Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Street Food Classifier')
    
    # Model loading
    parser.add_argument('--model', type=str, default=None,
                       help='Path to saved model file (.pth)')
    parser.add_argument('--model_folder', type=str, default=None,
                       help='Path to folder containing multiple models')
    parser.add_argument('--architecture', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'efficientnet_b0', 'custom_cnn', 'mobilenet_v2'],
                       help='Model architecture')
    
    # Evaluation options
    parser.add_argument('--eval_train', action='store_true',
                       help='Also evaluate on training set')
    parser.add_argument('--compare_all', action='store_true',
                       help='Compare all models in folder')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark')
    
    # Saved results
    parser.add_argument('--saved_results', type=str, default=None,
                       help='Path to saved evaluation results (.json/.pkl)')
    parser.add_argument('--list_results', action='store_true',
                       help='List all available saved results')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Create visualizations')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Skip visualizations')
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='Save plots to files')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--report_name', type=str, default=None,
                       help='Name for evaluation report')
    
    # Data
    parser.add_argument('--data_folder', type=str, default=None,
                       help='Path to data folder (if different from config)')
    
    return parser.parse_args()


def evaluate_single_model(args, config):
    """Evaluate a single model."""
    print(f"🔍 Evaluating single model: {args.model}")
    
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return False
    
    # Use StandaloneModelEvaluator for comprehensive evaluation
    evaluator = StandaloneModelEvaluator(config)
    
    model_name = Path(args.model).stem
    results = evaluator.evaluate_saved_model(
        model_path=args.model,
        model_name=model_name,
        architecture=args.architecture,
        evaluate_train=args.eval_train,
        evaluate_val=True,
        visualize=args.visualize and not args.no_visualize,
        save_results=True
    )
    
    print("\n📊 Evaluation Results:")
    if 'validation' in results:
        val_metrics = results['validation']
        print(f"  Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Validation F1-Score: {val_metrics['f1']:.4f}")
        print(f"  Validation Loss: {val_metrics['loss']:.4f}")
    
    if 'training' in results:
        train_metrics = results['training']
        print(f"  Training Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"  Training F1-Score: {train_metrics['f1']:.4f}")
        print(f"  Training Loss: {train_metrics['loss']:.4f}")
    
    return True


def evaluate_multiple_models(args, config):
    """Evaluate multiple models and compare them."""
    model_folder = Path(args.model_folder)
    
    if not model_folder.exists():
        print(f"❌ Model folder not found: {model_folder}")
        return False
    
    # Find all model files
    model_files = list(model_folder.glob("*.pth")) + list(model_folder.glob("*.pt"))
    
    if not model_files:
        print(f"❌ No model files found in {model_folder}")
        return False
    
    print(f"🔍 Found {len(model_files)} models to evaluate:")
    for model_file in model_files:
        print(f"  - {model_file.name}")
    
    # Use StandaloneModelEvaluator for batch evaluation
    evaluator = StandaloneModelEvaluator(config)
    
    # Create model configurations
    model_configs = []
    for model_file in model_files:
        config_entry = {
            'path': str(model_file),
            'name': model_file.stem,
            'arch': args.architecture
        }
        model_configs.append(config_entry)
    
    # Compare all models
    comparison_results = evaluator.compare_multiple_models(
        model_configs,
        save_comparison=True,
        visualize_comparison=args.visualize and not args.no_visualize
    )
    
    print("\n📊 Model Comparison Summary:")
    for result in comparison_results['individual_results']:
        if 'validation' in result:
            val_metrics = result['validation']
            model_name = result.get('model_info', {}).get('model_name', 'Unknown')
            print(f"  {model_name}: Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}")
    
    return True


def visualize_saved_results(args, config):
    """Visualize previously saved evaluation results."""
    print(f"📈 Visualizing saved results: {args.saved_results}")
    
    if not Path(args.saved_results).exists():
        print(f"❌ Results file not found: {args.saved_results}")
        return False
    
    workflow = EvaluationWorkflow(config)
    
    # Load and visualize
    plots = workflow.load_and_visualize(
        args.saved_results,
        create_confusion_matrix=True,
        create_classification_report=True,
        save_plots=args.save_plots,
        show_plots=args.visualize and not args.no_visualize
    )
    
    print("✅ Visualization completed")
    if plots.get('confusion_matrix'):
        print(f"  Confusion matrix: {plots['confusion_matrix']}")
    
    return True


def list_available_results(config):
    """List all available saved evaluation results."""
    print("📋 Available saved evaluation results:")
    
    workflow = EvaluationWorkflow(config)
    available_results = workflow.metrics_manager.list_saved_results()
    
    if not available_results:
        print("  No saved results found.")
        return False
    
    print(f"  Found {len(available_results)} saved results")
    return True


def run_benchmark(args, config):
    """Run performance benchmark."""
    if not args.model:
        print("❌ Model path required for benchmark")
        return False
    
    print("⚡ Running performance benchmark...")
    
    # Load model and run benchmark
    classifier = StreetFoodClassifier(config)
    classifier.load_model(args.model, args.architecture)
    
    # Use data folder from config or args
    data_path = args.data_folder or config.DATA_FOLDER
    
    if not Path(data_path).exists():
        print(f"❌ Data folder not found: {data_path}")
        return False
    
    benchmark_results = classifier.benchmark_performance(
        str(data_path),
        num_samples=200  # Limit for reasonable benchmark time
    )
    
    return True


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("🔍 Starting Street Food Classifier Evaluation")
    print("=" * 60)
    
    # Load configuration
    config = Config()
    
    # Update config with command line arguments
    if args.data_folder:
        config.DATA_FOLDER = Path(args.data_folder)
    if args.output_dir:
        config.OUTPUT_FOLDER = args.output_dir
        config.EVALUATION_FOLDER = f"{args.output_dir}/evaluation_results"
        config.PLOT_FOLDER = f"{args.output_dir}/plots"
    
    # Handle visualization flags
    if args.no_visualize:
        args.visualize = False
    
    print("Configuration:")
    print(f"  Data Folder: {config.DATA_FOLDER}")
    print(f"  Output Folder: {config.OUTPUT_FOLDER}")
    print(f"  Create Visualizations: {args.visualize}")
    print()
    
    success = False
    
    try:
        # List available results
        if args.list_results:
            success = list_available_results(config)
        
        # Visualize saved results
        elif args.saved_results:
            success = visualize_saved_results(args, config)
        
        # Multiple model evaluation
        elif args.model_folder:
            success = evaluate_multiple_models(args, config)
        
        # Single model evaluation
        elif args.model:
            success = evaluate_single_model(args, config)
            
            # Optional benchmark
            if args.benchmark and success:
                run_benchmark(args, config)
        
        else:
            print("❌ No evaluation target specified!")
            print("Use --model, --model_folder, --saved_results, or --list_results")
            return
        
        if success:
            print("\n✅ Evaluation completed successfully!")
        else:
            print("\n❌ Evaluation failed!")
    
    except KeyboardInterrupt:
        print("\n❌ Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()