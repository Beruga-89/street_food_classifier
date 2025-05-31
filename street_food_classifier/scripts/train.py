#!/usr/bin/env python3
"""
Training script for Street Food Classifier.

This script demonstrates how to use the modular framework for training
a classification model with different architectures and configurations.

Example usage:
    python scripts/train.py --architecture resnet18 --epochs 20 --batch_size 32
    python scripts/train.py --architecture custom_cnn --learning_rate 0.001
    python scripts/train.py --config configs/experiment_1.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.street_food_classifier import StreetFoodClassifier
from config import Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Street Food Classifier')
    
    # Model arguments
    parser.add_argument('--architecture', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'efficientnet_b0', 'custom_cnn', 'mobilenet_v2'],
                       help='Model architecture to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone layers (only train classifier)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=None,
                       help='Early stopping patience')
    
    # Data arguments
    parser.add_argument('--data_folder', type=str, default=None,
                       help='Path to data folder')
    parser.add_argument('--img_size', type=int, default=None,
                       help='Image size for training')
    
    # Output arguments
    parser.add_argument('--model_name', type=str, default=None,
                       help='Name for saved model and results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Flags
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save training results automatically')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Do not create visualizations')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmark after training')
    
    return parser.parse_args()


def update_config_from_args(config, args):
    """Update configuration with command line arguments."""
    # Training parameters
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
    if args.patience is not None:
        config.PATIENCE = args.patience
    
    # Data parameters
    if args.data_folder is not None:
        config.DATA_FOLDER = Path(args.data_folder)
    if args.img_size is not None:
        config.IMG_SIZE = args.img_size
    
    # Output parameters
    if args.output_dir is not None:
        config.OUTPUT_FOLDER = args.output_dir
        config.MODEL_FOLDER = f"{args.output_dir}/models"
        config.PLOT_FOLDER = f"{args.output_dir}/plots"
    
    # Seed
    config.SEED = args.seed
    
    return config


def main():
    """Main training function."""
    args = parse_args()
    
    print("🚀 Starting Street Food Classifier Training")
    print("=" * 60)
    
    # Load configuration
    if args.config:
        # Load from file (would need implementation)
        raise NotImplementedError("Config file loading not implemented yet")
    else:
        config = Config()
    
    # Update config with command line arguments
    config = update_config_from_args(config, args)
    
    # Print configuration
    print("Configuration:")
    print(f"  Architecture: {args.architecture}")
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Image Size: {config.IMG_SIZE}")
    print(f"  Data Folder: {config.DATA_FOLDER}")
    print(f"  Random Seed: {config.SEED}")
    print()
    
    try:
        # Initialize classifier
        print("🔧 Initializing classifier...")
        classifier = StreetFoodClassifier(config)
        
        # Print status
        print(classifier)
        print()
        
        # Training
        print("🎯 Starting training...")
        model_kwargs = {}
        if args.architecture == 'custom_cnn':
            model_kwargs['dropout_rate'] = 0.5
        
        history = classifier.train(
            architecture=args.architecture,
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone,
            save_results=not args.no_save,
            **model_kwargs
        )
        
        # Model summary
        print("\n📊 Model Summary:")
        model_summary = classifier.get_model_summary()
        print(f"  Parameters: {model_summary['total_params']:,}")
        print(f"  Trainable: {model_summary['trainable_params']:,}")
        print(f"  Size: {model_summary['model_size_mb']:.1f} MB")
        
        # Training summary
        print("\n📈 Training Summary:")
        training_summary = classifier.get_training_summary()
        print(f"  Best Accuracy: {training_summary['best_accuracy']:.4f}")
        print(f"  Best F1 Score: {training_summary['best_f1']:.4f}")
        print(f"  Best Loss: {training_summary['best_loss']:.4f}")
        print(f"  Epochs Trained: {training_summary['total_epochs_trained']}")
        
        # Evaluation
        print("\n🔍 Running evaluation...")
        eval_results = classifier.evaluate(create_visualizations=not args.no_visualize)
        
        # Save model for deployment
        if not args.no_save:
            model_name = args.model_name or f"{args.architecture}_trained"
            deployment_path = classifier.export_model_for_deployment(
                f"{model_name}_deployment.pth"
            )
            print(f"\n💾 Model exported for deployment: {deployment_path}")
        
        # Performance benchmark
        if args.benchmark and config.DATA_FOLDER.exists():
            print("\n⚡ Running performance benchmark...")
            # Use validation data for benchmark
            benchmark_results = classifier.benchmark_performance(
                str(config.DATA_FOLDER), 
                num_samples=100  # Limit for quick benchmark
            )
        
        # Create comprehensive report
        if not args.no_save:
            print("\n📋 Creating comprehensive report...")
            report_name = args.model_name or f"{args.architecture}_analysis"
            report_path = classifier.create_comprehensive_report(report_name)
            print(f"   Report created: {report_path}")
        
        print("\n✅ Training completed successfully!")
        
        # Final status
        print("\n" + "=" * 60)
        print("FINAL STATUS")
        print("=" * 60)
        print(classifier)
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if 'classifier' in locals():
            classifier.cleanup()


if __name__ == "__main__":
    main()