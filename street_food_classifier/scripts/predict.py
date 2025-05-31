#!/usr/bin/env python3
"""
Prediction script for Street Food Classifier.

This script makes predictions on single images, batch images, or entire folders.

Example usage:
    python scripts/predict.py --model models/best_f1_model.pth --image test.jpg
    python scripts/predict.py --model models/best_f1_model.pth --folder test_images/
    python scripts/predict.py --model models/best_f1_model.pth --batch img1.jpg img2.jpg img3.jpg
    python scripts/predict.py --model models/best_f1_model.pth --interactive
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.street_food_classifier import StreetFoodClassifier
from config import Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with Street Food Classifier')
    
    # Model
    parser.add_argument('--model', type=str, required=True,
                       help='Path to saved model file (.pth)')
    parser.add_argument('--architecture', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'efficientnet_b0', 'custom_cnn', 'mobilenet_v2'],
                       help='Model architecture')
    
    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str,
                            help='Single image to predict')
    input_group.add_argument('--folder', type=str,
                            help='Folder containing images to predict')
    input_group.add_argument('--batch', type=str, nargs='+',
                            help='Multiple image files to predict')
    input_group.add_argument('--interactive', action='store_true',
                            help='Start interactive prediction session')
    
    # Prediction options
    parser.add_argument('--confidence_threshold', type=float, default=None,
                       help='Confidence threshold for predictions')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for folder prediction')
    parser.add_argument('--recursive', action='store_true',
                       help='Search folder recursively')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (.csv or .json)')
    parser.add_argument('--show_top_k', type=int, default=3,
                       help='Show top K predictions')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed prediction information')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimize output')
    
    # Data folder (for setup)
    parser.add_argument('--data_folder', type=str, default=None,
                       help='Path to data folder (for class names)')
    
    return parser.parse_args()


def predict_single_image(classifier, image_path, args):
    """Predict single image."""
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    print(f"🔍 Predicting: {image_path}")
    
    try:
        result = classifier.predict(image_path)
        
        if not args.quiet:
            confidence_indicator = "✅" if result['is_confident'] else "❓"
            print(f"\n{confidence_indicator} Prediction Results:")
            print(f"  File: {Path(image_path).name}")
            print(f"  Predicted Class: {result['class_name']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Threshold: {result['threshold_used']:.2f}")
            
            if args.verbose:
                print(f"\n  Top {min(args.show_top_k, len(result['top_predictions']))} Predictions:")
                for i, pred in enumerate(result['top_predictions'][:args.show_top_k], 1):
                    print(f"    {i}. {pred['class_name']}: {pred['probability']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error predicting {image_path}: {e}")
        return None


def predict_batch_images(classifier, image_paths, args):
    """Predict multiple images."""
    print(f"🔍 Predicting {len(image_paths)} images...")
    
    try:
        results = classifier.predict_batch(
            image_paths, 
            batch_size=args.batch_size,
            show_progress=not args.quiet
        )
        
        if not args.quiet:
            print(f"\n📊 Batch Prediction Results:")
            print("=" * 80)
            
            confident_count = 0
            for result in results:
                if 'error' in result:
                    print(f"❌ {Path(result['file']).name}: Error - {result['error']}")
                    continue
                
                confidence_indicator = "✅" if result['is_confident'] else "❓"
                if result['is_confident']:
                    confident_count += 1
                
                print(f"{confidence_indicator} {Path(result['file']).name:<30} -> "
                      f"{result['class_name']:<15} (conf: {result['confidence']:.3f})")
            
            print("=" * 80)
            print(f"Summary: {confident_count}/{len(results)} confident predictions")
            
            if args.verbose:
                summary = classifier.predictor.get_prediction_summary(results)
                print(f"Average confidence: {summary['average_confidence']:.3f}")
                print(f"Most common class: {summary['most_common_class'][0]} ({summary['most_common_class'][1]} images)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in batch prediction: {e}")
        return None


def predict_folder(classifier, folder_path, args):
    """Predict all images in folder."""
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder_path}")
        return None
    
    print(f"🔍 Predicting images in folder: {folder_path}")
    print(f"  Recursive: {args.recursive}")
    print(f"  Batch size: {args.batch_size}")
    
    try:
        results = classifier.predictor.predict_folder(
            folder_path,
            recursive=args.recursive,
            batch_size=args.batch_size,
            show_progress=not args.quiet
        )
        
        if not results:
            print("❌ No images found in folder")
            return None
        
        if not args.quiet:
            # Summary statistics
            summary = classifier.predictor.get_prediction_summary(results)
            
            print(f"\n📊 Folder Prediction Summary:")
            print("=" * 60)
            print(f"Total images: {summary['total_predictions']}")
            print(f"Confident predictions: {summary['confident_count']} ({summary['confident_percentage']:.1f}%)")
            print(f"Average confidence: {summary['average_confidence']:.3f}")
            print(f"Threshold used: {summary['threshold_used']:.2f}")
            
            print(f"\nClass Distribution:")
            for class_name, count in summary['class_distribution'].items():
                percentage = (count / summary['total_predictions']) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error predicting folder: {e}")
        return None


def save_results(results, output_path, args):
    """Save prediction results to file."""
    if not results:
        print("❌ No results to save")
        return
    
    output_path = Path(output_path)
    
    try:
        if output_path.suffix.lower() == '.csv':
            # Save as CSV using predictor method
            if hasattr(results[0], 'get') and 'file' in results[0]:
                # It's from predictor
                from src.inference.predictor import Predictor
                # Create dummy predictor just for CSV export
                Predictor.export_predictions_csv(None, results, output_path)
            else:
                # Convert single result to list format
                import pandas as pd
                df = pd.DataFrame([results] if isinstance(results, dict) else results)
                df.to_csv(output_path, index=False)
                
        elif output_path.suffix.lower() == '.json':
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
        
        else:
            print(f"❌ Unsupported output format: {output_path.suffix}")
            return
        
        print(f"💾 Results saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def main():
    """Main prediction function."""
    args = parse_args()
    
    print("🎯 Street Food Classifier - Prediction Mode")
    print("=" * 60)
    
    # Load configuration
    config = Config()
    
    # Update config
    if args.data_folder:
        config.DATA_FOLDER = Path(args.data_folder)
    if args.confidence_threshold:
        config.PREDICTION_THRESHOLD = args.confidence_threshold
    
    if not args.quiet:
        print("Configuration:")
        print(f"  Model: {args.model}")
        print(f"  Architecture: {args.architecture}")
        print(f"  Confidence Threshold: {config.PREDICTION_THRESHOLD}")
        print(f"  Data Folder: {config.DATA_FOLDER}")
        print()
    
    try:
        # Initialize classifier and load model
        if not args.quiet:
            print("🔧 Loading model...")
        
        classifier = StreetFoodClassifier(config)
        classifier.load_model(args.model, args.architecture)
        
        if not args.quiet:
            print("✅ Model loaded successfully")
            
            # Show model info
            model_summary = classifier.get_model_summary()
            data_summary = classifier.get_data_summary()
            
            print(f"  Model: {model_summary['model_name']}")
            print(f"  Parameters: {model_summary['total_params']:,}")
            print(f"  Classes: {data_summary['num_classes']}")
            print()
        
        # Update confidence threshold if specified
        if args.confidence_threshold:
            classifier.predictor.update_threshold(args.confidence_threshold)
        
        results = None
        
        # Handle different input modes
        if args.interactive:
            print("🎮 Starting interactive mode...")
            classifier.interactive_prediction()
            
        elif args.image:
            results = predict_single_image(classifier, args.image, args)
            
        elif args.batch:
            results = predict_batch_images(classifier, args.batch, args)
            
        elif args.folder:
            results = predict_folder(classifier, args.folder, args)
        
        # Save results if requested
        if args.output and results:
            save_results(results, args.output, args)
        
        if not args.quiet and not args.interactive:
            print("\n✅ Prediction completed!")
    
    except KeyboardInterrupt:
        print("\n❌ Prediction interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Prediction failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if 'classifier' in locals():
            classifier.cleanup()


if __name__ == "__main__":
    main()