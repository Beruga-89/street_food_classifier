# PHASE 3: BESTE LÖSUNG - Professional Visualizer Class
# Ersetze src/visualization/visualizer.py komplett mit diesem Code:

"""
Professional Visualization System for Street Food Classifier.

This module provides production-ready visualization capabilities
optimized for ML research and academic presentations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report

from ..utils import get_logger, ensure_dir
from .plotting_utils import (
    set_plot_style, save_figure, create_subplot_grid,
    create_confusion_matrix_plot, plot_training_curves,
    add_timestamp_watermark
)


class ProfessionalVisualizer:
    """
    Production-ready visualization system for ML research.
    
    Features:
    - Clean, publication-quality plots
    - Configurable output formats
    - Error-resistant implementation
    - Academic presentation ready
    """
    
    def __init__(self, config):
        """Initialize the professional visualizer."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Output directories
        self.plot_folder = Path(getattr(config, 'PLOT_FOLDER', 'outputs/plots'))
        ensure_dir(self.plot_folder)
        
        # Set consistent plot style
        set_plot_style()
        
        self.logger.info(f"Professional Visualizer initialized: {self.plot_folder}")
    
    def plot_training_history(self, history: Dict, save: bool = True, 
                            show: bool = True, save_name: Optional[str] = None) -> Optional[str]:
        """
        Create publication-quality training history plots.
        
        Args:
            history: Training history dictionary with train/val metrics
            save: Whether to save the plot
            show: Whether to display the plot
            save_name: Custom filename for saving
            
        Returns:
            Path to saved file if save=True, None otherwise
        """
        self.logger.info("Creating training history visualization...")
        
        # Validate data
        if not self._validate_history(history):
            self.logger.warning("Invalid history data - skipping training history plot")
            return None
        
        # Determine available metrics
        available_metrics = []
        for metric in ['loss', 'accuracy', 'f1']:
            if metric in history.get('train', {}):
                available_metrics.append(metric)
        
        if not available_metrics:
            self.logger.warning("No valid metrics found in history")
            return None
        
        # Create plots
        fig = plot_training_curves(history, metrics=available_metrics)
        
        # Add timestamp watermark
        for ax in fig.axes:
            add_timestamp_watermark(ax)
        
        # Save and show
        saved_path = None
        if save:
            if save_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"training_history_{timestamp}.png"
            
            saved_path = self.plot_folder / save_name
            save_figure(fig, saved_path)
            
            # Also save data as JSON for reproducibility
            history_json = self.plot_folder / f"{Path(save_name).stem}_data.json"
            from ..utils import save_json
            save_json(history, history_json)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return str(saved_path) if saved_path else None
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: List[str], title: str = "Confusion Matrix",
                            save: bool = True, save_name: Optional[str] = None, 
                            show: bool = True) -> Optional[str]:
        """
        Create professional confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            title: Plot title
            save: Whether to save the plot
            save_name: Custom filename
            show: Whether to display the plot
            
        Returns:
            Path to saved file if save=True, None otherwise
        """
        self.logger.info(f"Creating confusion matrix: {title}")
        
        # Validate inputs
        if len(y_true) == 0 or len(y_pred) == 0:
            self.logger.error("Empty prediction arrays provided")
            return None
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create side-by-side absolute and normalized plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Absolute values
        self._plot_single_confusion_matrix(
            cm, class_names, ax1, 
            title=f'{title} - Absolute Values',
            normalize=False
        )
        
        # Normalized values
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        self._plot_single_confusion_matrix(
            cm_normalized, class_names, ax2,
            title=f'{title} - Normalized',
            normalize=True
        )
        
        plt.tight_layout()
        
        # Save and show
        saved_path = None
        if save:
            if save_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"confusion_matrix_{timestamp}.png"
            
            saved_path = self.plot_folder / save_name
            save_figure(fig, saved_path)
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        # Print detailed classification report
        self._print_classification_report(y_true, y_pred, class_names)
        
        return str(saved_path) if saved_path else None
    
    def create_comprehensive_dashboard(self, training_history: Dict, 
                                     evaluation_results: Dict,
                                     class_names: List[str],
                                     save: bool = True, show: bool = True) -> Optional[str]:
        """
        Create a comprehensive analysis dashboard.
        
        Args:
            training_history: Training history data
            evaluation_results: Evaluation results with predictions/labels
            class_names: List of class names
            save: Whether to save the dashboard
            show: Whether to display the dashboard
            
        Returns:
            Path to saved dashboard if save=True, None otherwise
        """
        self.logger.info("Creating comprehensive analysis dashboard...")
        
        try:
            # Create 2x3 dashboard layout
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Comprehensive Model Analysis Dashboard', 
                        fontsize=18, fontweight='bold')
            
            # Plot 1: Training curves
            self._create_training_curves_subplot(training_history, axes[0, 0])
            
            # Plot 2: Per-class performance
            self._create_per_class_performance_subplot(
                evaluation_results, class_names, axes[0, 1]
            )
            
            # Plot 3: Performance metrics summary
            self._create_metrics_summary_subplot(
                training_history, evaluation_results, axes[0, 2]
            )
            
            # Plot 4: Confusion matrix (compact)
            self._create_compact_confusion_matrix_subplot(
                evaluation_results, class_names, axes[1, 0]
            )
            
            # Plot 5: Class distribution
            self._create_class_distribution_subplot(
                evaluation_results, class_names, axes[1, 1]
            )
            
            # Plot 6: Summary statistics
            self._create_summary_statistics_subplot(
                training_history, evaluation_results, class_names, axes[1, 2]
            )
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            
            # Save and show
            saved_path = None
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"comprehensive_dashboard_{timestamp}.png"
                saved_path = self.plot_folder / save_name
                save_figure(fig, saved_path)
            
            if show:
                plt.show()
            else:
                plt.close(fig)
            
            return str(saved_path) if saved_path else None
            
        except Exception as e:
            self.logger.error(f"Dashboard creation failed: {e}")
            return None
    
    # === PRIVATE HELPER METHODS ===
    
    def _validate_history(self, history: Dict) -> bool:
        """Validate training history data structure."""
        if not isinstance(history, dict):
            return False
        
        required_keys = ['train', 'val']
        if not all(key in history for key in required_keys):
            return False
        
        # Check if at least one metric exists
        for split in ['train', 'val']:
            if not isinstance(history[split], dict):
                return False
            if not any(metric in history[split] for metric in ['loss', 'accuracy', 'f1']):
                return False
        
        return True
    
    def _plot_single_confusion_matrix(self, cm, class_names, ax, title, normalize=False):
        """Plot a single confusion matrix on given axes."""
        fmt = '.2f' if normalize else 'd'
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=title,
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Rotate labels if many classes
        if len(class_names) > 8:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=8)
    
    def _print_classification_report(self, y_true, y_pred, class_names):
        """Print detailed classification report."""
        from sklearn.metrics import accuracy_score, f1_score
        
        print(f"\n{'='*80}")
        print("DETAILED CLASSIFICATION REPORT")
        print(f"{'='*80}")
        
        report = classification_report(
            y_true, y_pred, 
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        print(report)
        
        # Additional metrics
        overall_accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\nOVERALL METRICS:")
        print(f"  Overall Accuracy: {overall_accuracy:.4f}")
        print(f"  Macro F1 Score:   {macro_f1:.4f}")
        print(f"  Weighted F1 Score: {weighted_f1:.4f}")
    
    # === DASHBOARD SUBPLOT METHODS ===
    
    def _create_training_curves_subplot(self, history, ax):
        """Create training curves subplot."""
        if not self._validate_history(history):
            ax.text(0.5, 0.5, 'No Training History\nAvailable', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Training Progress')
            return
        
        epochs = range(1, len(history['train']['loss']) + 1)
        
        # Plot loss
        ax.plot(epochs, history['train']['loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, history['val']['loss'], 'r-', label='Val Loss', linewidth=2)
        
        ax.set_title('Training Progress')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_per_class_performance_subplot(self, results, class_names, ax):
        """Create per-class performance subplot."""
        if 'labels' not in results or 'predictions' not in results:
            ax.text(0.5, 0.5, 'No Prediction Data\nAvailable', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Per-Class Performance')
            return
        
        cm = confusion_matrix(results['labels'], results['predictions'])
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        # Show top 10 classes
        if len(class_names) > 10:
            top_indices = np.argsort(per_class_acc)[-10:]
            display_names = [class_names[i] for i in top_indices]
            display_acc = per_class_acc[top_indices]
        else:
            display_names = class_names
            display_acc = per_class_acc
        
        bars = ax.barh(range(len(display_names)), display_acc, alpha=0.7, color='skyblue')
        ax.set_yticks(range(len(display_names)))
        ax.set_yticklabels(display_names, fontsize=9)
        ax.set_xlabel('Accuracy')
        ax.set_title(f'Per-Class Accuracy (Top {len(display_names)})')
        ax.grid(True, axis='x', alpha=0.3)
    
    def _create_metrics_summary_subplot(self, history, results, ax):
        """Create metrics summary subplot."""
        from sklearn.metrics import accuracy_score, f1_score
        
        if 'labels' in results and 'predictions' in results:
            accuracy = accuracy_score(results['labels'], results['predictions'])
            f1 = f1_score(results['labels'], results['predictions'], 
                         average='weighted', zero_division=0)
        else:
            accuracy = f1 = 0.0
        
        final_train_loss = history.get('train', {}).get('loss', [0])[-1] if history else 0
        final_val_loss = history.get('val', {}).get('loss', [0])[-1] if history else 0
        
        summary_text = f"""PERFORMANCE SUMMARY

Validation Metrics:
• Accuracy: {accuracy:.4f}
• F1-Score: {f1:.4f}

Training Status:
• Final Train Loss: {final_train_loss:.4f}
• Final Val Loss: {final_val_loss:.4f}
• Samples: {len(results.get('labels', []))}

Model Status: Ready"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax.set_title('Model Summary')
        ax.axis('off')
    
    def _create_compact_confusion_matrix_subplot(self, results, class_names, ax):
        """Create compact confusion matrix subplot."""
        if 'labels' not in results or 'predictions' not in results:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center')
            ax.set_title('Confusion Matrix')
            return
        
        cm = confusion_matrix(results['labels'], results['predictions'])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        ax.set_title('Confusion Matrix (Normalized)')
        
        # Simplified labels for compact view
        tick_marks = np.arange(len(class_names))
        if len(class_names) > 10:
            # Show every nth label
            step = len(class_names) // 8
            ax.set_xticks(tick_marks[::step])
            ax.set_yticks(tick_marks[::step])
            ax.set_xticklabels([class_names[i] for i in tick_marks[::step]], 
                             rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels([class_names[i] for i in tick_marks[::step]], fontsize=8)
        else:
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _create_class_distribution_subplot(self, results, class_names, ax):
        """Create class distribution subplot."""
        if 'labels' not in results:
            ax.text(0.5, 0.5, 'No Label Data\nAvailable', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Class Distribution')
            return
        
        unique, counts = np.unique(results['labels'], return_counts=True)
        class_counts = [counts[np.where(unique == i)[0][0]] if i in unique else 0 
                       for i in range(len(class_names))]
        
        ax.bar(range(len(class_names)), class_counts, alpha=0.7, color='lightcoral')
        ax.set_xlabel('Class Index')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Test Set Class Distribution')
        ax.grid(True, axis='y', alpha=0.3)
        
        avg_count = np.mean(class_counts)
        ax.axhline(avg_count, color='blue', linestyle='--', linewidth=2, 
                  label=f'Average: {avg_count:.1f}')
        ax.legend()
    
    def _create_summary_statistics_subplot(self, history, results, class_names, ax):
        """Create summary statistics subplot."""
        from sklearn.metrics import classification_report
        
        if 'labels' in results and 'predictions' in results:
            report = classification_report(
                results['labels'], results['predictions'], 
                target_names=class_names, output_dict=True
            )
            
            metrics = {
                'Accuracy': report['accuracy'],
                'Macro Precision': report['macro avg']['precision'],
                'Macro Recall': report['macro avg']['recall'],
                'Macro F1': report['macro avg']['f1-score'],
                'Weighted F1': report['weighted avg']['f1-score']
            }
        else:
            metrics = {key: 0.0 for key in ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 'Weighted F1']}
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(range(len(metric_names)), metric_values, 
                     color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
        
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Score')
        ax.set_title('Performance Metrics')
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)


# Create alias for backward compatibility
Visualizer = ProfessionalVisualizer