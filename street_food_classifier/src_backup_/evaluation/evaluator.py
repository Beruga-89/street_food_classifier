"""
Standalone model evaluation utilities.

This module provides the StandaloneModelEvaluator class for evaluating
saved models independently from the training process.
"""

from typing import Dict, Optional

from ..utils import get_device, get_logger
from ..data import DataManager
from ..models import ModelManager
from ..training import Trainer
from .workflow import EvaluationWorkflow


class StandaloneModelEvaluator:
    """
    Lädt gespeicherte Models und evaluiert sie unabhängig vom Training.
    
    Diese Klasse ermöglicht:
    - Evaluation von gespeicherten Models ohne Training-Setup
    - Vollständige Rekonstruktion des Evaluation-Pipelines
    - Vergleich verschiedener gespeicherter Models
    - Integration mit Visualization und Reporting
    
    Example:
        >>> evaluator = StandaloneModelEvaluator(config)
        >>> results = evaluator.evaluate_saved_model('best_f1_model.pth', 'ResNet18')
        >>> print(f"Validation Accuracy: {results['validation']['accuracy']:.4f}")
    """
    
    def __init__(self, config):
        """
        Initialisiert den StandaloneModelEvaluator.
        
        Args:
            config: Konfigurationsobjekt
        """
        self.config = config
        self.device = get_device()
        self.logger = get_logger(__name__)
        
        # Data Manager für DataLoader
        self.data_manager = DataManager(config)
        (self.train_loader, self.val_loader, 
         self.num_classes, self.class_names) = self.data_manager.create_dataloaders()
        
        # Model Manager
        self.model_manager = ModelManager(config, self.num_classes, self.device)
        
        # Evaluation Workflow
        self.eval_workflow = EvaluationWorkflow(config)
        
        self.logger.info(f"StandaloneModelEvaluator initialized")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Classes: {self.num_classes}")
        self.logger.info(f"  Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"  Val samples: {len(self.val_loader.dataset)}")
        
    def evaluate_saved_model(self, 
                           model_path: str,
                           model_name: str = "loaded_model",
                           architecture: str = "resnet18",
                           evaluate_train: bool = True,
                           evaluate_val: bool = True,
                           visualize: bool = True,
                           save_results: bool = True) -> Dict:
        """
        Lädt ein gespeichertes Model und evaluiert es vollständig.
        
        Args:
            model_path: Pfad zum Model (.pth Datei)
            model_name: Name für die Speicherung der Ergebnisse
            architecture: Model-Architektur für Rekonstruktion
            evaluate_train: Ob Training Set evaluiert werden soll
            evaluate_val: Ob Validation Set evaluiert werden soll
            visualize: Ob Visualisierungen erstellt werden sollen
            save_results: Ob Ergebnisse gespeichert werden sollen
            
        Returns:
            Dictionary mit Evaluationsergebnissen
            
        Example:
            >>> results = evaluator.evaluate_saved_model(
            ...     'models/best_f1_model.pth',
            ...     'ResNet18_Final',
            ...     visualize=True
            ... )
        """
        self.logger.info(f"[EVALUATING] Loading model from {model_path}")
        
        # Model erstellen und laden
        model = self.model_manager.create_model(architecture)
        model = self.model_manager.load_model(model, model_path)
        
        # Model-Info ausgeben
        model_info = self.model_manager.get_model_summary(model)
        self.logger.info(f"Model loaded: {model_info['model_name']}")
        self.logger.info(f"  Parameters: {model_info['total_params']:,}")
        
        # Trainer für Evaluation erstellen (Dummy optimizer)
        optimizer = self.model_manager.create_optimizer(model, 'adam')
        trainer = Trainer(self.config, model, optimizer, self.device)
        
        results = {'model_info': model_info, 'class_names': self.class_names}
        
        # Validation Set evaluieren
        if evaluate_val:
            self.logger.info("[EVALUATING] Running evaluation on validation set...")
            val_results = trainer.evaluate(self.val_loader, return_predictions=True)
            results['validation'] = val_results
            
            self.logger.info(f"Validation Results:")
            self.logger.info(f"  Accuracy: {val_results['accuracy']:.4f}")
            self.logger.info(f"  F1-Score: {val_results['f1']:.4f}")
            self.logger.info(f"  Loss: {val_results['loss']:.4f}")
        
        # Training Set evaluieren (optional)
        if evaluate_train:
            self.logger.info("[EVALUATING] Running evaluation on training set...")
            train_results = trainer.evaluate(self.train_loader, return_predictions=True)
            results['training'] = train_results
            
            self.logger.info(f"Training Results:")
            self.logger.info(f"  Accuracy: {train_results['accuracy']:.4f}")
            self.logger.info(f"  F1-Score: {train_results['f1']:.4f}")
            self.logger.info(f"  Loss: {train_results['loss']:.4f}")
        
        # Ergebnisse speichern
        if save_results:
            self.logger.info("[SAVING] Saving evaluation results...")
            saved_files = {}
            
            if evaluate_val:
                val_file = self.eval_workflow.metrics_manager.save_evaluation_results(
                    results['validation'], self.class_names, model_name, "validation"
                )
                saved_files['validation'] = val_file
            
            if evaluate_train:
                train_file = self.eval_workflow.metrics_manager.save_evaluation_results(
                    results['training'], self.class_names, model_name, "training"
                )
                saved_files['training'] = train_file
            
            results['saved_files'] = saved_files
        
        # Visualisierungen erstellen
        if visualize:
            self.logger.info("[VISUALIZING] Creating visualizations...")
            
            if evaluate_val:
                # Confusion Matrix für Validation
                self.eval_workflow.visualizer.plot_confusion_matrix(
                    results['validation']['labels'], 
                    results['validation']['predictions'], 
                    self.class_names,
                    title=f"{model_name} - Validation Results",
                    save=True,
                    save_name=f"{model_name}_validation_confusion_matrix.png"
                )
            
            if evaluate_train:
                # Confusion Matrix für Training
                self.eval_workflow.visualizer.plot_confusion_matrix(
                    results['training']['labels'], 
                    results['training']['predictions'], 
                    self.class_names,
                    title=f"{model_name} - Training Results", 
                    save=True,
                    save_name=f"{model_name}_training_confusion_matrix.png"
                )
        
        self.logger.info("[COMPLETED] Model evaluation finished!")
        return results
    
    def compare_multiple_models(self, 
                               model_configs: list,
                               save_comparison: bool = True,
                               visualize_comparison: bool = True) -> Dict:
        """
        Vergleicht mehrere gespeicherte Models.
        
        Args:
            model_configs: Liste von Dictionaries mit Model-Konfigurationen
                          [{'path': 'model1.pth', 'name': 'Model1', 'arch': 'resnet18'}, ...]
            save_comparison: Ob Vergleichsergebnisse gespeichert werden sollen
            visualize_comparison: Ob Vergleichsvisualisierung erstellt werden soll
            
        Returns:
            Dictionary mit Vergleichsergebnissen
            
        Example:
            >>> configs = [
            ...     {'path': 'best_f1_model.pth', 'name': 'Best_F1', 'arch': 'resnet18'},
            ...     {'path': 'best_acc_model.pth', 'name': 'Best_Acc', 'arch': 'resnet18'}
            ... ]
            >>> comparison = evaluator.compare_multiple_models(configs)
        """
        self.logger.info(f"[COMPARING] Starting comparison of {len(model_configs)} models")
        
        all_results = []
        comparison_data = {
            'models': [],
            'validation_results': [],
            'training_results': []
        }
        
        # Jedes Model evaluieren
        for i, config in enumerate(model_configs, 1):
            model_path = config['path']
            model_name = config['name']
            architecture = config.get('arch', 'resnet18')
            
            self.logger.info(f"[{i}/{len(model_configs)}] Evaluating {model_name}")
            
            try:
                results = self.evaluate_saved_model(
                    model_path=model_path,
                    model_name=model_name,
                    architecture=architecture,
                    evaluate_train=True,
                    evaluate_val=True,
                    visualize=False,  # Einzelne Visualisierungen später
                    save_results=save_comparison
                )
                
                all_results.append(results)
                comparison_data['models'].append(model_name)
                
                if 'validation' in results:
                    comparison_data['validation_results'].append(results['validation'])
                if 'training' in results:
                    comparison_data['training_results'].append(results['training'])
                    
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        # Vergleichsstatistiken berechnen
        comparison_stats = self._calculate_comparison_stats(comparison_data)
        
        # Visualisierung erstellen
        if visualize_comparison and len(all_results) > 1:
            self._create_comparison_visualizations(comparison_data, comparison_stats)
        
        # Zusammenfassung ausgeben
        self._print_comparison_summary(comparison_stats)
        
        return {
            'individual_results': all_results,
            'comparison_data': comparison_data,
            'comparison_stats': comparison_stats
        }
    
    def batch_evaluate_folder(self, models_folder: str, 
                             architecture: str = "resnet18",
                             pattern: str = "*.pth") -> Dict:
        """
        Evaluiert alle Models in einem Ordner.
        
        Args:
            models_folder: Pfad zum Models-Ordner
            architecture: Standard-Architektur
            pattern: Datei-Pattern für Models
            
        Returns:
            Dictionary mit allen Evaluationsergebnissen
        """
        from pathlib import Path
        
        models_path = Path(models_folder)
        model_files = list(models_path.glob(pattern))
        
        if not model_files:
            raise ValueError(f"No model files found in {models_folder} with pattern {pattern}")
        
        self.logger.info(f"Found {len(model_files)} model files for batch evaluation")
        
        # Model-Konfigurationen erstellen
        model_configs = []
        for model_file in model_files:
            model_name = model_file.stem  # Dateiname ohne Extension
            config = {
                'path': str(model_file),
                'name': model_name,
                'arch': architecture
            }
            model_configs.append(config)
        
        # Batch-Evaluation durchführen
        return self.compare_multiple_models(
            model_configs,
            save_comparison=True,
            visualize_comparison=True
        )
    
    def _calculate_comparison_stats(self, comparison_data: Dict) -> Dict:
        """Berechnet Vergleichsstatistiken zwischen Models."""
        stats = {
            'validation': {'accuracy': [], 'f1': [], 'loss': []},
            'training': {'accuracy': [], 'f1': [], 'loss': []}
        }
        
        # Validation Statistiken
        for result in comparison_data['validation_results']:
            stats['validation']['accuracy'].append(result['accuracy'])
            stats['validation']['f1'].append(result['f1'])
            stats['validation']['loss'].append(result['loss'])
        
        # Training Statistiken
        for result in comparison_data['training_results']:
            stats['training']['accuracy'].append(result['accuracy'])
            stats['training']['f1'].append(result['f1'])
            stats['training']['loss'].append(result['loss'])
        
        # Best/Worst Models finden
        models = comparison_data['models']
        best_models = {}
        worst_models = {}
        
        if stats['validation']['accuracy']:
            best_acc_idx = max(range(len(stats['validation']['accuracy'])), 
                              key=lambda i: stats['validation']['accuracy'][i])
            worst_acc_idx = min(range(len(stats['validation']['accuracy'])), 
                               key=lambda i: stats['validation']['accuracy'][i])
            
            best_models['accuracy'] = {
                'name': models[best_acc_idx],
                'value': stats['validation']['accuracy'][best_acc_idx]
            }
            worst_models['accuracy'] = {
                'name': models[worst_acc_idx],
                'value': stats['validation']['accuracy'][worst_acc_idx]
            }
        
        return {
            'stats': stats,
            'models': models,
            'best_models': best_models,
            'worst_models': worst_models
        }
    
    def _create_comparison_visualizations(self, comparison_data: Dict, 
                                        comparison_stats: Dict) -> None:
        """Erstellt Vergleichsvisualisierungen."""
        # Model Performance Comparison
        if len(comparison_data['validation_results']) > 1:
            # Daten für Comparison vorbereiten
            evaluation_results = []
            for model_name, val_result in zip(comparison_data['models'], 
                                            comparison_data['validation_results']):
                eval_data = {
                    'model_name': model_name,
                    'dataset_type': 'validation',
                    'metrics': {
                        'accuracy': val_result['accuracy'],
                        'f1': val_result['f1'],
                        'loss': val_result['loss']
                    }
                }
                evaluation_results.append(eval_data)
            
            # Vergleichsplot erstellen
            self.eval_workflow.visualizer.create_performance_comparison(
                evaluation_results,
                save=True,
                show=True,
                save_name="model_comparison.png"
            )
    
    def _print_comparison_summary(self, comparison_stats: Dict) -> None:
        """Gibt Vergleichszusammenfassung aus."""
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        if comparison_stats['best_models']:
            print(f"\nBest Performing Models (Validation):")
            for metric, info in comparison_stats['best_models'].items():
                print(f"  Best {metric.capitalize()}: {info['name']} ({info['value']:.4f})")
        
        if comparison_stats['stats']['validation']['accuracy']:
            val_stats = comparison_stats['stats']['validation']
            print(f"\nValidation Statistics:")
            print(f"  Accuracy: {min(val_stats['accuracy']):.4f} - {max(val_stats['accuracy']):.4f}")
            print(f"  F1-Score: {min(val_stats['f1']):.4f} - {max(val_stats['f1']):.4f}")
            print(f"  Loss: {min(val_stats['loss']):.4f} - {max(val_stats['loss']):.4f}")
        
        print(f"{'='*80}")