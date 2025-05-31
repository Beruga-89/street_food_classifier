"""
Evaluation workflow management.

This module provides the EvaluationWorkflow class for streamlined
evaluation processes and visualization workflows.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..utils import get_logger
from .metrics_manager import MetricsManager
from ..visualization import Visualizer


class EvaluationWorkflow:
    """
    Vereinfacht den Workflow für Evaluation und Visualisierung.
    
    Diese Klasse bietet:
    - Streamlined Evaluation Workflows
    - Integration zwischen Evaluation und Visualization
    - Automatisierte Report-Erstellung
    - Batch-Processing von Evaluationsergebnissen
    
    Example:
        >>> workflow = EvaluationWorkflow(config)
        >>> # Nach dem Training
        >>> workflow.save_training_results(classifier, "ResNet18_v1")
        >>> # Später für Analyse
        >>> workflow.load_and_visualize("ResNet18_v1_validation_results.json")
    """
    
    def __init__(self, config):
        """
        Initialisiert den EvaluationWorkflow.
        
        Args:
            config: Konfigurationsobjekt
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.metrics_manager = MetricsManager(config)
        self.visualizer = Visualizer(config)
        
    def save_training_results(self, 
                            classifier,
                            model_name: str = "resnet18",
                            save_train: bool = True,
                            save_val: bool = True,
                            create_visualizations: bool = True) -> Tuple[Optional[str], Optional[str]]:
        """
        Speichert Training- und Validierungsergebnisse nach dem Training.
        
        Args:
            classifier: Trainierte Classifier-Instanz (mit trainer, class_names, etc.)
            model_name: Name des Models für Dateibenennung
            save_train: Ob Training-Ergebnisse gespeichert werden sollen
            save_val: Ob Validation-Ergebnisse gespeichert werden sollen
            create_visualizations: Ob sofort Visualisierungen erstellt werden sollen
            
        Returns:
            Tuple mit (train_file_path, val_file_path)
            
        Example:
            >>> train_path, val_path = workflow.save_training_results(
            ...     classifier, "ResNet18_experiment_1"
            ... )
        """
        train_path = None
        val_path = None
        
        if save_train:
            self.logger.info("Evaluating and saving training results...")
            train_results = classifier.trainer.evaluate(classifier.train_loader)
            train_path = self.metrics_manager.save_evaluation_results(
                train_results, classifier.class_names, model_name, "training"
            )
            
        if save_val:
            self.logger.info("Evaluating and saving validation results...")
            val_results = classifier.trainer.evaluate(classifier.val_loader)
            val_path = self.metrics_manager.save_evaluation_results(
                val_results, classifier.class_names, model_name, "validation"
            )
            
            # Sofortige Visualisierung der Validation-Ergebnisse
            if create_visualizations:
                self.visualizer.plot_confusion_matrix(
                    val_results['labels'],
                    val_results['predictions'],
                    classifier.class_names,
                    title=f"{model_name} - Validation Results",
                    save=True,
                    save_name=f"{model_name}_validation_confusion_matrix.png"
                )
        
        return train_path, val_path
    
    def load_and_visualize(self, 
                          result_file: Union[str, Path],
                          create_confusion_matrix: bool = True,
                          create_classification_report: bool = True,
                          save_plots: bool = True,
                          show_plots: bool = True) -> Dict[str, Optional[str]]:
        """
        Lädt Evaluationsergebnisse und erstellt Visualisierungen.
        
        Args:
            result_file: Pfad zur Ergebnisdatei
            create_confusion_matrix: Ob Confusion Matrix erstellt werden soll
            create_classification_report: Ob Classification Report ausgegeben werden soll
            save_plots: Ob Plots gespeichert werden sollen
            show_plots: Ob Plots angezeigt werden sollen
            
        Returns:
            Dictionary mit Pfaden zu erstellten Visualisierungen
            
        Example:
            >>> plots = workflow.load_and_visualize("model_validation_results.json")
            >>> print(f"Confusion matrix saved to: {plots['confusion_matrix']}")
        """
        # Daten laden
        evaluation_data = self.metrics_manager.load_evaluation_results(result_file)
        
        saved_plots = {}
        
        if create_confusion_matrix:
            cm_path = self.visualizer.create_confusion_matrix_from_data(
                evaluation_data, 
                save=save_plots, 
                show=show_plots
            )
            saved_plots['confusion_matrix'] = cm_path
        
        if create_classification_report:
            # Classification Report in Console ausgeben
            self.visualizer.print_classification_report(
                evaluation_data['labels'],
                evaluation_data['predictions'],
                evaluation_data['metadata']['class_names']
            )
        
        return saved_plots
    
    def compare_models(self, 
                      result_files: List[Union[str, Path]],
                      save_comparison: bool = True,
                      show_comparison: bool = True,
                      comparison_name: Optional[str] = None) -> Optional[str]:
        """
        Vergleicht mehrere Models basierend auf gespeicherten Ergebnissen.
        
        Args:
            result_files: Liste von Pfaden zu Ergebnisdateien
            save_comparison: Ob Vergleichsplot gespeichert werden soll
            show_comparison: Ob Vergleichsplot angezeigt werden soll
            comparison_name: Name für den Vergleichsplot
            
        Returns:
            Pfad zum gespeicherten Vergleichsplot (falls save_comparison=True)
            
        Example:
            >>> files = ["model1_val_results.json", "model2_val_results.json"]
            >>> plot_path = workflow.compare_models(files, comparison_name="model_comparison")
        """
        if len(result_files) < 2:
            self.logger.warning("Need at least 2 result files for comparison")
            return None
        
        evaluation_results = []
        
        for file_path in result_files:
            try:
                data = self.metrics_manager.load_evaluation_results(file_path)
                evaluation_results.append(data)
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if len(evaluation_results) < 2:
            self.logger.error("Could not load enough valid result files for comparison")
            return None
        
        # Vergleichsplot erstellen
        save_name = None
        if save_comparison and comparison_name:
            save_name = f"{comparison_name}.png"
        
        return self.visualizer.create_performance_comparison(
            evaluation_results, 
            save=save_comparison, 
            show=show_comparison,
            save_name=save_name
        )
    
    def create_evaluation_report(self, 
                               result_files: List[Union[str, Path]],
                               report_name: str = "evaluation_report",
                               include_individual_plots: bool = True,
                               include_comparison: bool = True) -> str:
        """
        Erstellt einen umfassenden Evaluation-Report.
        
        Args:
            result_files: Liste von Ergebnisdateien
            report_name: Name des Reports
            include_individual_plots: Ob individuelle Confusion Matrices eingeschlossen werden sollen
            include_comparison: Ob Model-Vergleich eingeschlossen werden soll
            
        Returns:
            Pfad zum Report-Verzeichnis
            
        Example:
            >>> files = ["model1_results.json", "model2_results.json"]
            >>> report_path = workflow.create_evaluation_report(files, "final_comparison")
        """
        from datetime import datetime
        import shutil
        
        # Report-Verzeichnis erstellen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = Path(self.config.REPORT_FOLDER) / f"{report_name}_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Creating evaluation report in: {report_dir}")
        
        # Report-Metadaten sammeln
        report_data = {
            'report_name': report_name,
            'timestamp': timestamp,
            'models': [],
            'plots': [],
            'summary': {}
        }
        
        # Individuelle Model-Visualisierungen
        if include_individual_plots:
            individual_dir = report_dir / "individual_models"
            individual_dir.mkdir(exist_ok=True)
            
            for i, result_file in enumerate(result_files):
                try:
                    # Daten laden
                    evaluation_data = self.metrics_manager.load_evaluation_results(result_file)
                    model_name = evaluation_data['metadata']['model_name']
                    
                    # Confusion Matrix erstellen
                    plot_name = f"{model_name}_confusion_matrix.png"
                    cm_path = self.visualizer.create_confusion_matrix_from_data(
                        evaluation_data,
                        save=True,
                        show=False,
                        save_name=plot_name
                    )
                    
                    # Plot in Report-Verzeichnis kopieren
                    if cm_path:
                        target_path = individual_dir / plot_name
                        shutil.copy2(cm_path, target_path)
                        report_data['plots'].append(str(target_path))
                    
                    # Model-Info sammeln
                    model_info = {
                        'name': model_name,
                        'dataset_type': evaluation_data['metadata']['dataset_type'],
                        'accuracy': evaluation_data['metrics']['accuracy'],
                        'f1': evaluation_data['metrics']['f1'],
                        'loss': evaluation_data['metrics']['loss']
                    }
                    report_data['models'].append(model_info)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {result_file}: {e}")
        
        # Model-Vergleich
        if include_comparison and len(result_files) > 1:
            comparison_path = self.compare_models(
                result_files,
                save_comparison=True,
                show_comparison=False,
                comparison_name="model_comparison"
            )
            
            if comparison_path:
                target_comparison = report_dir / "model_comparison.png"
                shutil.copy2(comparison_path, target_comparison)
                report_data['plots'].append(str(target_comparison))
        
        # Summary Statistics berechnen
        if report_data['models']:
            accuracies = [m['accuracy'] for m in report_data['models']]
            f1_scores = [m['f1'] for m in report_data['models']]
            
            report_data['summary'] = {
                'num_models': len(report_data['models']),
                'best_accuracy': max(accuracies),
                'worst_accuracy': min(accuracies),
                'avg_accuracy': sum(accuracies) / len(accuracies),
                'best_f1': max(f1_scores),
                'worst_f1': min(f1_scores),
                'avg_f1': sum(f1_scores) / len(f1_scores),
                'best_model': max(report_data['models'], key=lambda x: x['f1'])['name']
            }
        
        # Report-Metadata speichern
        from ..utils import save_json
        save_json(report_data, report_dir / "report_metadata.json")
        
        # Text-Report erstellen
        self._create_text_report(report_data, report_dir / "report.txt")
        
        self.logger.info(f"Evaluation report created: {report_dir}")
        return str(report_dir)
    
    def _create_text_report(self, report_data: Dict, output_path: Path) -> None:
        """Erstellt einen Text-Report."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Report Name: {report_data['report_name']}\n")
            f.write(f"Generated: {report_data['timestamp']}\n")
            f.write(f"Number of Models: {report_data['summary'].get('num_models', 0)}\n\n")
            
            if report_data['summary']:
                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 30 + "\n")
                summary = report_data['summary']
                f.write(f"Best Model: {summary['best_model']}\n")
                f.write(f"Best Accuracy: {summary['best_accuracy']:.4f}\n")
                f.write(f"Best F1-Score: {summary['best_f1']:.4f}\n")
                f.write(f"Average Accuracy: {summary['avg_accuracy']:.4f}\n")
                f.write(f"Average F1-Score: {summary['avg_f1']:.4f}\n\n")
            
            if report_data['models']:
                f.write("INDIVIDUAL MODEL RESULTS\n")
                f.write("-" * 30 + "\n")
                for model in report_data['models']:
                    f.write(f"Model: {model['name']}\n")
                    f.write(f"  Dataset: {model['dataset_type']}\n")
                    f.write(f"  Accuracy: {model['accuracy']:.4f}\n")
                    f.write(f"  F1-Score: {model['f1']:.4f}\n")
                    f.write(f"  Loss: {model['loss']:.4f}\n\n")
    
    def quick_evaluate_latest(self) -> Optional[Dict]:
        """
        Schnelle Evaluation des neuesten gespeicherten Ergebnisses.
        
        Returns:
            Dictionary mit Visualisierungspfaden oder None
            
        Example:
            >>> plots = workflow.quick_evaluate_latest()
            >>> if plots:
            ...     print("Latest model visualized!")
        """
        available = self.metrics_manager.list_saved_results()
        
        if not available:
            self.logger.warning("No saved results found!")
            return None
        
        latest_result = available[0]['file']
        self.logger.info(f"Visualizing latest result: {latest_result}")
        
        return self.load_and_visualize(latest_result)
    
    def auto_compare_all(self, max_models: int = 5) -> Optional[str]:
        """
        Automatischer Vergleich aller verfügbaren Models.
        
        Args:
            max_models: Maximale Anzahl Models für Vergleich
            
        Returns:
            Pfad zum Vergleichsplot oder None
            
        Example:
            >>> comparison_plot = workflow.auto_compare_all(max_models=3)
        """
        available = self.metrics_manager.list_saved_results()
        
        if len(available) < 2:
            self.logger.warning("Need at least 2 saved results for comparison!")
            return None
        
        # Top N Models nehmen
        files = [r['file'] for r in available[:max_models]]
        self.logger.info(f"Comparing top {len(files)} models")
        
        return self.compare_models(files, comparison_name="auto_comparison")
    
    def cleanup_old_visualizations(self, keep_days: int = 30) -> int:
        """
        Löscht alte Visualisierungen.
        
        Args:
            keep_days: Anzahl Tage die behalten werden sollen
            
        Returns:
            Anzahl gelöschter Dateien
        """
        from datetime import datetime, timedelta
        import os
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        deleted_count = 0
        
        plot_folder = Path(getattr(self.config, 'PLOT_FOLDER', 'outputs/plots'))
        
        if not plot_folder.exists():
            return 0
        
        for file_path in plot_folder.glob("*.png"):
            try:
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    file_path.unlink()
                    deleted_count += 1
            except Exception as e:
                self.logger.warning(f"Error deleting {file_path}: {e}")
        
        self.logger.info(f"Cleanup completed: {deleted_count} old plots deleted")
        return deleted_count