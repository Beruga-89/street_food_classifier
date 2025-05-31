"""
Main application class for the Street Food Classifier.

This module contains the StreetFoodClassifier class which serves as the
main entry point and orchestrates all components of the ML pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch

from .utils import setup_logger, seed_everything, get_device
from .data import DataManager
from .models import ModelManager
from .training import Trainer
from .evaluation import EvaluationWorkflow
from .visualization import Visualizer
from .inference import Predictor


class StreetFoodClassifier:
    """
    Hauptapplikation für Street Food Klassifikation.
    
    Diese Klasse orchestriert das gesamte Machine Learning Pipeline:
    - Datenmanagement und -loading
    - Model-Erstellung und -Management
    - Training und Evaluation
    - Visualisierung und Reporting
    - Inferenz und Prediction
    
    Example:
        >>> from config import Config
        >>> config = Config()
        >>> classifier = StreetFoodClassifier(config)
        >>> 
        >>> # Training
        >>> history = classifier.train()
        >>> 
        >>> # Evaluation
        >>> results = classifier.evaluate()
        >>> 
        >>> # Prediction
        >>> prediction = classifier.predict("image.jpg")
    """
    
    def __init__(self, config):
        """
        Initialisiert den StreetFoodClassifier.
        
        Args:
            config: Konfigurationsobjekt mit allen Einstellungen
        """
        self.config = config
        self.logger = setup_logger(__name__, log_dir=config.LOG_FOLDER)
        
        # Reproduzierbarkeit sicherstellen
        seed_everything(config.SEED)
        self.device = get_device(getattr(config, 'DEVICE', None))
        
        # Pipeline-Komponenten
        self.data_manager = None
        self.model_manager = None
        self.model = None
        self.optimizer = None
        self.trainer = None
        self.predictor = None
        
        # Data Loader (werden beim ersten Aufruf erstellt)
        self.train_loader = None
        self.val_loader = None
        self.num_classes = None
        self.class_names = None
        
        # Workflow-Manager
        self.evaluation_workflow = EvaluationWorkflow(config)
        self.visualizer = Visualizer(config)
        
        # Training State
        self.is_trained = False
        self.training_history = None
        
        self.logger.info("StreetFoodClassifier initialized")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Random seed: {config.SEED}")
    
    def setup_data(self) -> Tuple[int, list]:
        """
        Setzt Datenmanagement auf und lädt Datasets.
        
        Returns:
            Tuple mit (num_classes, class_names)
            
        Example:
            >>> num_classes, class_names = classifier.setup_data()
            >>> print(f"Dataset has {num_classes} classes: {class_names}")
        """
        if self.data_manager is None:
            self.logger.info("Setting up data management...")
            
            # Data Manager erstellen
            self.data_manager = DataManager(self.config)
            
            # DataLoaders erstellen
            (self.train_loader, self.val_loader, 
             self.num_classes, self.class_names) = self.data_manager.create_dataloaders()
            
            self.logger.info(f"Data setup completed:")
            self.logger.info(f"  Classes: {self.num_classes}")
            self.logger.info(f"  Training samples: {len(self.train_loader.dataset)}")
            self.logger.info(f"  Validation samples: {len(self.val_loader.dataset)}")
        
        return self.num_classes, self.class_names
    
    def setup_model(self, architecture: str = 'resnet18', 
                   pretrained: bool = True, **kwargs) -> None:
        """
        Setzt Model und Training-Komponenten auf.
        
        Args:
            architecture: Model-Architektur
            pretrained: Ob vortrainierte Gewichte verwendet werden sollen
            **kwargs: Zusätzliche Model-Parameter
            
        Example:
            >>> classifier.setup_model('resnet18', pretrained=True)
            >>> classifier.setup_model('custom_cnn', dropout_rate=0.3)
        """
        # Daten müssen zuerst geladen werden
        if self.num_classes is None:
            self.setup_data()
        
        self.logger.info(f"Setting up model: {architecture}")
        
        # Model Manager erstellen
        self.model_manager = ModelManager(self.config, self.num_classes, self.device)
        
        # Model erstellen
        self.model = self.model_manager.create_model(
            architecture=architecture,
            pretrained=pretrained,
            **kwargs
        )
        
        # Optimizer erstellen
        self.optimizer = self.model_manager.create_optimizer(self.model)
        
        # Trainer erstellen
        self.trainer = Trainer(self.config, self.model, self.optimizer, self.device)
        
        self.logger.info("Model setup completed")
    
    def train(self, architecture: str = 'resnet18', pretrained: bool = True,
              save_results: bool = True, **kwargs) -> Dict:
        """Training with clean, professional visualizations."""
        
        # ... (existing training code until the end) ...
        
        # === REPLACE THIS SECTION ===
        # OLD VERSION (BROKEN):
        # if save_results:
        #     self.save_training_results()
        # 
        # # Visualisierung der Training History
        # self.visualizer.plot_history(self.training_history, save=True, show=True)
        
        # NEW VERSION (CLEAN):
        if save_results:
            self.save_training_results()
    
        # Professional visualization - only working plots
        self.logger.info("Creating training visualizations...")
        
        try:
            # Training history plot (always works)
            history_plot = self.visualizer.plot_training_history(
                self.training_history,
                save=True,
                show=True,
                save_name=f"{architecture}_training_history.png"
            )
            
            self.logger.info(f"Training visualization saved: {history_plot}")
            
        except Exception as e:
            self.logger.warning(f"Visualization creation failed: {e}")
            print("⚠️ Visualization skipped - training data saved successfully")
        
        print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("📊 For comprehensive analysis, use: python -c 'from ml_control_center import ml; ml.dashboard()'")
        
        self.logger.info("Training process completed!")
        return self.training_history

    
    def evaluate(self, model_path: Optional[str] = None, 
                create_visualizations: bool = True) -> Dict:
        """Evaluation with professional visualization system."""
        
        # ... (existing evaluation code until visualization section) ...
        
        # === REPLACE VISUALIZATION SECTION ===
        # OLD VERSION (PROBLEMATIC):
        # if create_visualizations:
        #     # Confusion Matrix
        #     self.visualizer.plot_confusion_matrix(...)
        #     
        #     # Performance Dashboard falls Training History verfügbar
        #     if self.training_history is not None:
        #         self.visualizer.plot_model_performance_summary(...)
        
        # NEW VERSION (PROFESSIONAL):
        if create_visualizations:
            self.logger.info("Creating evaluation visualizations...")
            
            try:
                # Confusion Matrix (always reliable)
                cm_plot = self.visualizer.plot_confusion_matrix(
                    val_results['labels'],
                    val_results['predictions'],
                    self.class_names,
                    title="Validation Results",
                    save=True,
                    show=True
                )
                
                # Comprehensive dashboard if training history available
                if self.training_history is not None:
                    dashboard_plot = self.visualizer.create_comprehensive_dashboard(
                        training_history=self.training_history,
                        evaluation_results=val_results,
                        class_names=self.class_names,
                        save=True,
                        show=False  # Don't show during evaluation
                    )
                    
                    if dashboard_plot:
                        self.logger.info(f"Comprehensive dashboard created: {dashboard_plot}")
                
                self.logger.info("Evaluation visualizations completed")
                
            except Exception as e:
                self.logger.warning(f"Visualization creation failed: {e}")
                print("⚠️ Visualizations skipped - evaluation data available")
        
        return val_results
    
    def predict(self, image_source: Union[str, Path]) -> Dict:
        """
        Macht Prediction für ein einzelnes Bild.
        
        Args:
            image_source: Pfad zum Bild oder Image-Objekt
            
        Returns:
            Prediction-Ergebnis
            
        Example:
            >>> result = classifier.predict("test_image.jpg")
            >>> print(f"Predicted: {result['class_name']} (confidence: {result['confidence']:.3f})")
        """
        if self.predictor is None:
            raise RuntimeError("Model not trained or loaded yet! Call train() or load_model() first.")
        
        return self.predictor.predict_image(image_source)
    
    def predict_batch(self, image_paths: list, **kwargs) -> list:
        """
        Macht Predictions für mehrere Bilder.
        
        Args:
            image_paths: Liste von Bildpfaden
            **kwargs: Zusätzliche Parameter für predict_batch
            
        Returns:
            Liste von Prediction-Ergebnissen
            
        Example:
            >>> image_files = ["img1.jpg", "img2.jpg", "img3.jpg"]
            >>> results = classifier.predict_batch(image_files)
        """
        if self.predictor is None:
            raise RuntimeError("Model not trained or loaded yet! Call train() or load_model() first.")
        
        return self.predictor.predict_batch(image_paths, **kwargs)
    
    def load_model(self, model_path: str, architecture: str = 'resnet18') -> None:
        """
        Lädt ein gespeichertes Model.
        
        Args:
            model_path: Pfad zur Model-Datei
            architecture: Model-Architektur für Rekonstruktion
            
        Example:
            >>> classifier.load_model("models/best_f1_model.pth")
            >>> # Model ist jetzt bereit für Evaluation/Prediction
        """
        # Setup falls noch nicht geschehen
        if self.num_classes is None:
            self.setup_data()
        
        if self.model_manager is None:
            self.model_manager = ModelManager(self.config, self.num_classes, self.device)
        
        # Model erstellen und laden
        self.model = self.model_manager.create_model(architecture)
        self.model = self.model_manager.load_model(self.model, model_path)
        
        # Trainer für Evaluation erstellen
        if self.trainer is None:
            self.optimizer = self.model_manager.create_optimizer(self.model)
            self.trainer = Trainer(self.config, self.model, self.optimizer, self.device)
        
        # Predictor erstellen
        _, val_transform = self.data_manager.get_transforms()
        self.predictor = Predictor(
            self.model, self.class_names, self.device, self.config, val_transform
        )
        
        self.logger.info(f"Model loaded successfully from: {model_path}")
    
    def save_training_results(self, model_name: Optional[str] = None) -> Tuple[str, str]:
        """
        Speichert Training- und Validierungsergebnisse.
        
        Args:
            model_name: Name für die Ergebnisdateien
            
        Returns:
            Tuple mit (train_file_path, val_file_path)
            
        Example:
            >>> train_path, val_path = classifier.save_training_results("my_experiment")
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet! Call train() first.")
        
        if model_name is None:
            from datetime import datetime
            model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return self.evaluation_workflow.save_training_results(
            self, model_name, save_train=True, save_val=True
        )
    
    def compare_with_saved_models(self, saved_model_paths: list) -> str:
        """
        Vergleicht aktuelles Model mit gespeicherten Models.
        
        Args:
            saved_model_paths: Liste von Pfaden zu gespeicherten Models
            
        Returns:
            Pfad zum Vergleichsplot
            
        Example:
            >>> models = ["model1.pth", "model2.pth"]
            >>> plot_path = classifier.compare_with_saved_models(models)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet! Call train() first.")
        
        # Aktuelle Ergebnisse speichern
        train_path, val_path = self.save_training_results("current_model")
        
        # Alle Ergebnisdateien sammeln
        result_files = [val_path]  # Aktuelle Validation-Ergebnisse
        
        # Gespeicherte Models laden und evaluieren
        from .evaluation import StandaloneModelEvaluator
        evaluator = StandaloneModelEvaluator(self.config)
        
        for model_path in saved_model_paths:
            model_name = Path(model_path).stem
            results = evaluator.evaluate_saved_model(
                model_path, model_name, save_results=True, visualize=False
            )
            if 'saved_files' in results and 'validation' in results['saved_files']:
                result_files.append(results['saved_files']['validation'])
        
        # Vergleichsplot erstellen
        return self.evaluation_workflow.compare_models(result_files)
    
    def create_comprehensive_report(self, report_name: str = "full_analysis") -> str:
        """
        Erstellt umfassenden Analyse-Report.
        
        Args:
            report_name: Name des Reports
            
        Returns:
            Pfad zum Report-Verzeichnis
            
        Example:
            >>> report_path = classifier.create_comprehensive_report("final_analysis")
            >>> print(f"Report created: {report_path}")
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet! Call train() first.")
        
        # Ergebnisse speichern
        train_path, val_path = self.save_training_results(f"{report_name}_model")
        
        # Report erstellen
        return self.evaluation_workflow.create_evaluation_report(
            [train_path, val_path],
            report_name=report_name,
            include_individual_plots=True,
            include_comparison=False  # Nur ein Model
        )
    
    def get_model_summary(self) -> Dict:
        """
        Gibt detaillierte Model-Zusammenfassung zurück.
        
        Returns:
            Dictionary mit Model-Informationen
            
        Example:
            >>> summary = classifier.get_model_summary()
            >>> print(f"Parameters: {summary['total_params']:,}")
        """
        if self.model is None:
            raise RuntimeError("Model not created yet! Call setup_model() or train() first.")
        
        return self.model_manager.get_model_summary(self.model)
    
    def get_training_summary(self) -> Dict:
        """
        Gibt Zusammenfassung des Trainings zurück.
        
        Returns:
            Dictionary mit Training-Informationen
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet! Call train() first.")
        
        return self.trainer.get_training_summary()
    
    def get_data_summary(self) -> Dict:
        """
        Gibt Zusammenfassung der Daten zurück.
        
        Returns:
            Dictionary mit Daten-Informationen
        """
        if self.data_manager is None:
            self.setup_data()
        
        return {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'train_samples': len(self.train_loader.dataset),
            'val_samples': len(self.val_loader.dataset),
            'batch_size': self.config.BATCH_SIZE,
            'image_size': self.config.IMG_SIZE
        }
    
    def export_model_for_deployment(self, output_path: str, 
                                  include_metadata: bool = True) -> str:
        """
        Exportiert Model für Deployment.
        
        Args:
            output_path: Pfad für exportiertes Model
            include_metadata: Ob Metadaten eingeschlossen werden sollen
            
        Returns:
            Pfad zum exportierten Model
            
        Example:
            >>> model_path = classifier.export_model_for_deployment("deployed_model.pth")
        """
        if self.model is None:
            raise RuntimeError("Model not created yet! Call train() or load_model() first.")
        
        # Metadaten sammeln
        metadata = {}
        if include_metadata:
            metadata = {
                'class_names': self.class_names,
                'num_classes': self.num_classes,
                'image_size': self.config.IMG_SIZE,
                'model_architecture': self.model.__class__.__name__,
                'confidence_threshold': getattr(self.config, 'PREDICTION_THRESHOLD', 0.5),
                'export_timestamp': time.time()
            }
            
            if self.is_trained:
                metadata.update({
                    'best_accuracy': getattr(self.trainer, 'best_accuracy', None),
                    'best_f1': getattr(self.trainer, 'best_f1', None),
                    'best_loss': getattr(self.trainer, 'best_loss', None)
                })
        
        # Model speichern
        return self.model_manager.save_model(
            self.model, 
            output_path, 
            save_full_model=False,
            metadata=metadata
        )
    
    def quick_test(self, test_images: list, show_results: bool = True) -> list:
        """
        Schneller Test mit einer Liste von Bildern.
        
        Args:
            test_images: Liste von Bildpfaden
            show_results: Ob Ergebnisse ausgegeben werden sollen
            
        Returns:
            Liste von Prediction-Ergebnissen
            
        Example:
            >>> test_imgs = ["test1.jpg", "test2.jpg"]
            >>> results = classifier.quick_test(test_imgs)
        """
        if self.predictor is None:
            raise RuntimeError("Model not ready for prediction! Call train() or load_model() first.")
        
        results = self.predict_batch(test_images, show_progress=False)
        
        if show_results:
            print(f"\n{'='*60}")
            print("QUICK TEST RESULTS")
            print(f"{'='*60}")
            
            for result in results:
                file_name = Path(result['file']).name
                confidence_indicator = "✓" if result['is_confident'] else "?"
                
                print(f"{confidence_indicator} {file_name:<20} -> {result['class_name']:<15} "
                      f"(confidence: {result['confidence']:.3f})")
            
            # Summary
            confident_count = sum(1 for r in results if r['is_confident'])
            print(f"\nSummary: {confident_count}/{len(results)} confident predictions")
            print(f"{'='*60}")
        
        return results
    
    def interactive_prediction(self) -> None:
        """
        Startet interaktive Prediction-Session.
        
        Example:
            >>> classifier.interactive_prediction()
            # Startet interaktive Session für Bildpfad-Eingabe
        """
        if self.predictor is None:
            raise RuntimeError("Model not ready for prediction! Call train() or load_model() first.")
        
        print(f"\n{'='*60}")
        print("INTERACTIVE PREDICTION MODE")
        print(f"{'='*60}")
        print("Enter image path (or 'quit' to exit):")
        
        while True:
            try:
                user_input = input("\nImage path: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                # Prediction durchführen
                result = self.predict(user_input)
                
                # Ergebnis anzeigen
                confidence_indicator = "✓" if result['is_confident'] else "?"
                print(f"\n{confidence_indicator} Prediction: {result['class_name']}")
                print(f"  Confidence: {result['confidence']:.3f}")
                
                # Top-3 Predictions anzeigen
                print(f"  Top 3 predictions:")
                for i, pred in enumerate(result['top_predictions'][:3], 1):
                    print(f"    {i}. {pred['class_name']}: {pred['probability']:.3f}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nInteractive session ended.")
    
    def benchmark_performance(self, test_data_path: str, 
                            num_samples: Optional[int] = None) -> Dict:
        """
        Führt Performance-Benchmark durch.
        
        Args:
            test_data_path: Pfad zu Test-Daten
            num_samples: Anzahl Samples für Benchmark (None = alle)
            
        Returns:
            Benchmark-Ergebnisse
            
        Example:
            >>> benchmark = classifier.benchmark_performance("test_data/")
            >>> print(f"Speed: {benchmark['images_per_second']:.1f} imgs/sec")
        """
        if self.predictor is None:
            raise RuntimeError("Model not ready for prediction! Call train() or load_model() first.")
        
        from .inference import BatchPredictor
        import time
        
        batch_predictor = BatchPredictor(self.model, self.class_names, self.device, self.config)
        
        self.logger.info(f"Starting performance benchmark on {test_data_path}")
        
        start_time = time.time()
        
        # Batch-Processing für Benchmark
        results = batch_predictor.process_large_dataset(
            test_data_path,
            batch_size=self.config.BATCH_SIZE * 2,  # Größere Batches für Speed
            save_intermediate=False
        )
        
        end_time = time.time()
        
        # Benchmark-Metriken berechnen
        total_time = end_time - start_time
        num_images = results['summary']['total_images']
        
        if num_samples and num_images > num_samples:
            # Nur subset verwenden
            scale_factor = num_samples / num_images
            total_time *= scale_factor
            num_images = num_samples
        
        benchmark_results = {
            'total_images': num_images,
            'total_time_seconds': total_time,
            'images_per_second': num_images / total_time,
            'average_time_per_image_ms': (total_time / num_images) * 1000,
            'device': str(self.device),
            'batch_size': self.config.BATCH_SIZE * 2,
            'model_architecture': self.model.__class__.__name__,
            'accuracy_metrics': {
                'confident_percentage': results['summary']['confident_percentage'],
                'average_confidence': results['summary']['average_confidence']
            }
        }
        
        # Benchmark-Report ausgeben
        print(f"\n{'='*60}")
        print("PERFORMANCE BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Images processed: {benchmark_results['total_images']}")
        print(f"Total time: {benchmark_results['total_time_seconds']:.2f} seconds")
        print(f"Speed: {benchmark_results['images_per_second']:.1f} images/second")
        print(f"Avg time per image: {benchmark_results['average_time_per_image_ms']:.1f} ms")
        print(f"Device: {benchmark_results['device']}")
        print(f"Confident predictions: {benchmark_results['accuracy_metrics']['confident_percentage']:.1f}%")
        print(f"{'='*60}")
        
        return benchmark_results
    
    def cleanup(self) -> None:
        """
        Räumt Ressourcen auf.
        
        Example:
            >>> classifier.cleanup()
            # Speicher wird freigegeben
        """
        if self.model is not None and torch.cuda.is_available():
            # GPU Memory cleanup
            del self.model
            del self.optimizer
            torch.cuda.empty_cache()
            
        self.logger.info("Cleanup completed")
    
    def __str__(self) -> str:
        """String-Darstellung des Classifiers."""
        status_lines = [
            "StreetFoodClassifier Status:",
            "=" * 40
        ]
        
        # Data Status
        if self.num_classes is not None:
            status_lines.append(f"Data: {self.num_classes} classes, {len(self.class_names)} names")
        else:
            status_lines.append("Data: Not loaded")
        
        # Model Status
        if self.model is not None:
            model_info = self.get_model_summary()
            status_lines.append(f"Model: {model_info['model_name']} ({model_info['total_params']:,} params)")
        else:
            status_lines.append("Model: Not created")
        
        # Training Status
        if self.is_trained:
            training_summary = self.get_training_summary()
            status_lines.append(f"Training: Completed ({training_summary['total_epochs_trained']} epochs)")
        else:
            status_lines.append("Training: Not completed")
        
        # Prediction Status
        if self.predictor is not None:
            status_lines.append("Prediction: Ready")
        else:
            status_lines.append("Prediction: Not ready")
        
        status_lines.append(f"Device: {self.device}")
        
        return "\n".join(status_lines)