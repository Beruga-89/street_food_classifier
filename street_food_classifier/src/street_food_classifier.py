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
        """
        Training with professional error handling and visualization.
        
        Args:
            architecture: Model architecture to use
            pretrained: Whether to use pretrained weights
            save_results: Whether to save training results
            **kwargs: Additional training parameters
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting training with {architecture}")
        self.logger.info(f"Configuration: pretrained={pretrained}, save_results={save_results}")
        
        try:
            # Setup data if not already done
            if self.num_classes is None:
                self.logger.info("Setting up data...")
                self.setup_data()
            
            # Setup model if not already done
            if self.model is None:
                self.logger.info(f"Setting up model: {architecture}")
                self.setup_model(architecture=architecture, pretrained=pretrained, **kwargs)
            
            # Ensure we have data loaders
            if self.train_loader is None or self.val_loader is None:
                self.logger.info("Creating data loaders...")
                (self.train_loader, self.val_loader, 
                 self.num_classes, self.class_names) = self.data_manager.create_dataloaders()
            
            # Start training
            self.logger.info("Starting training process...")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Epochs: {self.config.EPOCHS}")
            self.logger.info(f"Batch size: {self.config.BATCH_SIZE}")
            
            # Training durchführen
            self.training_history = self.trainer.train(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                epochs=self.config.EPOCHS
            )
            
            # Training Status setzen
            self.is_trained = True
            
            # Predictor für spätere Verwendung erstellen
            if self.predictor is None:
                _, val_transform = self.data_manager.get_transforms()
                self.predictor = Predictor(
                    self.model, self.class_names, self.device, self.config, val_transform
                )
            
            self.logger.info("Training completed successfully!")
            
            # Ergebnisse speichern
            if save_results:
                self.logger.info("Saving training results...")
                try:
                    self.save_training_results()
                    self.logger.info("Training results saved")
                except Exception as e:
                    self.logger.warning(f"Failed to save training results: {e}")
            
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
                
                if history_plot:
                    self.logger.info(f"Training visualization saved: {history_plot}")
                
            except Exception as e:
                self.logger.warning(f"Visualization creation failed: {e}")
                print("⚠️ Visualization skipped - training data saved successfully")
            
            # Training summary
            if self.training_history:
                final_train_loss = self.training_history['train']['loss'][-1]
                final_val_loss = self.training_history['val']['loss'][-1]
                
                if 'accuracy' in self.training_history['train']:
                    final_train_acc = self.training_history['train']['accuracy'][-1]
                    final_val_acc = self.training_history['val']['accuracy'][-1]
                    
                    print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
                    print(f"📊 Final Results:")
                    print(f"   Train Loss: {final_train_loss:.4f} | Train Acc: {final_train_acc:.4f}")
                    print(f"   Val Loss: {final_val_loss:.4f} | Val Acc: {final_val_acc:.4f}")
                else:
                    print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
                    print(f"📊 Final Results:")
                    print(f"   Train Loss: {final_train_loss:.4f}")
                    print(f"   Val Loss: {final_val_loss:.4f}")
            
            print("📊 For comprehensive analysis, use ml.evaluate() or ml.status()")
            
            self.logger.info("Training process completed!")
            return self.training_history
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.is_trained = False
            
            # Detailed error information
            import traceback
            self.logger.error(f"Training error traceback:\n{traceback.format_exc()}")
            
            # Try to provide helpful error message
            if "CUDA out of memory" in str(e):
                print("❌ CUDA out of memory! Try reducing batch size in config")
            elif "No such file or directory" in str(e):
                print("❌ Data files not found! Check your data paths in config")
            elif "dimension mismatch" in str(e):
                print("❌ Model dimension error! Check num_classes and data compatibility")
            else:
                print(f"❌ Training error: {e}")
            
            raise e
    
    def evaluate(self, model_path: Optional[str] = None, 
                create_visualizations: bool = True) -> Dict:
        """
        Evaluation with professional visualization system.
        """
        self.logger.info("Starting evaluation...")
        
        # Load model if path provided
        if model_path is not None:
            self.logger.info(f"Loading model from: {model_path}")
            self.load_model(model_path)
        
        # Check if model is ready
        if self.model is None:
            raise RuntimeError("No model available for evaluation! Train a model or provide model_path.")
        
        # Ensure data is loaded
        if self.train_loader is None or self.val_loader is None:
            self.setup_data()
        
        # Perform evaluation
        self.logger.info("Running validation...")
        val_results = self.trainer.evaluate(self.val_loader)
        
        self.logger.info("Evaluation completed")
        self.logger.info(f"Validation Accuracy: {val_results['accuracy']:.4f}")
        self.logger.info(f"Validation F1-Score: {val_results['f1']:.4f}")
        
        # Professional visualization
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