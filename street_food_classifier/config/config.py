"""
Central configuration module for the Street Food Classifier.

This module contains the main Config dataclass that centralizes
all hyperparameters, paths, and settings for the project.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """
    Zentrale Konfigurationsklasse für alle Hyperparameter und Pfade.
    
    Diese Klasse verwendet das dataclass-Pattern für typisierte Konfiguration
    und automatische Initialisierung. Alle Projekteinstellungen sind hier
    zentral verwaltet.
    
    Attributes:
        # Training Hyperparameter
        BATCH_SIZE: Batch-Größe für Training und Validation
        LEARNING_RATE: Lernrate für den Optimizer
        EPOCHS: Maximale Anzahl Trainings-Epochen
        IMG_SIZE: Bildgröße für Resize-Operation
        SEED: Random Seed für Reproduzierbarkeit
        
        # Directory Paths
        DATA_FOLDER: Pfad zum Datenverzeichnis
        MODEL_FOLDER: Pfad für gespeicherte Models
        OUTPUT_FOLDER: Pfad für Outputs (Plots, Logs, etc.)
        
        # Training Parameter
        PATIENCE: Early Stopping Patience
        GAMMA: Learning Rate Decay Factor
        
        # Prediction Settings
        PREDICTION_THRESHOLD: Confidence Threshold für Predictions
        UNKNOWN_LABEL: Label für unbekannte Klassen
        
        # Data Split
        TRAIN_SPLIT: Anteil der Trainingsdaten
    """
    
    # Training Hyperparameter
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 1e-4
    EPOCHS: int = 10
    IMG_SIZE: int = 224
    SEED: int = 42
    
    # Pfade - relative zu Projekt-Root
    DATA_FOLDER: Path = Path('data/processed/popular_street_foods/dataset/dataset')
    MODEL_FOLDER: str = "models/saved_models"
    OUTPUT_FOLDER: str = "outputs"
    
    # Unterordner für verschiedene Output-Typen
    PLOT_FOLDER: str = "outputs/plots"
    LOG_FOLDER: str = "outputs/logs"
    REPORT_FOLDER: str = "outputs/reports"
    EVALUATION_FOLDER: str = "outputs/evaluation_results"
    
    # Legacy compatibility - wird als OUTPUT_FOLDER/plots verwendet
    SAMPLE_FOLDER: str = "outputs/plots"
    
    # Dateinamen für verschiedene Model-Typen
    BEST_F1_MODEL_PATH: str = 'best_f1_model.pth'
    BEST_ACC_MODEL_PATH: str = 'best_acc_model.pth'
    BEST_LOSS_MODEL_PATH: str = 'best_loss_model.pth'
    
    # Weitere Dateien
    HISTORY_PATH: str = "history.json"
    PLOT_IMAGE_PATH: str = 'training_plots.png'
    
    # Training Parameter
    PATIENCE: int = 5
    GAMMA: float = 0.1  # Learning Rate Scheduler Factor
    
    # Prediction Parameter
    PREDICTION_THRESHOLD: float = 0.4
    UNKNOWN_LABEL: str = "unknown"
    
    # Data Split Parameter
    TRAIN_SPLIT: float = 0.8
    
    # Device Settings (optional override)
    DEVICE: Optional[str] = None  # None = auto-detect, 'cuda', 'cpu', 'mps'
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "training.log"
    
    # Data Loading Settings
    NUM_WORKERS: int = 4  # DataLoader workers
    PIN_MEMORY: bool = True  # Pin memory for faster GPU transfer
    
    def __post_init__(self):
        """
        Post-initialization um Ordner zu erstellen und Pfade zu validieren.
        
        Diese Methode wird automatisch nach der dataclass-Initialisierung
        aufgerufen und stellt sicher, dass alle benötigten Verzeichnisse
        existieren.
        """
        # Alle Output-Ordner erstellen
        directories = [
            self.MODEL_FOLDER,
            self.OUTPUT_FOLDER,
            self.PLOT_FOLDER,
            self.LOG_FOLDER,
            self.REPORT_FOLDER,
            self.EVALUATION_FOLDER,
            self.SAMPLE_FOLDER  # Legacy compatibility
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        # Datenordner validieren (warnt aber bricht nicht ab)
        if not self.DATA_FOLDER.exists():
            print(f"⚠️  Data folder not found: {self.DATA_FOLDER}")
            print("   Make sure to set the correct DATA_FOLDER path")
        
        # NUM_WORKERS an verfügbare CPUs anpassen
        if self.NUM_WORKERS < 0:
            self.NUM_WORKERS = min(os.cpu_count() or 4, 8)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """
        Erstellt Config-Instanz aus Dictionary.
        
        Args:
            config_dict: Dictionary mit Konfigurationswerten
            
        Returns:
            Config-Instanz
            
        Example:
            >>> config_data = {"BATCH_SIZE": 32, "LEARNING_RATE": 1e-3}
            >>> config = Config.from_dict(config_data)
        """
        # Nur bekannte Felder übernehmen
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    def to_dict(self) -> dict:
        """
        Konvertiert Config zu Dictionary.
        
        Returns:
            Dictionary mit allen Konfigurationswerten
            
        Example:
            >>> config = Config()
            >>> config_dict = config.to_dict()
            >>> print(config_dict["BATCH_SIZE"])
        """
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def update(self, **kwargs) -> 'Config':
        """
        Erstellt neue Config-Instanz mit aktualisierten Werten.
        
        Args:
            **kwargs: Zu aktualisierende Konfigurationswerte
            
        Returns:
            Neue Config-Instanz mit aktualisierten Werten
            
        Example:
            >>> config = Config()
            >>> new_config = config.update(BATCH_SIZE=32, EPOCHS=20)
        """
        current_dict = self.to_dict()
        current_dict.update(kwargs)
        return self.from_dict(current_dict)
    
    def get_model_path(self, model_type: str = "best_f1") -> str:
        """
        Gibt vollständigen Pfad für Model-Datei zurück.
        
        Args:
            model_type: Typ des Models ('best_f1', 'best_acc', 'best_loss')
            
        Returns:
            Vollständiger Pfad zur Model-Datei
            
        Example:
            >>> config = Config()
            >>> model_path = config.get_model_path("best_f1")
        """
        model_files = {
            "best_f1": self.BEST_F1_MODEL_PATH,
            "best_acc": self.BEST_ACC_MODEL_PATH,
            "best_loss": self.BEST_LOSS_MODEL_PATH
        }
        
        if model_type not in model_files:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Choose from: {list(model_files.keys())}")
        
        return os.path.join(self.MODEL_FOLDER, model_files[model_type])
    
    def get_output_path(self, filename: str, output_type: str = "plots") -> str:
        """
        Gibt vollständigen Output-Pfad zurück.
        
        Args:
            filename: Name der Datei
            output_type: Typ des Outputs ('plots', 'logs', 'reports', 'evaluation')
            
        Returns:
            Vollständiger Pfad zur Output-Datei
            
        Example:
            >>> config = Config()
            >>> plot_path = config.get_output_path("confusion_matrix.png", "plots")
        """
        output_folders = {
            "plots": self.PLOT_FOLDER,
            "logs": self.LOG_FOLDER,
            "reports": self.REPORT_FOLDER,
            "evaluation": self.EVALUATION_FOLDER
        }
        
        if output_type not in output_folders:
            raise ValueError(f"Unknown output type: {output_type}. "
                           f"Choose from: {list(output_folders.keys())}")
        
        return os.path.join(output_folders[output_type], filename)
    
    def __str__(self) -> str:
        """String representation der Konfiguration."""
        lines = ["Configuration Settings:"]
        lines.append("=" * 50)
        
        # Gruppiert nach Kategorien
        categories = {
            "Training": ["BATCH_SIZE", "LEARNING_RATE", "EPOCHS", "IMG_SIZE", "SEED"],
            "Paths": ["DATA_FOLDER", "MODEL_FOLDER", "OUTPUT_FOLDER"],
            "Training Control": ["PATIENCE", "GAMMA", "TRAIN_SPLIT"],
            "Prediction": ["PREDICTION_THRESHOLD", "UNKNOWN_LABEL"],
            "Hardware": ["DEVICE", "NUM_WORKERS", "PIN_MEMORY"]
        }
        
        for category, fields in categories.items():
            lines.append(f"\n{category}:")
            lines.append("-" * len(category))
            for field in fields:
                if hasattr(self, field):
                    value = getattr(self, field)
                    lines.append(f"  {field}: {value}")
        
        return "\n".join(lines)