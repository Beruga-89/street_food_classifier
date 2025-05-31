"""
Model management utilities for the Street Food Classifier.

This module contains the ModelManager class responsible for creating,
loading, saving, and managing PyTorch models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any
from pathlib import Path

from ..utils import get_logger
from .architectures import (
    create_resnet18, create_resnet50, create_efficientnet_b0,
    create_custom_cnn, create_mobilenet_v2, get_model_info
)


class ModelManager:
    """
    Verwaltet Model-bezogene Operationen.
    
    Diese Klasse ist verantwortlich für:
    - Erstellen verschiedener Model-Architekturen
    - Laden und Speichern von Models
    - Optimizer-Erstellung
    - Model-Transfer zwischen Devices
    
    Example:
        >>> manager = ModelManager(config, num_classes=10, device=device)
        >>> model = manager.create_model('resnet18', pretrained=True)
        >>> optimizer = manager.create_optimizer(model, 'adam')
    """
    
    def __init__(self, config, num_classes: int, device: torch.device):
        """
        Initialisiert den ModelManager.
        
        Args:
            config: Konfigurationsobjekt mit MODEL_FOLDER, etc.
            num_classes: Anzahl der Ausgabe-Klassen
            device: Device (CPU/GPU) für Model-Operations
        """
        self.config = config
        self.num_classes = num_classes
        self.device = device
        self.logger = get_logger(__name__)
        
        # Verfügbare Architekturen
        self.available_architectures = {
            'resnet18': create_resnet18,
            'resnet50': create_resnet50,
            'efficientnet_b0': create_efficientnet_b0,
            'custom_cnn': create_custom_cnn,
            'mobilenet_v2': create_mobilenet_v2
        }
        
    def create_model(self, architecture: str = 'resnet18', pretrained: bool = True,
                    freeze_backbone: bool = False, **kwargs) -> nn.Module:
        """
        Erstellt und konfiguriert ein Model.
        
        Args:
            architecture: Model-Architektur ('resnet18', 'resnet50', etc.)
            pretrained: Ob vortrainierte Gewichte verwendet werden sollen
            freeze_backbone: Ob Backbone eingefroren werden soll
            **kwargs: Zusätzliche Parameter für die Architektur
            
        Returns:
            Konfiguriertes PyTorch Model
            
        Example:
            >>> model = manager.create_model('resnet18', pretrained=True)
            >>> model = manager.create_model('custom_cnn', dropout_rate=0.3)
        """
        if architecture not in self.available_architectures:
            raise ValueError(f"Unknown architecture: {architecture}. "
                           f"Available: {list(self.available_architectures.keys())}")
        
        # Model erstellen
        create_fn = self.available_architectures[architecture]
        
        if architecture == 'custom_cnn':
            # Custom CNN braucht keine pretrained Parameter
            model = create_fn(
                num_classes=self.num_classes,
                **kwargs
            )
        else:
            # Vortrainierte Models
            model = create_fn(
                num_classes=self.num_classes,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                **kwargs
            )
        
        # Model zu Device verschieben
        model = model.to(self.device)
        
        # Model Info loggen
        info = get_model_info(model)
        self.logger.info(f"Created {architecture} model:")
        self.logger.info(f"  Total parameters: {info['total_params']:,}")
        self.logger.info(f"  Trainable parameters: {info['trainable_params']:,}")
        self.logger.info(f"  Model size: {info['model_size_mb']:.1f} MB")
        
        if freeze_backbone:
            frozen_ratio = info['frozen_params'] / info['total_params'] * 100
            self.logger.info(f"  Frozen parameters: {frozen_ratio:.1f}%")
        
        return model
    
    def create_optimizer(self, model: nn.Module, optimizer_type: str = 'adam',
                        learning_rate: Optional[float] = None, **kwargs) -> torch.optim.Optimizer:
        """
        Erstellt einen Optimizer für das Model.
        
        Args:
            model: PyTorch Model
            optimizer_type: Typ des Optimizers ('adam', 'sgd', 'adamw', 'rmsprop')
            learning_rate: Learning Rate (falls None, aus Config)
            **kwargs: Zusätzliche Optimizer-Parameter
            
        Returns:
            Konfigurierter Optimizer
            
        Example:
            >>> optimizer = manager.create_optimizer(model, 'adam', weight_decay=1e-4)
            >>> optimizer = manager.create_optimizer(model, 'sgd', momentum=0.9)
        """
        if learning_rate is None:
            learning_rate = getattr(self.config, 'LEARNING_RATE', 1e-4)
        
        optimizer_type = optimizer_type.lower()
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                model.parameters(), 
                lr=learning_rate,
                **kwargs
            )
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                **kwargs
            )
        elif optimizer_type == 'sgd':
            # Standard SGD Parameter falls nicht gegeben
            if 'momentum' not in kwargs:
                kwargs['momentum'] = 0.9
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                **kwargs
            )
        elif optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}. "
                           f"Available: adam, adamw, sgd, rmsprop")
        
        self.logger.info(f"Created {optimizer_type} optimizer with LR: {learning_rate:.2e}")
        
        return optimizer
    
    def save_model(self, model: nn.Module, filename: str, 
                  save_full_model: bool = False, metadata: Optional[Dict] = None) -> str:
        """
        Speichert ein Model.
        
        Args:
            model: Zu speicherndes Model
            filename: Dateiname (ohne Pfad)
            save_full_model: Ob das komplette Model oder nur state_dict gespeichert werden soll
            metadata: Zusätzliche Metadaten
            
        Returns:
            Vollständiger Pfad der gespeicherten Datei
            
        Example:
            >>> path = manager.save_model(model, 'my_model.pth')
            >>> path = manager.save_model(model, 'my_model.pth', 
            ...                          metadata={'epoch': 10, 'accuracy': 0.95})
        """
        # Vollständigen Pfad erstellen
        save_path = os.path.join(self.config.MODEL_FOLDER, filename)
        
        # Verzeichnis erstellen falls nötig
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if save_full_model:
            # Komplettes Model speichern (inkl. Architektur)
            save_data = {
                'model': model,
                'metadata': metadata or {}
            }
        else:
            # Nur state_dict speichern (empfohlen)
            save_data = {
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'num_classes': self.num_classes,
                'metadata': metadata or {}
            }
        
        torch.save(save_data, save_path)
        
        self.logger.info(f"Model saved to: {save_path}")
        return save_path
    
    def load_model(self, model: nn.Module, model_path: str, 
                  strict: bool = True) -> nn.Module:
        """
        Lädt ein gespeichertes Model.
        
        Args:
            model: Model-Instanz (für state_dict loading)
            model_path: Pfad zum Model (kann relativ zum MODEL_FOLDER sein)
            strict: Ob strict loading verwendet werden soll
            
        Returns:
            Model mit geladenen Gewichten
            
        Example:
            >>> model = manager.create_model('resnet18')
            >>> model = manager.load_model(model, 'best_f1_model.pth')
        """
        # Vollständigen Pfad erstellen falls relativer Pfad
        if not os.path.isabs(model_path):
            full_path = os.path.join(self.config.MODEL_FOLDER, model_path)
        else:
            full_path = model_path
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")
        
        # Model laden
        try:
            if torch.cuda.is_available() and self.device.type == 'cuda':
                checkpoint = torch.load(full_path, weights_only=True)
            else:
                checkpoint = torch.load(full_path, weights_only=True, map_location='cpu')
        except Exception:
            # Fallback für ältere PyTorch Versionen
            if torch.cuda.is_available() and self.device.type == 'cuda':
                checkpoint = torch.load(full_path)
            else:
                checkpoint = torch.load(full_path, map_location='cpu')
        
        # Unterschiedliche Formate handhaben
        if 'model_state_dict' in checkpoint:
            # Standard Format (state_dict + metadata)
            state_dict = checkpoint['model_state_dict']
            model.load_state_dict(state_dict, strict=strict)
            
            # Metadata loggen falls vorhanden
            if 'metadata' in checkpoint:
                metadata = checkpoint['metadata']
                self.logger.info(f"Model metadata: {metadata}")
                
        elif 'model' in checkpoint:
            # Komplettes Model Format
            saved_model = checkpoint['model']
            model.load_state_dict(saved_model.state_dict(), strict=strict)
            
        else:
            # Nur state_dict (direktes Format)
            model.load_state_dict(checkpoint, strict=strict)
        
        # Model zu korrektem Device verschieben
        model = model.to(self.device)
        
        self.logger.info(f"Model loaded from: {full_path}")
        return model
    
    def get_model_path(self, model_type: str) -> str:
        """
        Gibt vollständigen Pfad für vordefinierte Model-Typen zurück.
        
        Args:
            model_type: Typ des Models ('best_f1', 'best_acc', 'best_loss')
            
        Returns:
            Vollständiger Pfad zur Model-Datei
        """
        model_files = {
            "best_f1": getattr(self.config, 'BEST_F1_MODEL_PATH', 'best_f1_model.pth'),
            "best_acc": getattr(self.config, 'BEST_ACC_MODEL_PATH', 'best_acc_model.pth'),
            "best_loss": getattr(self.config, 'BEST_LOSS_MODEL_PATH', 'best_loss_model.pth')
        }
        
        if model_type not in model_files:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Choose from: {list(model_files.keys())}")
        
        return os.path.join(self.config.MODEL_FOLDER, model_files[model_type])
    
    def list_saved_models(self) -> list:
        """
        Listet alle gespeicherten Models im MODEL_FOLDER auf.
        
        Returns:
            Liste von Model-Dateien
            
        Example:
            >>> models = manager.list_saved_models()
            >>> for model_file in models:
            ...     print(f"Found model: {model_file}")
        """
        model_folder = Path(self.config.MODEL_FOLDER)
        
        if not model_folder.exists():
            return []
        
        # Alle .pth und .pt Dateien finden
        model_files = []
        for pattern in ['*.pth', '*.pt']:
            model_files.extend(model_folder.glob(pattern))
        
        # Zu relativen Pfaden konvertieren
        relative_paths = [str(f.relative_to(model_folder)) for f in model_files]
        relative_paths.sort()
        
        return relative_paths
    
    def copy_model(self, source_path: str, target_name: str) -> str:
        """
        Kopiert ein Model zu einem neuen Namen.
        
        Args:
            source_path: Quell-Model Pfad
            target_name: Neuer Dateiname
            
        Returns:
            Pfad der kopierten Datei
        """
        import shutil
        
        source_full = os.path.join(self.config.MODEL_FOLDER, source_path)
        target_full = os.path.join(self.config.MODEL_FOLDER, target_name)
        
        if not os.path.exists(source_full):
            raise FileNotFoundError(f"Source model not found: {source_full}")
        
        shutil.copy2(source_full, target_full)
        self.logger.info(f"Model copied: {source_path} -> {target_name}")
        
        return target_full
    
    def get_model_summary(self, model: nn.Module) -> Dict[str, Any]:
        """
        Gibt detaillierte Model-Zusammenfassung zurück.
        
        Args:
            model: PyTorch Model
            
        Returns:
            Dictionary mit Model-Informationen
        """
        info = get_model_info(model)
        
        # Zusätzliche Informationen
        info.update({
            'device': str(next(model.parameters()).device),
            'num_classes': self.num_classes,
            'training_mode': model.training,
        })
        
        # Layer-Count
        conv_layers = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        linear_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        bn_layers = sum(1 for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)))
        
        info.update({
            'conv_layers': conv_layers,
            'linear_layers': linear_layers,
            'batchnorm_layers': bn_layers
        })
        
        return info