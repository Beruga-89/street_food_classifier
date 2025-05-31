"""
Prediction and inference utilities.

This module provides classes for making predictions with trained models,
including single image prediction and batch processing.
"""

import os
import time
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from ..utils import get_logger
from ..data.transforms import get_inference_transforms


class Predictor:
    """
    Handhabt Model Predictions für einzelne Bilder und Batch-Verarbeitung.
    
    Diese Klasse bietet:
    - Einzelbild-Prediction mit Confidence-Bewertung
    - Batch-Prediction für mehrere Bilder
    - Support für verschiedene Input-Formate
    - Confidence-basierte Filterung
    
    Example:
        >>> predictor = Predictor(model, class_names, device, config)
        >>> prediction = predictor.predict_image("path/to/image.jpg")
        >>> print(f"Predicted: {prediction['class_name']} (confidence: {prediction['confidence']:.3f})")
    """
    
    def __init__(self, model: nn.Module, class_names: List[str], 
                 device: torch.device, config, transform: Optional[transforms.Compose] = None):
        """
        Initialisiert den Predictor.
        
        Args:
            model: Trainiertes PyTorch Model
            class_names: Liste der Klassennamen
            device: Device (CPU/GPU)
            config: Konfigurationsobjekt mit Prediction-Einstellungen
            transform: Transformationen (optional, wird automatisch erstellt)
        """
        self.model = model
        self.class_names = class_names
        self.device = device
        self.config = config
        self.logger = get_logger(__name__)
        
        # Model in Evaluation-Modus
        self.model.eval()
        
        # Transformationen
        if transform is None:
            img_size = getattr(config, 'IMG_SIZE', 224)
            self.transform = get_inference_transforms(img_size)
        else:
            self.transform = transform
        
        # Prediction-Einstellungen
        self.confidence_threshold = getattr(config, 'PREDICTION_THRESHOLD', 0.5)
        self.unknown_label = getattr(config, 'UNKNOWN_LABEL', 'unknown')
        
        self.logger.info(f"Predictor initialized with {len(class_names)} classes")
        self.logger.info(f"Confidence threshold: {self.confidence_threshold}")
    
    def predict_image(self, source: Union[str, Path, np.ndarray, Image.Image]) -> Dict:
        """
        Sagt Klasse für ein einzelnes Bild vorher.
        
        Args:
            source: Bildquelle (Dateipfad, numpy array, oder PIL Image)
            
        Returns:
            Dictionary mit Prediction-Ergebnissen
            
        Example:
            >>> result = predictor.predict_image("image.jpg")
            >>> print(f"Class: {result['class_name']}")
            >>> print(f"Confidence: {result['confidence']:.3f}")
        """
        # Bild laden und vorbereiten
        image = self._load_image(source)
        input_tensor = self._prepare_input(image)
        
        # Prediction durchführen
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Ergebnisse extrahieren
        confidence_value = confidence.item()
        predicted_class_idx = predicted_idx.item()
        all_probabilities = probabilities[0].cpu().numpy()
        
        # Confidence-basierte Entscheidung
        if confidence_value < self.confidence_threshold:
            predicted_class = self.unknown_label
            is_confident = False
        else:
            predicted_class = self.class_names[predicted_class_idx]
            is_confident = True
        
        # Top-K Predictions
        top_k_probs, top_k_indices = torch.topk(probabilities[0], k=min(5, len(self.class_names)))
        top_predictions = [
            {
                'class_name': self.class_names[idx.item()],
                'probability': prob.item(),
                'class_index': idx.item()
            }
            for prob, idx in zip(top_k_probs, top_k_indices)
        ]
        
        return {
            'class_name': predicted_class,
            'class_index': predicted_class_idx if is_confident else -1,
            'confidence': confidence_value,
            'is_confident': is_confident,
            'probabilities': all_probabilities,
            'top_predictions': top_predictions,
            'threshold_used': self.confidence_threshold
        }
    
    def _load_image(self, source: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        """
        Lädt Bild aus verschiedenen Quellen.
        
        Args:
            source: Bildquelle
            
        Returns:
            PIL Image
        """
        if isinstance(source, (str, Path)):
            # Dateipfad
            if not Path(source).exists():
                raise FileNotFoundError(f"Image file not found: {source}")
            image = Image.open(source).convert('RGB')
            
        elif isinstance(source, np.ndarray):
            # NumPy Array
            if source.ndim == 2:
                # Grayscale zu RGB
                source = np.stack([source] * 3, axis=-1)
            elif source.ndim == 3 and source.shape[2] == 1:
                # Single-channel zu RGB
                source = np.repeat(source, 3, axis=2)
            
            # Normalisierung falls nötig
            if source.dtype != np.uint8:
                if source.max() <= 1.0:
                    source = (source * 255).astype(np.uint8)
                else:
                    source = source.astype(np.uint8)
            
            image = Image.fromarray(source).convert('RGB')
            
        elif isinstance(source, Image.Image):
            # PIL Image
            image = source.convert('RGB')
            
        else:
            raise TypeError(f"Unsupported image source type: {type(source)}")
        
        return image
    
    def _prepare_input(self, image: Image.Image, add_batch_dim: bool = True) -> torch.Tensor:
        """
        Bereitet Bild für Model-Input vor.
        
        Args:
            image: PIL Image
            add_batch_dim: Ob Batch-Dimension hinzugefügt werden soll
            
        Returns:
            Preprocessed tensor
        """
        tensor = self.transform(image)
        
        if add_batch_dim:
            tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)


class BatchPredictor:
    """
    Spezialisierte Klasse für effiziente Batch-Verarbeitung großer Datenmengen.
    
    Diese Klasse optimiert die Verarbeitung für:
    - Große Bildsammlungen
    - Memory-effiziente Verarbeitung
    - Parallele Datenladung
    - Robuste Fehlerbehandlung
    
    Example:
        >>> batch_predictor = BatchPredictor(model, class_names, device, config)
        >>> results = batch_predictor.process_large_dataset("dataset/", batch_size=64)
    """
    
    def __init__(self, model: nn.Module, class_names: List[str], 
                 device: torch.device, config):
        """
        Initialisiert den BatchPredictor.
        
        Args:
            model: Trainiertes PyTorch Model
            class_names: Liste der Klassennamen
            device: Device (CPU/GPU)
            config: Konfigurationsobjekt
        """
        self.model = model
        self.class_names = class_names
        self.device = device
        self.config = config
        self.logger = get_logger(__name__)
        
        # Model in Evaluation-Modus
        self.model.eval()
        
        # Transformationen
        img_size = getattr(config, 'IMG_SIZE', 224)
        self.transform = get_inference_transforms(img_size)
        
        # Batch-Einstellungen
        self.default_batch_size = getattr(config, 'BATCH_SIZE', 32)
        self.confidence_threshold = getattr(config, 'PREDICTION_THRESHOLD', 0.5)
        
        self.logger.info("BatchPredictor initialized")