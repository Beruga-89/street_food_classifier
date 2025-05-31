"""
Metrics calculation utilities for training and evaluation.

This module provides classes and functions for calculating various
machine learning metrics during training and evaluation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class Metrics:
    """
    Datenklasse für Trainingsmetriken.
    
    Diese Klasse kapselt alle wichtigen Metriken die während des
    Trainings und der Evaluation berechnet werden.
    
    Attributes:
        loss: Durchschnittlicher Verlust
        accuracy: Genauigkeit (Anteil korrekter Predictions)
        f1: F1-Score (harmonisches Mittel aus Precision und Recall)
        precision: Precision (optional)
        recall: Recall (optional)
    """
    loss: float
    accuracy: float
    f1: float
    precision: float = None
    recall: float = None
    
    def __str__(self) -> str:
        """String-Darstellung der Metriken."""
        base_str = f"Loss: {self.loss:.4f}, Acc: {self.accuracy:.4f}, F1: {self.f1:.4f}"
        
        if self.precision is not None and self.recall is not None:
            base_str += f", Prec: {self.precision:.4f}, Rec: {self.recall:.4f}"
        
        return base_str
    
    def to_dict(self) -> Dict[str, float]:
        """
        Konvertiert Metriken zu Dictionary.
        
        Returns:
            Dictionary mit Metrik-Namen als Keys
        """
        result = {
            'loss': self.loss,
            'accuracy': self.accuracy,
            'f1': self.f1
        }
        
        if self.precision is not None:
            result['precision'] = self.precision
        if self.recall is not None:
            result['recall'] = self.recall
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'Metrics':
        """
        Erstellt Metrics-Objekt aus Dictionary.
        
        Args:
            data: Dictionary mit Metrik-Werten
            
        Returns:
            Metrics-Instanz
        """
        return cls(
            loss=data['loss'],
            accuracy=data['accuracy'],
            f1=data['f1'],
            precision=data.get('precision'),
            recall=data.get('recall')
        )


class MetricsCalculator:
    """
    Berechnet verschiedene Metriken für Machine Learning Modelle.
    
    Diese Klasse bietet statische Methoden zur Berechnung von Standard-Metriken
    sowie erweiterte Analysefunktionen.
    """
    
    @staticmethod
    def calculate_metrics(predictions: np.ndarray, labels: np.ndarray, 
                         total_loss: float, dataset_size: int,
                         include_precision_recall: bool = False,
                         average: str = 'weighted') -> Metrics:
        """
        Berechnet Metriken aus Predictions und Labels.
        
        Args:
            predictions: Predicted labels (numpy array)
            labels: True labels (numpy array)
            total_loss: Gesamtverlust über alle Samples
            dataset_size: Größe des Datasets
            include_precision_recall: Ob Precision und Recall berechnet werden sollen
            average: Averaging-Methode für F1, Precision, Recall ('weighted', 'macro', 'micro')
            
        Returns:
            Metrics object mit berechneten Werten
            
        Example:
            >>> predictions = np.array([0, 1, 2, 1])
            >>> labels = np.array([0, 1, 1, 1])
            >>> metrics = MetricsCalculator.calculate_metrics(
            ...     predictions, labels, total_loss=1.5, dataset_size=4
            ... )
            >>> print(metrics)
        """
        avg_loss = total_loss / dataset_size
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average=average, zero_division=0)
        
        precision = None
        recall = None
        
        if include_precision_recall:
            precision = precision_score(labels, predictions, average=average, zero_division=0)
            recall = recall_score(labels, predictions, average=average, zero_division=0)
        
        return Metrics(
            loss=avg_loss,
            accuracy=accuracy,
            f1=f1,
            precision=precision,
            recall=recall
        )
    
    @staticmethod
    def calculate_per_class_metrics(predictions: np.ndarray, labels: np.ndarray,
                                   class_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Berechnet Per-Klassen-Metriken.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            class_names: Namen der Klassen
            
        Returns:
            Dictionary mit Per-Klassen-Metriken
            
        Example:
            >>> per_class = MetricsCalculator.calculate_per_class_metrics(
            ...     predictions, labels, ['cat', 'dog', 'bird']
            ... )
            >>> print(per_class['cat']['f1'])
        """
        # Per-Klassen F1, Precision, Recall berechnen
        f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)
        precision_per_class = precision_score(labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(labels, predictions, average=None, zero_division=0)
        
        # Per-Klassen Accuracy (aus Confusion Matrix)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, predictions)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Dictionary erstellen
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            if i < len(f1_per_class):  # Sicherheitscheck
                per_class_metrics[class_name] = {
                    'accuracy': per_class_accuracy[i],
                    'precision': precision_per_class[i],
                    'recall': recall_per_class[i],
                    'f1': f1_per_class[i]
                }
        
        return per_class_metrics
    
    @staticmethod
    def calculate_confusion_matrix_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Berechnet Confusion Matrix und abgeleitete Metriken.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            
        Returns:
            Dictionary mit Confusion Matrix Informationen
            
        Example:
            >>> cm_metrics = MetricsCalculator.calculate_confusion_matrix_metrics(
            ...     predictions, labels
            ... )
            >>> print(cm_metrics['confusion_matrix'])
        """
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(labels, predictions)
        
        # Normalisierte Confusion Matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Weitere Metriken
        total_samples = cm.sum()
        correct_predictions = np.trace(cm)
        
        return {
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'total_samples': int(total_samples),
            'correct_predictions': int(correct_predictions),
            'per_class_accuracy': cm.diagonal() / cm.sum(axis=1),
            'per_class_samples': cm.sum(axis=1)
        }
    
    @staticmethod
    def calculate_macro_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Berechnet Macro-averaged Metriken.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            
        Returns:
            Dictionary mit Macro-Metriken
        """
        return {
            'macro_accuracy': accuracy_score(labels, predictions),
            'macro_precision': precision_score(labels, predictions, average='macro', zero_division=0),
            'macro_recall': recall_score(labels, predictions, average='macro', zero_division=0),
            'macro_f1': f1_score(labels, predictions, average='macro', zero_division=0)
        }
    
    @staticmethod
    def calculate_weighted_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Berechnet Weighted-averaged Metriken.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            
        Returns:
            Dictionary mit Weighted-Metriken
        """
        return {
            'weighted_accuracy': accuracy_score(labels, predictions),
            'weighted_precision': precision_score(labels, predictions, average='weighted', zero_division=0),
            'weighted_recall': recall_score(labels, predictions, average='weighted', zero_division=0),
            'weighted_f1': f1_score(labels, predictions, average='weighted', zero_division=0)
        }


class MetricsHistory:
    """
    Verwaltet Historie von Metriken über mehrere Epochen.
    
    Diese Klasse sammelt Metriken während des Trainings und bietet
    Funktionen zur Analyse und Speicherung.
    """
    
    def __init__(self):
        """Initialisiert leere Metrik-Historie."""
        self.history = {
            "train": {"loss": [], "accuracy": [], "f1": [], "precision": [], "recall": []}, 
            "val": {"loss": [], "accuracy": [], "f1": [], "precision": [], "recall": []}
        }
        
    def add_epoch(self, train_metrics: Metrics, val_metrics: Metrics) -> None:
        """
        Fügt Metriken einer Epoche hinzu.
        
        Args:
            train_metrics: Training-Metriken
            val_metrics: Validation-Metriken
        """
        # Training Metriken
        self.history["train"]["loss"].append(train_metrics.loss)
        self.history["train"]["accuracy"].append(train_metrics.accuracy)
        self.history["train"]["f1"].append(train_metrics.f1)
        
        if train_metrics.precision is not None:
            self.history["train"]["precision"].append(train_metrics.precision)
        if train_metrics.recall is not None:
            self.history["train"]["recall"].append(train_metrics.recall)
        
        # Validation Metriken
        self.history["val"]["loss"].append(val_metrics.loss)
        self.history["val"]["accuracy"].append(val_metrics.accuracy)
        self.history["val"]["f1"].append(val_metrics.f1)
        
        if val_metrics.precision is not None:
            self.history["val"]["precision"].append(val_metrics.precision)
        if val_metrics.recall is not None:
            self.history["val"]["recall"].append(val_metrics.recall)
    
    def get_best_epoch(self, metric: str = "f1", split: str = "val") -> int:
        """
        Findet Epoche mit bestem Wert für gegebene Metrik.
        
        Args:
            metric: Metrik-Name ('loss', 'accuracy', 'f1', etc.)
            split: 'train' oder 'val'
            
        Returns:
            Epoch-Index (0-basiert) mit bestem Wert
        """
        if metric not in self.history[split]:
            raise ValueError(f"Metric '{metric}' not found in history")
        
        values = self.history[split][metric]
        if not values:
            raise ValueError("No history data available")
        
        if metric == "loss":
            # Für Loss: niedrigster Wert ist besser
            return np.argmin(values)
        else:
            # Für andere Metriken: höchster Wert ist besser
            return np.argmax(values)
    
    def get_latest_metrics(self) -> tuple:
        """
        Gibt die Metriken der letzten Epoche zurück.
        
        Returns:
            Tuple mit (train_metrics, val_metrics)
        """
        if not self.history["train"]["loss"]:
            raise ValueError("No history data available")
        
        train_metrics = Metrics(
            loss=self.history["train"]["loss"][-1],
            accuracy=self.history["train"]["accuracy"][-1],
            f1=self.history["train"]["f1"][-1],
            precision=self.history["train"]["precision"][-1] if self.history["train"]["precision"] else None,
            recall=self.history["train"]["recall"][-1] if self.history["train"]["recall"] else None
        )
        
        val_metrics = Metrics(
            loss=self.history["val"]["loss"][-1],
            accuracy=self.history["val"]["accuracy"][-1],
            f1=self.history["val"]["f1"][-1],
            precision=self.history["val"]["precision"][-1] if self.history["val"]["precision"] else None,
            recall=self.history["val"]["recall"][-1] if self.history["val"]["recall"] else None
        )
        
        return train_metrics, val_metrics
    
    def to_dict(self) -> Dict:
        """Konvertiert Historie zu Dictionary."""
        return self.history.copy()
    
    def from_dict(self, data: Dict) -> None:
        """Lädt Historie aus Dictionary."""
        self.history = data.copy()