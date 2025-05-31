"""
Early stopping utilities for training.

This module provides classes for implementing early stopping
to prevent overfitting during training.
"""

import numpy as np
from typing import Optional


class EarlyStopping:
    """
    Early Stopping Implementation um Overfitting zu verhindern.
    
    Diese Klasse überwacht eine Metrik und stoppt das Training wenn
    sich diese über mehrere Epochen nicht verbessert.
    
    Example:
        >>> early_stopping = EarlyStopping(patience=5, monitor='val_f1', mode='max')
        >>> 
        >>> for epoch in range(epochs):
        >>>     # ... training ...
        >>>     val_f1 = evaluate_model()
        >>>     
        >>>     if early_stopping.should_stop(val_f1):
        >>>         print("Early stopping triggered!")
        >>>         break
    """
    
    def __init__(self, patience: int = 5, monitor: str = 'val_loss', 
                 mode: str = 'min', min_delta: float = 0.0,
                 restore_best_weights: bool = True):
        """
        Initialisiert Early Stopping.
        
        Args:
            patience: Anzahl Epochen ohne Verbesserung vor dem Stoppen
            monitor: Name der zu überwachenden Metrik
            mode: 'min' für Minimierung (z.B. Loss) oder 'max' für Maximierung (z.B. Accuracy)
            min_delta: Minimale Änderung um als Verbesserung zu gelten
            restore_best_weights: Ob beste Gewichte wiederhergestellt werden sollen
        """
        self.patience = patience
        self.monitor = monitor
        self.mode = mode.lower()
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        # Validierung
        if self.mode not in ['min', 'max']:
            raise ValueError("Mode must be 'min' or 'max'")
        
        # Interne Variablen
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.should_stop_flag = False
        
        # Für restore_best_weights (wird von außen gesetzt)
        self.best_weights = None
    
    def __call__(self, current_score: float, epoch: int = None, 
                 model_weights: Optional[dict] = None) -> bool:
        """
        Überprüft ob Training gestoppt werden sollte.
        
        Args:
            current_score: Aktueller Wert der überwachten Metrik
            epoch: Aktuelle Epoche (optional, für Logging)
            model_weights: Model Gewichte (für restore_best_weights)
            
        Returns:
            True wenn Training gestoppt werden sollte
        """
        return self.should_stop(current_score, epoch, model_weights)
    
    def should_stop(self, current_score: float, epoch: int = None,
                   model_weights: Optional[dict] = None) -> bool:
        """
        Hauptlogik für Early Stopping Entscheidung.
        
        Args:
            current_score: Aktueller Wert der überwachten Metrik
            epoch: Aktuelle Epoche (optional)
            model_weights: Model Gewichte (optional)
            
        Returns:
            True wenn Training gestoppt werden sollte
        """
        if epoch is not None:
            self.current_epoch = epoch
        
        if self.best_score is None:
            # Erste Epoche
            self.best_score = current_score
            self.best_epoch = epoch or 0
            if model_weights is not None:
                self.best_weights = model_weights.copy()
            return False
        
        # Prüfen ob Verbesserung vorliegt
        if self._is_improvement(current_score):
            self.best_score = current_score
            self.best_epoch = epoch or 0
            self.counter = 0
            
            if model_weights is not None:
                self.best_weights = model_weights.copy()
                
        else:
            self.counter += 1
            
        # Early Stopping auslösen?
        if self.counter >= self.patience:
            self.should_stop_flag = True
            self.stopped_epoch = epoch or 0
            return True
            
        return False
    
    def _is_improvement(self, current_score: float) -> bool:
        """
        Prüft ob aktueller Score eine Verbesserung darstellt.
        
        Args:
            current_score: Zu prüfender Score
            
        Returns:
            True wenn Verbesserung vorliegt
        """
        if self.mode == 'min':
            return current_score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            return current_score > (self.best_score + self.min_delta)
    
    def get_best_score(self) -> float:
        """Gibt den besten bisher erreichten Score zurück."""
        return self.best_score
    
    def get_best_epoch(self) -> int:
        """Gibt die Epoche mit dem besten Score zurück.""" 
        return self.best_epoch
    
    def get_patience_counter(self) -> int:
        """Gibt aktuellen Patience Counter zurück."""
        return self.counter
    
    def reset(self) -> None:
        """Setzt Early Stopping zurück."""
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.should_stop_flag = False
        self.best_weights = None
    
    def get_summary(self) -> dict:
        """
        Gibt Zusammenfassung des Early Stopping Verlaufs zurück.
        
        Returns:
            Dictionary mit Early Stopping Informationen
        """
        return {
            'monitor': self.monitor,
            'mode': self.mode,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'stopped_epoch': self.stopped_epoch,
            'final_patience_counter': self.counter,
            'early_stopped': self.should_stop_flag
        }
    
    def __str__(self) -> str:
        """String-Darstellung des Early Stopping Status."""
        if self.best_score is None:
            return f"EarlyStopping(monitor={self.monitor}, patience={self.patience}) - No data yet"
        
        status = "STOPPED" if self.should_stop_flag else "ACTIVE"
        return (f"EarlyStopping({status}) - "
                f"Best {self.monitor}: {self.best_score:.4f} at epoch {self.best_epoch}, "
                f"Patience: {self.counter}/{self.patience}")


class LearningRateScheduler:
    """
    Custom Learning Rate Scheduler mit Early Stopping Integration.
    
    Diese Klasse kann als Ergänzung zum Early Stopping verwendet werden
    um die Learning Rate bei Stagnation zu reduzieren.
    """
    
    def __init__(self, patience: int = 3, factor: float = 0.5, 
                 monitor: str = 'val_loss', mode: str = 'min',
                 min_lr: float = 1e-7, verbose: bool = True):
        """
        Initialisiert Learning Rate Scheduler.
        
        Args:
            patience: Epochen ohne Verbesserung vor LR-Reduktion
            factor: Faktor um den LR reduziert wird
            monitor: Zu überwachende Metrik
            mode: 'min' oder 'max'
            min_lr: Minimale Learning Rate
            verbose: Ob Änderungen geloggt werden sollen
        """
        self.patience = patience
        self.factor = factor
        self.monitor = monitor
        self.mode = mode.lower()
        self.min_lr = min_lr
        self.verbose = verbose
        
        # Tracking
        self.best_score = None
        self.counter = 0
        self.num_reductions = 0
        
    def should_reduce_lr(self, current_score: float) -> bool:
        """
        Prüft ob Learning Rate reduziert werden sollte.
        
        Args:
            current_score: Aktueller Score der überwachten Metrik
            
        Returns:
            True wenn LR reduziert werden sollte
        """
        if self.best_score is None:
            self.best_score = current_score
            return False
            
        # Prüfen auf Verbesserung (gleiche Logik wie EarlyStopping)
        is_improvement = False
        if self.mode == 'min':
            is_improvement = current_score < self.best_score
        else:
            is_improvement = current_score > self.best_score
            
        if is_improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def reduce_lr(self, optimizer, current_lr: float) -> float:
        """
        Reduziert Learning Rate des Optimizers.
        
        Args:
            optimizer: PyTorch Optimizer
            current_lr: Aktuelle Learning Rate
            
        Returns:
            Neue Learning Rate
        """
        new_lr = max(current_lr * self.factor, self.min_lr)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
            
        self.counter = 0  # Reset counter nach Reduktion
        self.num_reductions += 1
        
        if self.verbose:
            print(f"Learning Rate reduced: {current_lr:.2e} -> {new_lr:.2e}")
            
        return new_lr