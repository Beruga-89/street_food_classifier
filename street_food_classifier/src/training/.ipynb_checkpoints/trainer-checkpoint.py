"""
Main training logic for the Street Food Classifier.

This module contains the Trainer class which handles the complete
training pipeline including training loops, evaluation, and model saving.
"""

import os
import time
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ..utils import get_logger
from .metrics import Metrics, MetricsCalculator, MetricsHistory
from .early_stopping import EarlyStopping


class Trainer:
    """
    Hauptklasse für das Training von Deep Learning Modellen.
    
    Diese Klasse verwaltet den kompletten Trainingsprozess:
    - Training und Validation Loops
    - Metrik-Berechnung
    - Model Checkpointing
    - Early Stopping
    - Learning Rate Scheduling
    
    Example:
        >>> trainer = Trainer(config, model, optimizer, device)
        >>> history = trainer.fit(train_loader, val_loader)
        >>> results = trainer.evaluate(test_loader)
    """
    
    def __init__(self, config, model: nn.Module, optimizer: torch.optim.Optimizer,
                 device: torch.device, criterion: Optional[nn.Module] = None):
        """
        Initialisiert den Trainer.
        
        Args:
            config: Konfigurationsobjekt
            model: PyTorch Model
            optimizer: Optimizer (z.B. Adam, SGD)
            device: Device (CPU/GPU)
            criterion: Loss-Funktion (default: CrossEntropyLoss)
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.logger = get_logger(__name__)
        
        # Best values tracking für Model Saving
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
        
        # Metrics History
        self.metrics_history = MetricsHistory()
        
        # Learning Rate Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',  # Maximiere F1-Score
            factor=getattr(config, 'GAMMA', 0.1),
            patience=getattr(config, 'SCHEDULER_PATIENCE', 3),
        )
        
        # Early Stopping (optional)
        self.early_stopping = None
        if hasattr(config, 'PATIENCE') and config.PATIENCE > 0:
            self.early_stopping = EarlyStopping(
                patience=config.PATIENCE,
                monitor='val_f1',
                mode='max',
                restore_best_weights=True
            )
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader, 
                return_predictions: bool = True) -> Dict:
        """
        Evaluiert das Model auf einem Dataset.
        
        Args:
            data_loader: DataLoader für Evaluation
            return_predictions: Ob Predictions zurückgegeben werden sollen
            
        Returns:
            Dictionary mit Evaluation-Ergebnissen
            
        Example:
            >>> results = trainer.evaluate(val_loader)
            >>> print(f"Accuracy: {results['accuracy']:.4f}")
            >>> print(f"F1-Score: {results['f1']:.4f}")
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_samples = 0
        
        # Progress bar für Evaluation
        eval_loop = tqdm(data_loader, desc="Evaluating", leave=False, disable=False)
        
        for batch_idx, (images, labels) in enumerate(eval_loop):
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = images.size(0)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Accumulate loss
            total_loss += loss.item() * batch_size
            num_samples += batch_size
            
            # Predictions
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            current_acc = (preds == labels.cpu().numpy()).mean()
            eval_loop.set_postfix({
                'loss': loss.item(),
                'acc': current_acc
            })
        
        # Konvertiere zu numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Berechne finale Metriken
        avg_loss = total_loss / num_samples
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1
        }
        
        # Optional: Predictions zurückgeben
        if return_predictions:
            results.update({
                'predictions': all_preds,
                'labels': all_labels
            })
        
        return results
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Trainiert eine Epoche.
        
        Args:
            train_loader: DataLoader für Training
            
        Returns:
            Dictionary mit Trainingsergebnissen der Epoche
        """
        self.model.train()
        
        losses = []
        all_preds = []
        all_labels = []
        
        # Progress bar für Training
        train_loop = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (images, labels) in enumerate(train_loop):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            losses.append(loss.item())
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Predictions für Metriken
            predicted = outputs.argmax(dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            batch_acc = (predicted == labels).float().mean().item()
            train_loop.set_postfix({
                'loss': loss.item(),
                'acc': batch_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # Berechne Epoche-Metriken
        avg_loss = np.mean(losses)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1
        }
    
    def save_best_models(self, val_metrics: Dict[str, float], epoch: int) -> bool:
        """
        Speichert beste Models basierend auf verschiedenen Metriken.
        
        Args:
            val_metrics: Validation Metriken
            epoch: Aktuelle Epoche
            
        Returns:
            True wenn mindestens ein Model gespeichert wurde
        """
        updated = False
        
        # Best Loss Model
        if val_metrics['loss'] < self.best_loss:
            self.best_loss = val_metrics['loss']
            loss_path = os.path.join(self.config.MODEL_FOLDER, 'best_loss_model.pth')
            torch.save(self.model.state_dict(), loss_path)
            self.logger.info(f"[BEST LOSS] Model saved at epoch {epoch + 1} (Loss: {val_metrics['loss']:.4f})")
            updated = True
        
        # Best Accuracy Model
        if val_metrics['accuracy'] > self.best_accuracy:
            self.best_accuracy = val_metrics['accuracy']
            acc_path = os.path.join(self.config.MODEL_FOLDER, 'best_acc_model.pth')
            torch.save(self.model.state_dict(), acc_path)
            self.logger.info(f"[BEST ACC] Model saved at epoch {epoch + 1} (Acc: {val_metrics['accuracy']:.4f})")
            updated = True
            
        # Best F1 Model
        if val_metrics['f1'] > self.best_f1:
            self.best_f1 = val_metrics['f1']
            f1_path = os.path.join(self.config.MODEL_FOLDER, 'best_f1_model.pth')
            torch.save(self.model.state_dict(), f1_path)
            self.logger.info(f"[BEST F1] Model saved at epoch {epoch + 1} (F1: {val_metrics['f1']:.4f})")
            updated = True
            
        return updated
    
    def save_checkpoint(self, epoch: int, train_metrics: Dict, val_metrics: Dict,
                       is_best: bool = False) -> str:
        """
        Speichert einen vollständigen Training Checkpoint.
        
        Args:
            epoch: Aktuelle Epoche
            train_metrics: Training Metriken
            val_metrics: Validation Metriken
            is_best: Ob dies der beste Checkpoint ist
            
        Returns:
            Pfad zum gespeicherten Checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_loss': self.best_loss,
            'best_accuracy': self.best_accuracy,
            'best_f1': self.best_f1,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else None
        }
        
        # Checkpoint Dateiname
        checkpoint_name = f"checkpoint_epoch_{epoch + 1}.pth"
        if is_best:
            checkpoint_name = f"best_checkpoint.pth"
            
        checkpoint_path = os.path.join(self.config.MODEL_FOLDER, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Lädt einen Training Checkpoint.
        
        Args:
            checkpoint_path: Pfad zum Checkpoint
            
        Returns:
            Checkpoint Dictionary
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Model und Optimizer State laden
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Best values wiederherstellen
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {checkpoint['epoch']}")
        
        return checkpoint
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Haupttraining Loop.
        
        Args:
            train_loader: DataLoader für Training
            val_loader: DataLoader für Validation
            
        Returns:
            Training history Dictionary
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Training for {self.config.EPOCHS} epochs")
        self.logger.info(f"Device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(self.config.EPOCHS):
            epoch_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.EPOCHS}")
            print(f"{'='*60}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.evaluate(val_loader, return_predictions=False)
            
            # Metrics zu History hinzufügen
            train_metrics_obj = Metrics(
                loss=train_metrics['loss'],
                accuracy=train_metrics['accuracy'],
                f1=train_metrics['f1']
            )
            val_metrics_obj = Metrics(
                loss=val_metrics['loss'],
                accuracy=val_metrics['accuracy'],
                f1=val_metrics['f1']
            )
            
            self.metrics_history.add_epoch(train_metrics_obj, val_metrics_obj)
            
            # Best models speichern
            model_updated = self.save_best_models(val_metrics, epoch)
            
            # Early Stopping Check
            should_stop = False
            if self.early_stopping is not None:
                current_weights = self.model.state_dict().copy()
                should_stop = self.early_stopping.should_stop(
                    val_metrics['f1'], epoch, current_weights
                )
                
                if should_stop:
                    self.logger.info(f"[EARLY STOP] Training stopped at epoch {epoch + 1}")
                    self.logger.info(f"Best F1: {self.early_stopping.get_best_score():.4f} "
                                   f"at epoch {self.early_stopping.get_best_epoch() + 1}")
                    
                    # Restore best weights falls gewünscht
                    if (self.early_stopping.restore_best_weights and 
                        self.early_stopping.best_weights is not None):
                        self.model.load_state_dict(self.early_stopping.best_weights)
                        self.logger.info("Best weights restored")
            
            # Learning Rate Scheduling
            self.scheduler.step(val_metrics['f1'])
            
            # Epoch Logging
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nResults:")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            print(f"LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")
            
            if self.early_stopping is not None:
                print(f"Early Stopping: {self.early_stopping.get_patience_counter()}/{self.early_stopping.patience}")
            
            # Optional: Checkpoint speichern
            if hasattr(self.config, 'SAVE_CHECKPOINTS') and self.config.SAVE_CHECKPOINTS:
                self.save_checkpoint(epoch, train_metrics, val_metrics, is_best=model_updated)
            
            if should_stop:
                break
        
        # Training Summary
        total_time = time.time() - start_time
        self.logger.info(f"\nTraining completed!")
        self.logger.info(f"Total time: {total_time/60:.1f} minutes")
        self.logger.info(f"Best Loss: {self.best_loss:.4f}")
        self.logger.info(f"Best Accuracy: {self.best_accuracy:.4f}")
        self.logger.info(f"Best F1: {self.best_f1:.4f}")
        
        return self.metrics_history.to_dict()
    
    def get_learning_rate(self) -> float:
        """Gibt aktuelle Learning Rate zurück."""
        return self.optimizer.param_groups[0]['lr']
    
    def set_learning_rate(self, lr: float) -> None:
        """Setzt neue Learning Rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.logger.info(f"Learning rate set to {lr:.2e}")
    
    def get_training_summary(self) -> Dict:
        """
        Gibt Zusammenfassung des Trainings zurück.
        
        Returns:
            Dictionary mit Training-Zusammenfassung
        """
        return {
            'best_loss': self.best_loss,
            'best_accuracy': self.best_accuracy,
            'best_f1': self.best_f1,
            'total_epochs_trained': len(self.metrics_history.history['train']['loss']),
            'final_lr': self.get_learning_rate(),
            'early_stopping_summary': (
                self.early_stopping.get_summary() 
                if self.early_stopping is not None 
                else None
            )
        }