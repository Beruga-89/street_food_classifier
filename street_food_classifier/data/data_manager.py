"""
Data management and loading utilities.

This module contains the DataManager class responsible for creating
data loaders with proper train/validation splits and transforms.
"""

import os
import logging
from typing import Tuple, List
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

from ..utils import get_logger
from .transforms import get_transforms_pair


class DataManager:
    """
    Verwaltet alle Datenoperationen für das Machine Learning Pipeline.
    
    Diese Klasse ist verantwortlich für:
    - Laden von Datasets
    - Erstellen von Train/Validation Splits
    - Konfiguration von DataLoadern
    - Anwendung von Transformationen
    
    Example:
        >>> from config import Config
        >>> config = Config()
        >>> data_manager = DataManager(config)
        >>> train_loader, val_loader, num_classes, class_names = data_manager.create_dataloaders()
    """
    
    def __init__(self, config):
        """
        Initialisiert den DataManager.
        
        Args:
            config: Konfigurationsobjekt mit DATA_FOLDER, BATCH_SIZE, etc.
            
        Raises:
            AttributeError: Falls required config attributes fehlen
            FileNotFoundError: Falls Datenordner nicht existiert
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Validierung der Config
        self._validate_config()
        
        # Prüfe ob Datenordner existiert
        if not Path(self.config.DATA_FOLDER).exists():
            raise FileNotFoundError(
                f"Data folder not found: {self.config.DATA_FOLDER}\n"
                f"Please check your DATA_FOLDER path in config."
            )
    
    def _validate_config(self) -> None:
        """Validiert dass Config alle benötigten Attribute hat."""
        required_attrs = [
            'DATA_FOLDER', 'BATCH_SIZE', 'IMG_SIZE', 
            'TRAIN_SPLIT', 'SEED'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr):
                raise AttributeError(f"Config must have {attr} attribute")
    
    def get_transforms(self) -> Tuple:
        """
        Erstellt Transformationen für Training und Validierung.
        
        Returns:
            Tuple mit (train_transform, val_transform)
            
        Example:
            >>> train_transform, val_transform = data_manager.get_transforms()
        """
        return get_transforms_pair(self.config.IMG_SIZE)
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, int, List[str]]:
        """
        Erstellt DataLoader für Training und Validierung mit Stratified Split.
        
        Diese Methode:
        1. Lädt den vollständigen Dataset
        2. Führt einen stratifizierten Split durch (gleiche Klassenverteilung)
        3. Erstellt separate Datasets mit entsprechenden Transformationen
        4. Konfiguriert DataLoader mit optimalen Einstellungen
        
        Returns:
            Tuple mit (train_loader, val_loader, num_classes, class_names)
            
        Example:
            >>> train_loader, val_loader, num_classes, class_names = data_manager.create_dataloaders()
            >>> print(f"Dataset has {num_classes} classes: {class_names}")
            >>> print(f"Training batches: {len(train_loader)}")
            >>> print(f"Validation batches: {len(val_loader)}")
        """
        train_transform, val_transform = self.get_transforms()
        
        # Vollständigen Dataset laden für Stratified Split
        full_dataset = ImageFolder(self.config.DATA_FOLDER, transform=None)
        num_classes = len(full_dataset.classes)
        class_names = full_dataset.classes
        
        self.logger.info(f"Found {num_classes} classes: {class_names}")
        self.logger.info(f"Total images: {len(full_dataset)}")
        
        # Labels für Stratified Split extrahieren
        targets = [full_dataset[i][1] for i in range(len(full_dataset))]
        indices = list(range(len(full_dataset)))
        
        # Stratified Split durchführen
        train_indices, val_indices = train_test_split(
            indices,
            test_size=1 - self.config.TRAIN_SPLIT,
            stratify=targets,
            random_state=self.config.SEED
        )
        
        # Separate Datasets mit entsprechenden Transformationen erstellen
        train_dataset = ImageFolder(self.config.DATA_FOLDER, transform=train_transform)
        val_dataset = ImageFolder(self.config.DATA_FOLDER, transform=val_transform)
        
        # Subsets erstellen
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        
        # DataLoader Konfiguration
        num_workers = self._get_optimal_num_workers()
        pin_memory = getattr(self.config, 'PIN_MEMORY', torch.cuda.is_available())
        
        # DataLoader erstellen
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # Für konsistente Batch-Größen
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        # Logging der finalen Split-Informationen
        self._log_split_info(train_subset, val_subset, class_names, targets, 
                           train_indices, val_indices)
        
        return train_loader, val_loader, num_classes, class_names
    
    def _get_optimal_num_workers(self) -> int:
        """
        Bestimmt optimale Anzahl Worker für DataLoader.
        
        Returns:
            Optimale Anzahl Worker
        """
        if hasattr(self.config, 'NUM_WORKERS') and self.config.NUM_WORKERS >= 0:
            return self.config.NUM_WORKERS
        
        # Automatische Bestimmung
        cpu_count = os.cpu_count() or 4
        optimal_workers = min(cpu_count, 8)  # Begrenzt auf max 8
        
        self.logger.info(f"Auto-detected {optimal_workers} workers for DataLoader")
        return optimal_workers
    
    def _log_split_info(self, train_subset: Subset, val_subset: Subset, 
                       class_names: List[str], targets: List[int],
                       train_indices: List[int], val_indices: List[int]) -> None:
        """
        Loggt detaillierte Informationen über den Train/Val Split.
        """
        self.logger.info(f'[STRATIFIED SPLIT] Completed successfully')
        self.logger.info(f'Train Dataset: {len(train_subset)} images')
        self.logger.info(f'Val Dataset: {len(val_subset)} images')
        self.logger.info(f'Split ratio: {self.config.TRAIN_SPLIT:.1%} train / '
                        f'{1-self.config.TRAIN_SPLIT:.1%} validation')
        
        # Per-Klassen-Verteilung loggen
        self._log_class_distribution(class_names, targets, train_indices, val_indices)
    
    def _log_class_distribution(self, class_names: List[str], targets: List[int],
                               train_indices: List[int], val_indices: List[int]) -> None:
        """
        Loggt die Klassenverteilung im Train/Val Split.
        """
        from collections import Counter
        
        # Klassenverteilung berechnen
        train_targets = [targets[i] for i in train_indices]
        val_targets = [targets[i] for i in val_indices]
        
        train_counts = Counter(train_targets)
        val_counts = Counter(val_targets)
        
        self.logger.info("Class distribution:")
        for i, class_name in enumerate(class_names):
            train_count = train_counts.get(i, 0)
            val_count = val_counts.get(i, 0)
            total_count = train_count + val_count
            
            self.logger.info(
                f"  {class_name}: {total_count} total "
                f"({train_count} train, {val_count} val)"
            )
    
    def create_single_dataloader(self, dataset_path: str, 
                                batch_size: int = None,
                                transform_type: str = 'val',
                                shuffle: bool = False) -> Tuple[DataLoader, List[str]]:
        """
        Erstellt einen einzelnen DataLoader für Inferenz oder Test.
        
        Args:
            dataset_path: Pfad zum Dataset
            batch_size: Batch-Größe (default: config.BATCH_SIZE)
            transform_type: 'train' oder 'val' transforms
            shuffle: Ob Dataset gemischt werden soll
            
        Returns:
            Tuple mit (dataloader, class_names)
            
        Example:
            >>> test_loader, classes = data_manager.create_single_dataloader(
            ...     'data/test', transform_type='val'
            ... )
        """
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        
        # Transform wählen
        if transform_type == 'train':
            transform, _ = self.get_transforms()
        else:
            _, transform = self.get_transforms()
        
        # Dataset laden
        dataset = ImageFolder(dataset_path, transform=transform)
        
        # DataLoader erstellen
        num_workers = self._get_optimal_num_workers()
        pin_memory = getattr(self.config, 'PIN_MEMORY', torch.cuda.is_available())
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        self.logger.info(f"Created single DataLoader: {len(dataset)} images, "
                        f"{len(dataloader)} batches")
        
        return dataloader, dataset.classes
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Berechnet Klassengewichte für unbalancierte Datasets.
        
        Returns:
            Tensor mit Klassengewichten
            
        Example:
            >>> weights = data_manager.get_class_weights()
            >>> criterion = nn.CrossEntropyLoss(weight=weights)
        """
        # Dataset laden um Klassenverteilung zu analysieren
        dataset = ImageFolder(self.config.DATA_FOLDER, transform=None)
        targets = [dataset[i][1] for i in range(len(dataset))]
        
        from collections import Counter
        class_counts = Counter(targets)
        
        # Gewichte berechnen (invers proportional zur Häufigkeit)
        total_samples = len(targets)
        num_classes = len(dataset.classes)
        
        weights = []
        for i in range(num_classes):
            count = class_counts.get(i, 1)  # Mindestens 1 um Division durch 0 zu vermeiden
            weight = total_samples / (num_classes * count)
            weights.append(weight)
        
        weights_tensor = torch.FloatTensor(weights)
        
        self.logger.info("Calculated class weights:")
        for i, (class_name, weight) in enumerate(zip(dataset.classes, weights)):
            self.logger.info(f"  {class_name}: {weight:.4f}")
        
        return weights_tensor