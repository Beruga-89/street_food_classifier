"""
Logging utilities for the Street Food Classifier project.

This module provides standardized logging setup that is Windows-compatible
and follows best practices for ML projects.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = __name__, 
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Erstellt einen strukturierten Logger für das Projekt.
    Windows-kompatibel ohne Unicode-Probleme.
    
    Args:
        name: Name des Loggers
        level: Logging Level (default: INFO)
        log_file: Name der Log-Datei (default: 'training.log')
        log_dir: Verzeichnis für Log-Dateien (default: current directory)
        
    Returns:
        Konfigurierter Logger
        
    Example:
        >>> logger = setup_logger("my_module", logging.DEBUG)
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Verhindere doppelte Handler
    if not logger.handlers:
        # Log directory setup
        if log_dir is None:
            log_dir = Path.cwd()
        else:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
        if log_file is None:
            log_file = 'training.log'
            
        log_path = log_dir / log_file
        
        # File Handler mit UTF-8 Encoding für Windows
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        
        # Console Handler - nur für wichtige Meldungen
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Weniger Console-Output
        
        # Formatter ohne Emojis für Windows-Kompatibilität
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get an existing logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)