"""
Input/Output utilities for file and directory operations.

This module provides helper functions for common I/O operations
used throughout the project.
"""

import os
import shutil
from pathlib import Path
from typing import Union, List, Optional
import logging


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Stellt sicher, dass ein Verzeichnis existiert (erstellt es falls nötig).
    
    Args:
        path: Pfad zum Verzeichnis
        
    Returns:
        Path-Objekt des Verzeichnisses
        
    Example:
        >>> model_dir = ensure_dir("models/experiments")
        >>> # Verzeichnis existiert jetzt garantiert
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(*paths: Union[str, Path]) -> List[Path]:
    """
    Stellt sicher, dass mehrere Verzeichnisse existieren.
    
    Args:
        *paths: Variable Anzahl von Verzeichnispfaden
        
    Returns:
        Liste von Path-Objekten
        
    Example:
        >>> dirs = ensure_dirs("models", "outputs", "logs")
        >>> model_dir, output_dir, log_dir = dirs
    """
    return [ensure_dir(path) for path in paths]


def copy_file(src: Union[str, Path], dst: Union[str, Path], 
              create_dirs: bool = True) -> None:
    """
    Kopiert eine Datei von Quelle zu Ziel.
    
    Args:
        src: Quell-Dateipfad
        dst: Ziel-Dateipfad
        create_dirs: Ob Zielverzeichnis erstellt werden soll
        
    Example:
        >>> copy_file("config.yaml", "backup/config.yaml")
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    if create_dirs:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(src_path, dst_path)


def move_file(src: Union[str, Path], dst: Union[str, Path], 
              create_dirs: bool = True) -> None:
    """
    Verschiebt eine Datei von Quelle zu Ziel.
    
    Args:
        src: Quell-Dateipfad
        dst: Ziel-Dateipfad  
        create_dirs: Ob Zielverzeichnis erstellt werden soll
        
    Example:
        >>> move_file("temp_model.pth", "models/final_model.pth")
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    if create_dirs:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.move(str(src_path), str(dst_path))


def remove_file(path: Union[str, Path], missing_ok: bool = True) -> None:
    """
    Löscht eine Datei.
    
    Args:
        path: Pfad zur zu löschenden Datei
        missing_ok: Ob Fehler ignoriert werden soll falls Datei nicht existiert
        
    Example:
        >>> remove_file("temp_file.txt")
    """
    path = Path(path)
    
    try:
        path.unlink()
    except FileNotFoundError:
        if not missing_ok:
            raise


def remove_dir(path: Union[str, Path], missing_ok: bool = True) -> None:
    """
    Löscht ein Verzeichnis und alle Inhalte.
    
    Args:
        path: Pfad zum zu löschenden Verzeichnis
        missing_ok: Ob Fehler ignoriert werden soll falls Verzeichnis nicht existiert
        
    Example:
        >>> remove_dir("temp_outputs")
    """
    path = Path(path)
    
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        if not missing_ok:
            raise


def get_file_size(path: Union[str, Path], unit: str = 'MB') -> float:
    """
    Gibt die Dateigröße in der gewünschten Einheit zurück.
    
    Args:
        path: Pfad zur Datei
        unit: Einheit ('B', 'KB', 'MB', 'GB')
        
    Returns:
        Dateigröße in der gewünschten Einheit
        
    Example:
        >>> size_mb = get_file_size("model.pth", "MB")
        >>> print(f"Model size: {size_mb:.2f} MB")
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    size_bytes = path.stat().st_size
    
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
        'TB': 1024**4
    }
    
    if unit not in units:
        raise ValueError(f"Invalid unit: {unit}. Choose from {list(units.keys())}")
    
    return size_bytes / units[unit]


def list_files(directory: Union[str, Path], pattern: str = "*", 
               recursive: bool = False) -> List[Path]:
    """
    Listet Dateien in einem Verzeichnis auf.
    
    Args:
        directory: Verzeichnispfad
        pattern: Dateinamenmuster (glob pattern)
        recursive: Ob Unterverzeichnisse durchsucht werden sollen
        
    Returns:
        Liste von Dateipfaden
        
    Example:
        >>> model_files = list_files("models", "*.pth")
        >>> json_files = list_files("outputs", "*.json", recursive=True)
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def backup_file(file_path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None, 
                timestamp: bool = True) -> Path:
    """
    Erstellt ein Backup einer Datei.
    
    Args:
        file_path: Pfad zur zu sichernden Datei
        backup_dir: Backup-Verzeichnis (default: same directory)
        timestamp: Ob Zeitstempel an Dateinamen angehängt werden soll
        
    Returns:
        Pfad zur Backup-Datei
        
    Example:
        >>> backup_path = backup_file("important_config.yaml")
        >>> print(f"Backup created: {backup_path}")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File to backup not found: {file_path}")
    
    if backup_dir is None:
        backup_dir = file_path.parent
    else:
        backup_dir = Path(backup_dir)
        ensure_dir(backup_dir)
    
    # Generate backup filename
    if timestamp:
        from datetime import datetime
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp_str}{file_path.suffix}"
    else:
        backup_name = f"{file_path.stem}_backup{file_path.suffix}"
    
    backup_path = backup_dir / backup_name
    copy_file(file_path, backup_path)
    
    logging.getLogger(__name__).info(f"Backup created: {backup_path}")
    return backup_path