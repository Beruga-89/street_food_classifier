"""
JSON serialization utilities for NumPy and PyTorch data types.

This module provides utilities to handle serialization of NumPy arrays,
PyTorch tensors, and other non-standard Python types to JSON format.
"""

import json
import numpy as np
import torch
from typing import Any, Dict, List, Union
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder für NumPy und PyTorch Datentypen.
    
    Dieser Encoder kann folgende Typen serialisieren:
    - NumPy integers, floats, booleans
    - NumPy arrays
    - PyTorch tensors (werden zu NumPy konvertiert)
    - Python standard types
    
    Example:
        >>> data = {"array": np.array([1, 2, 3]), "value": np.float32(3.14)}
        >>> json_str = json.dumps(data, cls=NumpyEncoder)
    """
    
    def default(self, obj: Any) -> Any:
        """
        Konvertiert Objekte zu JSON-serialisierbaren Typen.
        
        Args:
            obj: Objekt zum Konvertieren
            
        Returns:
            JSON-serialisierbares Objekt
        """
        # NumPy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        # PyTorch types
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        
        # Python Path objects
        elif isinstance(obj, Path):
            return str(obj)
        
        # Fallback to default JSON encoder
        return super().default(obj)


def convert_numpy_types(obj: Any) -> Any:
    """
    Konvertiert NumPy und PyTorch Datentypen rekursiv zu Python Standard-Typen.
    
    Diese Funktion ist nützlich für die Vorbereitung von Daten zur JSON-Serialisierung
    ohne die Verwendung eines benutzerdefinierten Encoders.
    
    Args:
        obj: Objekt das konvertiert werden soll
        
    Returns:
        Konvertiertes Objekt mit Python Standard-Typen
        
    Example:
        >>> data = {"array": np.array([1, 2, 3]), "nested": {"value": np.float32(3.14)}}
        >>> clean_data = convert_numpy_types(data)
        >>> json_str = json.dumps(clean_data)  # Works without custom encoder
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def save_json(data: Dict[str, Any], file_path: Union[str, Path], 
              use_numpy_encoder: bool = True, **kwargs) -> None:
    """
    Speichert Daten als JSON-Datei mit NumPy-Unterstützung.
    
    Args:
        data: Zu speichernde Daten
        file_path: Pfad zur Ausgabedatei
        use_numpy_encoder: Ob NumpyEncoder verwendet werden soll
        **kwargs: Zusätzliche Argumente für json.dump()
        
    Example:
        >>> data = {"metrics": {"accuracy": np.float32(0.95)}}
        >>> save_json(data, "results.json")
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Default kwargs
    json_kwargs = {
        'indent': 2,
        'ensure_ascii': False,
        **kwargs
    }
    
    if use_numpy_encoder:
        json_kwargs['cls'] = NumpyEncoder
    else:
        # Convert data first
        data = convert_numpy_types(data)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, **json_kwargs)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Lädt JSON-Daten aus einer Datei.
    
    Args:
        file_path: Pfad zur JSON-Datei
        
    Returns:
        Geladene Daten als Dictionary
        
    Raises:
        FileNotFoundError: Falls Datei nicht existiert
        json.JSONDecodeError: Falls JSON-Format ungültig
        
    Example:
        >>> data = load_json("results.json")
        >>> print(data["metrics"]["accuracy"])
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_json_serializable(obj: Any) -> bool:
    """
    Prüft ob ein Objekt JSON-serialisierbar ist.
    
    Args:
        obj: Zu prüfendes Objekt
        
    Returns:
        True falls serialisierbar, False sonst
        
    Example:
        >>> data = {"array": np.array([1, 2, 3])}
        >>> is_valid = validate_json_serializable(data)
        >>> if not is_valid:
        >>>     data = convert_numpy_types(data)
    """
    try:
        json.dumps(obj, cls=NumpyEncoder)
        return True
    except (TypeError, ValueError):
        return False