"""
Device management utilities for PyTorch.

This module provides functions for device detection and management,
including GPU memory management and device information.
"""

import logging
import torch
from typing import Optional, Dict, Any


def get_device(preferred_device: Optional[str] = None) -> torch.device:
    """
    Bestimmt das beste verfügbare Device oder verwendet ein bevorzugtes.
    
    Args:
        preferred_device: Bevorzugtes Device ('cuda', 'cpu', 'mps')
                         Falls None, wird automatisch das beste gewählt
    
    Returns:
        torch.device: CUDA falls verfügbar, sonst CPU
        
    Example:
        >>> device = get_device()
        >>> print(f"Using: {device}")
        >>> 
        >>> # Spezifisches Device erzwingen
        >>> cpu_device = get_device('cpu')
    """
    logger = logging.getLogger(__name__)
    
    if preferred_device is not None:
        device = torch.device(preferred_device)
        logger.info(f"Using preferred device: {device}")
        return device
    
    # Automatische Device-Wahl
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {device}")
        logger.info(f"GPU Count: {gpu_count}, GPU Name: {gpu_name}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info(f"Using Apple Silicon MPS: {device}")
    else:
        device = torch.device('cpu')
        logger.info(f"Using CPU: {device}")
    
    return device


def get_device_info() -> Dict[str, Any]:
    """
    Sammelt detaillierte Informationen über verfügbare Devices.
    
    Returns:
        Dictionary mit Device-Informationen
        
    Example:
        >>> info = get_device_info()
        >>> print(f"CUDA available: {info['cuda_available']}")
        >>> print(f"GPU memory: {info['gpu_memory_gb']:.1f} GB")
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'torch_version': torch.__version__,
    }
    
    if torch.cuda.is_available():
        # GPU-spezifische Informationen
        info.update({
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
            'current_device': torch.cuda.current_device(),
        })
    
    return info


def clear_gpu_memory() -> None:
    """
    Leert den GPU-Speicher (falls CUDA verfügbar).
    
    Nützlich zwischen Experimenten oder bei OutOfMemory-Fehlern.
    
    Example:
        >>> clear_gpu_memory()
        >>> # GPU memory is now cleared
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logging.getLogger(__name__).info("GPU memory cleared")


def set_gpu_device(device_id: int) -> torch.device:
    """
    Setzt ein spezifisches GPU-Device.
    
    Args:
        device_id: GPU Device ID (0, 1, 2, ...)
        
    Returns:
        torch.device für die gewählte GPU
        
    Raises:
        RuntimeError: Falls CUDA nicht verfügbar oder Device ID ungültig
        
    Example:
        >>> device = set_gpu_device(1)  # Verwende GPU 1
        >>> model = model.to(device)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    if device_id >= torch.cuda.device_count():
        raise RuntimeError(f"Device ID {device_id} not available. "
                          f"Only {torch.cuda.device_count()} GPUs found.")
    
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')
    
    logger = logging.getLogger(__name__)
    logger.info(f"Set GPU device: {device}")
    logger.info(f"GPU name: {torch.cuda.get_device_name(device_id)}")
    
    return device


def get_memory_usage() -> Dict[str, float]:
    """
    Gibt GPU-Speicherverbrauch zurück (falls CUDA verfügbar).
    
    Returns:
        Dictionary mit Speicherinformationen in GB
        
    Example:
        >>> memory = get_memory_usage()
        >>> print(f"Used: {memory['used']:.1f} GB")
        >>> print(f"Free: {memory['free']:.1f} GB")
    """
    if not torch.cuda.is_available():
        return {'used': 0.0, 'free': 0.0, 'total': 0.0}
    
    used = torch.cuda.memory_allocated() / 1e9
    cached = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - cached
    
    return {
        'used': used,
        'cached': cached,
        'free': free,
        'total': total
    }