"""
Random seed utilities for reproducible experiments.

This module provides functions to set random seeds across all major libraries
used in deep learning to ensure reproducible results.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Setzt alle Random Seeds für vollständige Reproduzierbarkeit.
    
    Diese Funktion setzt Seeds für:
    - Python's random module
    - NumPy
    - PyTorch (CPU und GPU)
    - CUDA operations
    - cuDNN deterministic mode
    
    Args:
        seed: Random Seed Wert
        
    Note:
        Das Setzen von `torch.backends.cudnn.deterministic = True` kann
        die Performance beeinträchtigen, garantiert aber Reproduzierbarkeit.
        
    Example:
        >>> seed_everything(42)
        >>> # Alle nachfolgenden Operationen sind reproduzierbar
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Für vollständige Reproduzierbarkeit
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Python hash seed für string hashing
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_random_state() -> dict:
    """
    Erfasst den aktuellen Random State aller Libraries.
    
    Returns:
        Dictionary mit Random States
        
    Example:
        >>> state = get_random_state()
        >>> # ... einige Operationen ...
        >>> restore_random_state(state)  # Wiederherstellen
    """
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }


def restore_random_state(state: dict) -> None:
    """
    Stellt einen zuvor gespeicherten Random State wieder her.
    
    Args:
        state: Dictionary mit Random States von get_random_state()
        
    Example:
        >>> state = get_random_state()
        >>> # ... einige Operationen ...
        >>> restore_random_state(state)
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['torch_cuda'])