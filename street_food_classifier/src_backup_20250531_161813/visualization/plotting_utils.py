"""
Plotting utilities and helper functions.

This module provides utility functions for consistent plotting
across the entire project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

from ..utils import ensure_dir, get_logger


def set_plot_style(style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6),
                   dpi: int = 300) -> None:
    """
    Setzt konsistenten Plot-Style für das gesamte Projekt.
    
    Args:
        style: Matplotlib Style ('seaborn-v0_8', 'ggplot', 'classic', etc.)
        figsize: Standard Figure-Größe
        dpi: DPI für hochauflösende Plots
        
    Example:
        >>> set_plot_style('seaborn-v0_8', figsize=(12, 8))
        >>> # Alle nachfolgenden Plots verwenden diesen Style
    """
    try:
        plt.style.use(style)
    except OSError:
        # Fallback falls Style nicht verfügbar
        plt.style.use('default')
        
    # Global settings
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    # Bessere Schriftarten falls verfügbar
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    except:
        pass


def save_figure(fig: plt.Figure, filepath: str, bbox_inches: str = 'tight',
                facecolor: str = 'white', transparent: bool = False,
                create_dirs: bool = True) -> str:
    """
    Speichert Figure mit konsistenten Einstellungen.
    
    Args:
        fig: Matplotlib Figure
        filepath: Pfad zur Ausgabedatei
        bbox_inches: Bounding box Einstellung
        facecolor: Hintergrundfarbe
        transparent: Ob transparenter Hintergrund verwendet werden soll
        create_dirs: Ob Verzeichnisse erstellt werden sollen
        
    Returns:
        Vollständiger Pfad zur gespeicherten Datei
        
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [4, 5, 6])
        >>> save_figure(fig, "outputs/plots/my_plot.png")
    """
    filepath = Path(filepath)
    
    if create_dirs:
        ensure_dir(filepath.parent)
    
    fig.savefig(
        filepath,
        bbox_inches=bbox_inches,
        facecolor=facecolor,
        transparent=transparent,
        dpi=plt.rcParams['savefig.dpi']
    )
    
    logger = get_logger(__name__)
    logger.info(f"Figure saved: {filepath}")
    
    return str(filepath)


def create_subplot_grid(nrows: int, ncols: int, figsize: Optional[Tuple[int, int]] = None,
                       sharex: bool = False, sharey: bool = False,
                       subplot_kw: Optional[Dict] = None) -> Tuple[plt.Figure, np.ndarray]:
    """
    Erstellt ein Grid von Subplots mit konsistenten Einstellungen.
    
    Args:
        nrows: Anzahl Zeilen
        ncols: Anzahl Spalten
        figsize: Figure-Größe (auto-berechnet falls None)
        sharex: Ob X-Achsen geteilt werden sollen
        sharey: Ob Y-Achsen geteilt werden sollen
        subplot_kw: Zusätzliche Subplot-Parameter
        
    Returns:
        Tuple mit (Figure, Axes-Array)
        
    Example:
        >>> fig, axes = create_subplot_grid(2, 3, figsize=(15, 10))
        >>> axes[0, 0].plot([1, 2, 3])
        >>> axes[0, 1].bar(['A', 'B'], [1, 2])
    """
    if figsize is None:
        # Auto-berechne Figure-Größe basierend auf Grid
        width = ncols * 5
        height = nrows * 4
        figsize = (width, height)
    
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        subplot_kw=subplot_kw or {}
    )
    
    # Sicherstellen dass axes immer ein Array ist
    if nrows * ncols == 1:
        axes = np.array([axes])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(-1)
    
    plt.tight_layout()
    return fig, axes


def add_value_labels(ax: plt.Axes, bars, format_str: str = '{:.3f}',
                    offset: float = 0.01, fontsize: int = 9) -> None:
    """
    Fügt Wert-Labels zu Balken hinzu.
    
    Args:
        ax: Matplotlib Axes
        bars: Bar-Container von ax.bar()
        format_str: Format-String für Werte
        offset: Offset für Label-Position
        fontsize: Schriftgröße der Labels
        
    Example:
        >>> bars = ax.bar(['A', 'B', 'C'], [1, 2, 3])
        >>> add_value_labels(ax, bars)
    """
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + height * offset,
            format_str.format(height),
            ha='center', va='bottom',
            fontsize=fontsize
        )


def create_confusion_matrix_plot(cm: np.ndarray, class_names: list,
                               normalize: bool = False, cmap: str = 'Blues',
                               title: str = 'Confusion Matrix') -> Tuple[plt.Figure, plt.Axes]:
    """
    Erstellt einen Confusion Matrix Plot.
    
    Args:
        cm: Confusion Matrix als numpy array
        class_names: Namen der Klassen
        normalize: Ob Matrix normalisiert werden soll
        cmap: Colormap
        title: Plot-Titel
        
    Returns:
        Tuple mit (Figure, Axes)
        
    Example:
        >>> from sklearn.metrics import confusion_matrix
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> fig, ax = create_confusion_matrix_plot(cm, class_names)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Achsen-Labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotiere x-Labels falls nötig
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Text-Annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    return fig, ax


def plot_training_curves(history: Dict, metrics: list = ['loss', 'accuracy'],
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plottet Training-Kurven.
    
    Args:
        history: Training History Dictionary
        metrics: Liste der zu plottenden Metriken
        save_path: Pfad zum Speichern (optional)
        
    Returns:
        Matplotlib Figure
        
    Example:
        >>> fig = plot_training_curves(history, ['loss', 'accuracy', 'f1'])
    """
    n_metrics = len(metrics)
    fig, axes = create_subplot_grid(1, n_metrics, figsize=(n_metrics * 5, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    epochs = range(1, len(history['train'][metrics[0]]) + 1)
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        train_values = history['train'][metric]
        val_values = history['val'][metric]
        
        ax.plot(epochs, train_values, 'bo-', label=f'Train {metric.title()}', linewidth=2)
        ax.plot(epochs, val_values, 'ro-', label=f'Val {metric.title()}', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.title())
        ax.set_title(f'Training and Validation {metric.title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def create_comparison_bar_plot(data: Dict[str, float], title: str = "Model Comparison",
                              ylabel: str = "Score", xlabel: str = "Models",
                              color_map: Optional[Dict] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Erstellt einen Balkenplot für Model-Vergleiche.
    
    Args:
        data: Dictionary mit {model_name: score}
        title: Plot-Titel
        ylabel: Y-Achsen-Label
        xlabel: X-Achsen-Label
        color_map: Dictionary mit {model_name: color}
        
    Returns:
        Tuple mit (Figure, Axes)
        
    Example:
        >>> data = {"ResNet18": 0.95, "ResNet50": 0.97, "EfficientNet": 0.96}
        >>> fig, ax = create_comparison_bar_plot(data, "Accuracy Comparison")
    """
    fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.2), 6))
    
    models = list(data.keys())
    scores = list(data.values())
    
    # Farben bestimmen
    if color_map:
        colors = [color_map.get(model, 'skyblue') for model in models]
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    bars = ax.bar(models, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Value labels hinzufügen
    add_value_labels(ax, bars)
    
    # Styling
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # X-Labels rotieren falls nötig
    if len(max(models, key=len)) > 8:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    return fig, ax


def create_metric_dashboard(metrics_data: Dict, model_name: str = "Model",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Erstellt ein Dashboard mit mehreren Metriken.
    
    Args:
        metrics_data: Dictionary mit Metrik-Namen und -Werten
        model_name: Name des Models für den Titel
        save_path: Pfad zum Speichern (optional)
        
    Returns:
        Matplotlib Figure
        
    Example:
        >>> metrics = {
        ...     'accuracy': {'train': 0.95, 'val': 0.93},
        ...     'f1_score': {'train': 0.94, 'val': 0.92},
        ...     'loss': {'train': 0.15, 'val': 0.18}
        ... }
        >>> fig = create_metric_dashboard(metrics, "ResNet18")
    """
    n_metrics = len(metrics_data)
    fig, axes = create_subplot_grid(1, n_metrics, figsize=(n_metrics * 4, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        ax = axes[idx]
        
        if isinstance(values, dict):
            # Train/Val Vergleich
            categories = list(values.keys())
            scores = list(values.values())
            
            colors = ['skyblue' if cat == 'train' else 'lightcoral' for cat in categories]
            bars = ax.bar(categories, scores, color=colors, alpha=0.7)
            add_value_labels(ax, bars)
            
        else:
            # Einzelner Wert
            bars = ax.bar([metric_name], [values], color='lightgreen', alpha=0.7)
            add_value_labels(ax, bars)
        
        ax.set_title(metric_name.replace('_', ' ').title(), fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f"{model_name} - Performance Dashboard", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_class_distribution(class_counts: Dict[str, int], title: str = "Class Distribution",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plottet Klassenverteilung als Balkendiagramm.
    
    Args:
        class_counts: Dictionary mit {class_name: count}
        title: Plot-Titel
        save_path: Pfad zum Speichern (optional)
        
    Returns:
        Matplotlib Figure
        
    Example:
        >>> counts = {"Cat": 1000, "Dog": 800, "Bird": 600}
        >>> fig = plot_class_distribution(counts)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Balkendiagramm
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    bars = ax1.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')
    add_value_labels(ax1, bars, format_str='{:.0f}')
    
    ax1.set_title("Class Distribution", fontweight='bold')
    ax1.set_ylabel("Number of Samples")
    ax1.grid(True, alpha=0.3, axis='y')
    
    # X-Labels rotieren
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    
    # Pie Chart
    ax2.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title("Class Distribution (%)", fontweight='bold')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def create_learning_curve_plot(train_sizes: np.ndarray, train_scores: np.ndarray,
                              val_scores: np.ndarray, metric_name: str = "Accuracy",
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Erstellt Learning Curve Plot.
    
    Args:
        train_sizes: Array mit Training Set Größen
        train_scores: Training Scores für jede Größe
        val_scores: Validation Scores für jede Größe
        metric_name: Name der Metrik
        save_path: Pfad zum Speichern (optional)
        
    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Mean und Std berechnen falls 2D Arrays
    if train_scores.ndim > 1:
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
    else:
        train_mean = train_scores
        train_std = np.zeros_like(train_scores)
        val_mean = val_scores
        val_std = np.zeros_like(val_scores)
    
    # Training Curve
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label=f'Training {metric_name}')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                    alpha=0.1, color='blue')
    
    # Validation Curve
    ax.plot(train_sizes, val_mean, 'o-', color='red', label=f'Validation {metric_name}')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color='red')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Learning Curve - {metric_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def add_timestamp_watermark(ax: plt.Axes, position: str = 'bottom_right',
                           fontsize: int = 8, alpha: float = 0.5) -> None:
    """
    Fügt Zeitstempel-Wasserzeichen zu Plot hinzu.
    
    Args:
        ax: Matplotlib Axes
        position: Position ('bottom_right', 'top_left', etc.)
        fontsize: Schriftgröße
        alpha: Transparenz
        
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> add_timestamp_watermark(ax)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    position_map = {
        'bottom_right': (0.99, 0.01, 'right', 'bottom'),
        'bottom_left': (0.01, 0.01, 'left', 'bottom'),
        'top_right': (0.99, 0.99, 'right', 'top'),
        'top_left': (0.01, 0.99, 'left', 'top')
    }
    
    if position in position_map:
        x, y, ha, va = position_map[position]
        ax.text(x, y, timestamp, transform=ax.transAxes,
                fontsize=fontsize, alpha=alpha, ha=ha, va=va,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))


def create_model_architecture_plot(layer_info: list, save_path: Optional[str] = None) -> plt.Figure:
    """
    Erstellt Visualisierung der Model-Architektur.
    
    Args:
        layer_info: Liste mit Layer-Informationen
        save_path: Pfad zum Speichern (optional)
        
    Returns:
        Matplotlib Figure
        
    Example:
        >>> layers = [
        ...     {'name': 'Conv2d', 'params': 1728, 'output_shape': (64, 224, 224)},
        ...     {'name': 'BatchNorm2d', 'params': 128, 'output_shape': (64, 224, 224)},
        ...     {'name': 'ReLU', 'params': 0, 'output_shape': (64, 224, 224)}
        ... ]
        >>> fig = create_model_architecture_plot(layers)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Parameter Count per Layer
    layer_names = [layer['name'] for layer in layer_info]
    param_counts = [layer['params'] for layer in layer_info]
    
    y_pos = np.arange(len(layer_names))
    bars = ax1.barh(y_pos, param_counts, alpha=0.7, color='skyblue')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(layer_names)
    ax1.set_xlabel('Number of Parameters')
    ax1.set_title('Parameters per Layer')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add parameter labels
    for i, (bar, count) in enumerate(zip(bars, param_counts)):
        if count > 0:
            ax1.text(bar.get_width() + max(param_counts) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{count:,}', ha='left', va='center', fontsize=8)
    
    # Cumulative Parameters
    cumulative_params = np.cumsum(param_counts)
    ax2.plot(range(len(layer_names)), cumulative_params, 'o-', linewidth=2, markersize=6)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Cumulative Parameters')
    ax2.set_title('Cumulative Parameter Growth')
    ax2.grid(True, alpha=0.3)
    
    # Format y-axis with thousands separator
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig