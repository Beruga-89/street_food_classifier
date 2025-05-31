"""
Visualization package for Street Food Classifier.
"""

from .visualizer import Visualizer

# plotting_utils falls vorhanden
try:
    from .plotting_utils import save_figure, create_subplot_grid, set_plot_style
    plotting_available = True
except ImportError:
    # Fallback falls plotting.py statt plotting_utils.py existiert
    plotting_available = False

if plotting_available:
    __all__ = [
        'Visualizer',
        'save_figure',
        'create_subplot_grid', 
        'set_plot_style'
    ]
else:
    __all__ = ['Visualizer']
