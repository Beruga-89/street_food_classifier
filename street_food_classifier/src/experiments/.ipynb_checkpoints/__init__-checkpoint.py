"""
Experiments package for systematic model comparison and research.

This package contains all experiment workflows for the Street Food Classifier,
designed for systematic research and PINN masterarbeit documentation.
"""

from .model_comparison import (
    run_comprehensive_experiment,
    quick_experiment, 
    full_experiment,
    research_experiment
)

from .experiment_analysis import (
    create_detailed_comparison_table,
    analyze_best_models,
    analyze_efficiency,
    save_experiment_results
)

from .experiment_config import ExperimentConfig

__all__ = [
    'run_comprehensive_experiment',
    'quick_experiment',
    'full_experiment', 
    'research_experiment',
    'create_detailed_comparison_table',
    'analyze_best_models',
    'analyze_efficiency',
    'save_experiment_results',
    'ExperimentConfig'
]