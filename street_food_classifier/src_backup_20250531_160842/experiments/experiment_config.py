"""
Configuration for experiments.
"""

class ExperimentConfig:
    """Configuration class for experiments."""
    
    # Default experiment parameters
    DEFAULT_EPOCHS = {
        'quick': 5,
        'standard': 15, 
        'full': 20,
        'research': 30
    }
    
    # Models to compare
    DEFAULT_MODELS = ['resnet18', 'resnet50', 'efficientnet_b0']
    
    # Extended model list for advanced experiments
    EXTENDED_MODELS = ['resnet18', 'resnet50', 'efficientnet_b0', 'custom_cnn', 'mobilenet_v2']
    
    # Experiment output directory
    OUTPUT_DIR = "outputs/experiments"
    
    # Visualization settings
    SAVE_VISUALIZATIONS = True
    SAVE_DETAILED_RESULTS = True
    
    # Performance thresholds for analysis
    GOOD_ACCURACY_THRESHOLD = 0.85
    EXCELLENT_ACCURACY_THRESHOLD = 0.90
    
    # Efficiency metrics weights
    EFFICIENCY_WEIGHTS = {
        'accuracy_weight': 0.4,
        'speed_weight': 0.3,
        'size_weight': 0.3
    }
