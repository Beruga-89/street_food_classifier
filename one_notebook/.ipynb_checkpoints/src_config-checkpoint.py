
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')
LOG_DIR = os.path.join(BASE_DIR, '..', 'logs')
PLOT_DIR = os.path.join(BASE_DIR, '..', 'plots')
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'outputs')
SEED = 42
BATCH_SIZE = 32
NUM_CLASSES = 20
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_STEP_SIZE = 7
LR_GAMMA = 0.1
EARLY_STOPPING_PATIENCE = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
HISTORY_PATH = os.path.join(OUTPUT_DIR, 'history.json')
METRICS_PLOT_PATH = os.path.join(PLOT_DIR, 'metrics.png')
CONFUSION_MATRIX_PATH = os.path.join(PLOT_DIR, 'confusion_matrix.png')
