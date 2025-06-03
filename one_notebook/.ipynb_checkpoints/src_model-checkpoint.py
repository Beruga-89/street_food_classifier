
from torchvision import models
import torch.nn as nn
from src.config import NUM_CLASSES
def get_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model
