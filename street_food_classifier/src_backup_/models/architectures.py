"""
Neural network architectures for the Street Food Classifier.

This module contains functions to create different model architectures
including pre-trained models and custom architectures.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


def create_resnet18(num_classes: int, pretrained: bool = True,
                   freeze_backbone: bool = False) -> nn.Module:
    """
    Erstellt ein ResNet-18 Model für Klassifikation.
    
    Args:
        num_classes: Anzahl der Ausgabe-Klassen
        pretrained: Ob vortrainierte ImageNet Gewichte verwendet werden sollen
        freeze_backbone: Ob das Backbone eingefroren werden soll (nur Classifier trainieren)
        
    Returns:
        ResNet-18 Model
        
    Example:
        >>> model = create_resnet18(num_classes=10, pretrained=True)
        >>> model = model.to(device)
    """
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)
    
    # Backbone einfrieren falls gewünscht
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Finale Klassifikations-Schicht anpassen
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # Classifier Layer ist immer trainierbar
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def create_resnet50(num_classes: int, pretrained: bool = True,
                   freeze_backbone: bool = False) -> nn.Module:
    """
    Erstellt ein ResNet-50 Model für Klassifikation.
    
    Args:
        num_classes: Anzahl der Ausgabe-Klassen
        pretrained: Ob vortrainierte ImageNet Gewichte verwendet werden sollen
        freeze_backbone: Ob das Backbone eingefroren werden soll
        
    Returns:
        ResNet-50 Model
    """
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=None)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def create_efficientnet_b0(num_classes: int, pretrained: bool = True,
                          freeze_backbone: bool = False) -> nn.Module:
    """
    Erstellt ein EfficientNet-B0 Model für Klassifikation.
    
    Args:
        num_classes: Anzahl der Ausgabe-Klassen
        pretrained: Ob vortrainierte ImageNet Gewichte verwendet werden sollen
        freeze_backbone: Ob das Backbone eingefroren werden soll
        
    Returns:
        EfficientNet-B0 Model
    """
    if pretrained:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    else:
        model = models.efficientnet_b0(weights=None)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # EfficientNet hat 'classifier' statt 'fc'
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model


def create_custom_cnn(num_classes: int, input_channels: int = 3,
                     dropout_rate: float = 0.5) -> nn.Module:
    """
    Erstellt ein custom CNN von Grund auf.
    
    Args:
        num_classes: Anzahl der Ausgabe-Klassen
        input_channels: Anzahl Input-Kanäle (3 für RGB)
        dropout_rate: Dropout Rate für Regularisierung
        
    Returns:
        Custom CNN Model
        
    Example:
        >>> model = create_custom_cnn(num_classes=10)
        >>> # Für Grayscale-Bilder:
        >>> model = create_custom_cnn(num_classes=10, input_channels=1)
    """
    
    class CustomCNN(nn.Module):
        def __init__(self, num_classes: int, input_channels: int, dropout_rate: float):
            super(CustomCNN, self).__init__()
            
            # Convolutional Layers
            self.conv_layers = nn.Sequential(
                # Block 1
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
                
                # Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
                
                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
                
                # Block 4
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
            )
            
            # Adaptive Average Pooling (macht Output-Größe unabhängig von Input-Größe)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(256 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            x = self.conv_layers(x)
            x = self.adaptive_pool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    return CustomCNN(num_classes, input_channels, dropout_rate)


def create_mobilenet_v2(num_classes: int, pretrained: bool = True,
                       freeze_backbone: bool = False) -> nn.Module:
    """
    Erstellt ein MobileNet-V2 Model (leichtgewichtig für mobile Anwendungen).
    
    Args:
        num_classes: Anzahl der Ausgabe-Klassen
        pretrained: Ob vortrainierte ImageNet Gewichte verwendet werden sollen
        freeze_backbone: Ob das Backbone eingefroren werden soll
        
    Returns:
        MobileNet-V2 Model
    """
    if pretrained:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    else:
        model = models.mobilenet_v2(weights=None)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # MobileNet hat 'classifier' 
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model


def create_vision_transformer(num_classes: int, pretrained: bool = True,
                             freeze_backbone: bool = False) -> nn.Module:
    """
    Erstellt ein Vision Transformer (ViT) Model.
    
    Args:
        num_classes: Anzahl der Ausgabe-Klassen
        pretrained: Ob vortrainierte ImageNet Gewichte verwendet werden sollen
        freeze_backbone: Ob das Backbone eingefroren werden soll
        
    Returns:
        Vision Transformer Model
        
    Note:
        Erfordert PyTorch >= 1.13 und torchvision >= 0.14
    """
    try:
        if pretrained:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        else:
            model = models.vit_b_16(weights=None)
        
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        # ViT hat 'heads'
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        
        for param in model.heads.parameters():
            param.requires_grad = True
        
        return model
        
    except AttributeError:
        raise RuntimeError("Vision Transformer requires PyTorch >= 1.13 and torchvision >= 0.14")


def get_model_info(model: nn.Module) -> dict:
    """
    Gibt Informationen über ein Model zurück.
    
    Args:
        model: PyTorch Model
        
    Returns:
        Dictionary mit Model-Informationen
        
    Example:
        >>> model = create_resnet18(num_classes=10)
        >>> info = get_model_info(model)
        >>> print(f"Parameters: {info['total_params']:,}")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'model_name': model.__class__.__name__,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }


def apply_weight_init(model: nn.Module, init_type: str = 'xavier') -> None:
    """
    Wendet Gewichts-Initialisierung auf ein Model an.
    
    Args:
        model: PyTorch Model
        init_type: Typ der Initialisierung ('xavier', 'kaiming', 'normal')
        
    Example:
        >>> model = create_custom_cnn(num_classes=10)
        >>> apply_weight_init(model, 'kaiming')
    """
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0, 0.02)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.Linear):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0, 0.02)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)