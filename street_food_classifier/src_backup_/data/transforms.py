"""
Data transformation and augmentation utilities.

This module contains functions to create training and validation
transforms for image preprocessing and data augmentation.
"""

import torchvision.transforms as T
from typing import Tuple


def get_train_transforms(img_size: int = 224) -> T.Compose:
    """
    Erstellt Transformationen für das Training mit Data Augmentation.
    
    Args:
        img_size: Zielgröße für Bilder (default: 224 für ImageNet-kompatible Models)
        
    Returns:
        Compose-Objekt mit Training-Transformationen
        
    Example:
        >>> train_transform = get_train_transforms(224)
        >>> dataset = ImageFolder('data/', transform=train_transform)
    """
    return T.Compose([
        # Größe anpassen
        T.Resize((img_size, img_size)),
        
        # Data Augmentation für bessere Generalisierung
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=45),
        
        # Zu Tensor konvertieren
        T.ToTensor(),
        
        # Zusätzliche Farb-Augmentation
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2, 
            saturation=0.2,
            hue=0.05
        ),
        
        # ImageNet Normalisierung (für Transfer Learning)
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(img_size: int = 224) -> T.Compose:
    """
    Erstellt Transformationen für Validation/Test ohne Augmentation.
    
    Args:
        img_size: Zielgröße für Bilder (default: 224)
        
    Returns:
        Compose-Objekt mit Validation-Transformationen
        
    Example:
        >>> val_transform = get_val_transforms(224)
        >>> dataset = ImageFolder('data/', transform=val_transform)
    """
    return T.Compose([
        # Nur Größe anpassen, keine Augmentation
        T.Resize((img_size, img_size)),
        
        # Zu Tensor konvertieren
        T.ToTensor(),
        
        # ImageNet Normalisierung
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_transforms_pair(img_size: int = 224) -> Tuple[T.Compose, T.Compose]:
    """
    Convenience-Funktion um beide Transform-Sets zu erhalten.
    
    Args:
        img_size: Zielgröße für Bilder
        
    Returns:
        Tuple mit (train_transform, val_transform)
        
    Example:
        >>> train_transform, val_transform = get_transforms_pair(224)
    """
    return get_train_transforms(img_size), get_val_transforms(img_size)


def get_custom_train_transforms(
    img_size: int = 224,
    horizontal_flip_prob: float = 0.5,
    vertical_flip_prob: float = 0.5,
    rotation_degrees: float = 45,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.05
) -> T.Compose:
    """
    Erstellt anpassbare Training-Transformationen.
    
    Args:
        img_size: Bildgröße
        horizontal_flip_prob: Wahrscheinlichkeit für horizontales Spiegeln
        vertical_flip_prob: Wahrscheinlichkeit für vertikales Spiegeln
        rotation_degrees: Maximale Rotation in Grad
        brightness: Brightness-Jitter Faktor
        contrast: Contrast-Jitter Faktor
        saturation: Saturation-Jitter Faktor
        hue: Hue-Jitter Faktor
        
    Returns:
        Angepasste Training-Transformationen
        
    Example:
        >>> # Mildere Augmentation
        >>> transform = get_custom_train_transforms(
        ...     img_size=224,
        ...     horizontal_flip_prob=0.3,
        ...     rotation_degrees=15
        ... )
    """
    transforms = [
        T.Resize((img_size, img_size)),
    ]
    
    # Augmentations nur hinzufügen wenn Wahrscheinlichkeit > 0
    if horizontal_flip_prob > 0:
        transforms.append(T.RandomHorizontalFlip(p=horizontal_flip_prob))
    
    if vertical_flip_prob > 0:
        transforms.append(T.RandomVerticalFlip(p=vertical_flip_prob))
    
    if rotation_degrees > 0:
        transforms.append(T.RandomRotation(degrees=rotation_degrees))
    
    transforms.append(T.ToTensor())
    
    # ColorJitter nur wenn mindestens ein Parameter > 0
    if any([brightness, contrast, saturation, hue]):
        transforms.append(T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        ))
    
    # Normalisierung
    transforms.append(T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    
    return T.Compose(transforms)


def get_inference_transforms(img_size: int = 224) -> T.Compose:
    """
    Transformationen für Inferenz auf einzelnen Bildern.
    
    Identisch mit Validation-Transforms, aber explizit für Clarity.
    
    Args:
        img_size: Bildgröße
        
    Returns:
        Inferenz-Transformationen
        
    Example:
        >>> transform = get_inference_transforms()
        >>> prediction = model(transform(image).unsqueeze(0))
    """
    return get_val_transforms(img_size)