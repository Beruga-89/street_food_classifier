# Street Food Classifier

Ein Deep Learning Projekt zur Klassifikation von Street Food Bildern.

## 🎯 Projekt-Übersicht

- **Ziel:** Automatische Erkennung verschiedener Street Food Kategorien
- **Methode:** Transfer Learning mit PyTorch (ResNet, EfficientNet)
- **Framework:** Custom ML Pipeline mit Professional Visualizer

## 🏗️ Architektur

```
street_food_classifier/
├── src/                    # Source Code
├── data/                   # Datasets
├── models/                 # Gespeicherte Modelle
├── outputs/                # Ergebnisse & Plots
├── config/                 # Konfiguration
└── scripts/                # Training Scripts
```

## 🚀 Quick Start

```python
from ml_control_center import ml

# Status prüfen
ml.status()

# Model trainieren
history, exp_id = ml.train('resnet18', epochs=10)

# Model evaluieren
results = ml.evaluate()
```

## 📊 Experimente

Alle Experimente werden automatisch in `outputs/experiments/` dokumentiert.

## 🔧 Installation

```bash
pip install -r requirements.txt
python setup.py develop
```

## 📈 Ergebnisse

- **Beste Architektur:** TBD
- **Beste Accuracy:** TBD
- **Training Zeit:** TBD

## 📝 Masterarbeit

Teil der Masterarbeit über Physics Informed Neural Networks (PINNs).

---
**Author:** Oliver  
**Created:** 2025-05-31  
**Status:** In Development
