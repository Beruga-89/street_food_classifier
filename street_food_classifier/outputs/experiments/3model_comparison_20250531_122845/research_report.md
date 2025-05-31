
# Street Food Classification - Model Comparison Report

**Experiment ID:** 3model_comparison_20250531_122845
**Date:** 2025-05-31 12:46:04

## Executive Summary

This report presents a systematic comparison of three state-of-the-art neural network architectures 
for street food image classification:


- **Best Overall Performance:** resnet18 (F1-Score: 0.8319)
- **Models Evaluated:** 1/3
- **Dataset:** Street Food Classification (20 classes)

## Detailed Results

| Model | Accuracy | F1-Score | Parameters | Training Time |
|-------|----------|----------|------------|---------------|
| resnet18 | 0.8340 | 0.8319 | 11.2M | 16.9min |


## Conclusions

resnet18 achieved the highest F1-score of 0.8319, making it the recommended 
architecture for street food classification tasks.

## Methodology

All models were trained using:
- Transfer learning with ImageNet pretrained weights
- Standard data augmentation
- Cross-entropy loss
- Adam optimizer
- Same training/validation split

This ensures fair comparison across architectures.
