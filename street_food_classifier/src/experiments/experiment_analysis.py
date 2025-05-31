"""
Analysis utilities for experiment results.

This module contains functions for analyzing and visualizing
experiment results, creating comparison tables, and generating
research-quality reports.
"""

import json
from datetime import datetime
from pathlib import Path

def create_detailed_comparison_table(results):
    """Erstellt detaillierte Vergleichstabelle."""
    
    print(f"\n📊 DETAILLIERTE VERGLEICHSTABELLE:")
    print("=" * 85)
    print(f"{'Model':<15} {'Accuracy':<10} {'F1-Score':<10} {'Loss':<8} {'Time':<8} {'Params':<10} {'Size':<8}")
    print("-" * 85)
    
    # Sortiere nach F1-Score (beste Metrik für Multi-Class)
    valid_results = {k: v for k, v in results.items() if v is not None}
    sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    
    for model, data in sorted_models:
        print(f"{model:<15} {data['accuracy']:<10.4f} {data['f1_score']:<10.4f} "
              f"{data['loss']:<8.3f} {data['training_time_minutes']:<8.1f}min "
              f"{data['parameters']/1e6:<10.1f}M {data['model_size_mb']:<8.1f}MB")
    
    print("-" * 85)
    
    if sorted_models:
        winner = sorted_models[0]
        print(f"🏆 OVERALL WINNER: {winner[0].upper()}")
        print(f"   📊 F1-Score: {winner[1]['f1_score']:.4f}")
        print(f"   🎯 Accuracy: {winner[1]['accuracy']:.4f}")

def analyze_best_models(results):
    """Analysiert beste Models in verschiedenen Kategorien."""
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        return
    
    print(f"\n🏆 KATEGORIE-GEWINNER:")
    print("-" * 40)
    
    # Beste Accuracy
    best_acc = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"🎯 Beste Accuracy: {best_acc[0]} ({best_acc[1]['accuracy']:.4f})")
    
    # Beste F1-Score
    best_f1 = max(valid_results.items(), key=lambda x: x[1]['f1_score'])
    print(f"📊 Beste F1-Score: {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
    
    # Schnellstes Training
    fastest = min(valid_results.items(), key=lambda x: x[1]['training_time_minutes'])
    print(f"⚡ Schnellstes Training: {fastest[0]} ({fastest[1]['training_time_minutes']:.1f} min)")
    
    # Kleinste Model-Größe
    smallest = min(valid_results.items(), key=lambda x: x[1]['model_size_mb'])
    print(f"💾 Kleinstes Model: {smallest[0]} ({smallest[1]['model_size_mb']:.1f} MB)")
    
    # Wenigste Parameter
    least_params = min(valid_results.items(), key=lambda x: x[1]['parameters'])
    print(f"🔢 Wenigste Parameter: {least_params[0]} ({least_params[1]['parameters']/1e6:.1f}M)")

def analyze_efficiency(results):
    """Analysiert Effizienz-Metriken."""
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        return
    
    print(f"\n⚖️  EFFIZIENZ-ANALYSE:")
    print("-" * 50)
    
    for model, data in valid_results.items():
        # Accuracy per Parameter (höher = besser)
        acc_per_param = data['accuracy'] / (data['parameters'] / 1e6)
        
        # Accuracy per Training Time (höher = besser)
        acc_per_time = data['accuracy'] / data['training_time_minutes']
        
        # F1 per MB (höher = besser)
        f1_per_mb = data['f1_score'] / data['model_size_mb']
        
        print(f"{model:<15}:")
        print(f"  📊 Acc/Million Params: {acc_per_param:.3f}")
        print(f"  ⏱️  Acc/Training Minute: {acc_per_time:.3f}")
        print(f"  💾 F1/Model MB: {f1_per_mb:.3f}")

def save_experiment_results(experiment_id, results, detailed_results):
    """Speichert Experiment-Ergebnisse für später."""
    
    # Erstelle Experiment-Verzeichnis
    exp_dir = Path(f"outputs/experiments/{experiment_id}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary speichern
    summary = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'models_tested': list(results.keys()),
        'successful_models': [k for k, v in results.items() if v is not None]
    }
    
    with open(exp_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Detaillierte Ergebnisse speichern
    if detailed_results:
        with open(exp_dir / 'detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\n💾 EXPERIMENT GESPEICHERT:")
    print(f"   📁 {exp_dir}")
    print(f"   📄 Summary: experiment_summary.json")
    if detailed_results:
        print(f"   📄 Details: detailed_results.json")

def create_research_report(experiment_id, results):
    """Erstellt Research-Quality Report für Masterarbeit."""
    
    exp_dir = Path(f"outputs/experiments/{experiment_id}")
    
    report = f"""
# Street Food Classification - Model Comparison Report

**Experiment ID:** {experiment_id}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a systematic comparison of three state-of-the-art neural network architectures 
for street food image classification:

"""
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if valid_results:
        # Best model
        best_model = max(valid_results.items(), key=lambda x: x[1]['f1_score'])
        
        report += f"""
- **Best Overall Performance:** {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})
- **Models Evaluated:** {len(valid_results)}/3
- **Dataset:** Street Food Classification (20 classes)

## Detailed Results

| Model | Accuracy | F1-Score | Parameters | Training Time |
|-------|----------|----------|------------|---------------|
"""
        
        for model, data in sorted(valid_results.items(), key=lambda x: x[1]['f1_score'], reverse=True):
            report += f"| {model} | {data['accuracy']:.4f} | {data['f1_score']:.4f} | {data['parameters']/1e6:.1f}M | {data['training_time_minutes']:.1f}min |\n"
        
        report += f"""

## Conclusions

{best_model[0]} achieved the highest F1-score of {best_model[1]['f1_score']:.4f}, making it the recommended 
architecture for street food classification tasks.

## Methodology

All models were trained using:
- Transfer learning with ImageNet pretrained weights
- Standard data augmentation
- Cross-entropy loss
- Adam optimizer
- Same training/validation split

This ensures fair comparison across architectures.
"""
    
    # Speichere Report
    with open(exp_dir / 'research_report.md', 'w') as f:
        f.write(report)
    
    print(f"📄 Research Report erstellt: {exp_dir / 'research_report.md'}")