"""
Model comparison experiments for systematic architecture evaluation.

This module contains the main experiment workflows for comparing
different neural network architectures on the Street Food dataset.
"""

import time
from datetime import datetime
from .experiment_analysis import (
    create_detailed_comparison_table, 
    analyze_best_models, 
    analyze_efficiency,
    save_experiment_results,
    create_research_report
)
from .experiment_config import ExperimentConfig

def run_comprehensive_experiment(epochs=15, save_detailed_results=True, models=None):
    """
    Führt umfassendes Model Experiment durch.
    
    Args:
        epochs: Epochen pro Model (empfohlen: 15-20 für aussagekräftige Ergebnisse)
        save_detailed_results: Ob detaillierte Ergebnisse gespeichert werden sollen
        models: Liste der zu testenden Models (default: ResNet18, ResNet50, EfficientNet-B0)
    """
    
    # Import hier um zirkuläre Imports zu vermeiden
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd() / 'src'))
    
    models_to_test = models or ExperimentConfig.DEFAULT_MODELS
    
    print("🔬 COMPREHENSIVE MODEL EXPERIMENT")
    print("=" * 60)
    print(f"Models: {', '.join(models_to_test)}")
    print(f"Epochen pro Model: {epochs}")
    print(f"Geschätzte Gesamtzeit: {epochs * len(models_to_test) * 1.5:.0f}-{epochs * len(models_to_test) * 3:.0f} Minuten")
    print("=" * 60)
    
    # Experiment Setup
    experiment_start = time.time()
    experiment_id = f"{len(models_to_test)}model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    results = {}
    detailed_results = {}
    
    print(f"🚀 EXPERIMENT ID: {experiment_id}")
    print(f"⏰ Start: {datetime.now().strftime('%H:%M:%S')}")
    
    # Import ML Control Center
    try:
        # Globales ml object sollte verfügbar sein
        global ml
        if 'ml' not in globals():
            from .. import StreetFoodClassifier
            from config import Config
            
            # Erstelle temporäres Control Center
            class TempControlCenter:
                def __init__(self):
                    self.config = Config()
                    self.classifier = None
                
                def _setup_classifier(self):
                    if self.classifier is None:
                        self.classifier = StreetFoodClassifier(self.config)
                
                def train(self, architecture, epochs):
                    self._setup_classifier()
                    original_epochs = self.config.EPOCHS
                    self.config.EPOCHS = epochs
                    
                    try:
                        history = self.classifier.train(architecture, pretrained=True, save_results=True)
                        exp_id = f"{architecture}_{epochs}ep_{int(time.time())}"
                        return history, exp_id
                    finally:
                        self.config.EPOCHS = original_epochs
                
                def evaluate(self, visualizations=True):
                    return self.classifier.evaluate(create_visualizations=visualizations)
            
            ml = TempControlCenter()
            
    except Exception as e:
        print(f"❌ Setup Fehler: {e}")
        return None
    
    for i, model_name in enumerate(models_to_test, 1):
        print(f"\n{'='*20} MODEL {i}/{len(models_to_test)}: {model_name.upper()} {'='*20}")
        
        model_start = time.time()
        
        try:
            # 1. Training
            print(f"🚀 Training {model_name} für {epochs} Epochen...")
            history, exp_id = ml.train(model_name, epochs=epochs)
            
            if not history:
                print(f"❌ Training für {model_name} fehlgeschlagen!")
                results[model_name] = None
                continue
            
            training_time = time.time() - model_start
            
            # 2. Evaluation
            print(f"📊 Evaluiere {model_name}...")
            eval_results = ml.evaluate(visualizations=save_detailed_results)
            
            if not eval_results:
                print(f"❌ Evaluation für {model_name} fehlgeschlagen!")
                results[model_name] = None
                continue
            
            # 3. Model Info sammeln
            model_info = ml.classifier.get_model_summary()
            
            # 4. Ergebnisse sammeln
            results[model_name] = {
                'accuracy': eval_results['accuracy'],
                'f1_score': eval_results['f1'],
                'loss': eval_results['loss'],
                'training_time_minutes': training_time / 60,
                'parameters': model_info['total_params'],
                'model_size_mb': model_info['model_size_mb'],
                'experiment_id': exp_id
            }
            
            # 5. Detaillierte Ergebnisse (optional)
            if save_detailed_results and hasattr(ml.classifier, 'get_training_summary'):
                training_summary = ml.classifier.get_training_summary()
                detailed_results[model_name] = {
                    'training_history': history,
                    'training_summary': training_summary,
                    'evaluation_results': eval_results,
                    'model_summary': model_info
                }
            
            # 6. Zwischenergebnisse anzeigen
            print(f"✅ {model_name.upper()} ABGESCHLOSSEN:")
            print(f"   🎯 Accuracy: {eval_results['accuracy']:.4f}")
            print(f"   📊 F1-Score: {eval_results['f1']:.4f}")
            print(f"   ⏱️  Zeit: {training_time/60:.1f} min")
            print(f"   💾 Parameter: {model_info['total_params']:,}")
            
            # Memory cleanup
            if hasattr(ml.classifier, 'cleanup'):
                ml.classifier.cleanup()
            
        except Exception as e:
            print(f"❌ FEHLER bei {model_name}: {e}")
            results[model_name] = None
    
    total_time = time.time() - experiment_start
    
    # === COMPREHENSIVE ANALYSIS ===
    print(f"\n{'='*20} EXPERIMENT ABGESCHLOSSEN {'='*20}")
    print(f"⏰ Gesamtzeit: {total_time/60:.1f} Minuten")
    print(f"📊 Ergebnisse für {len([r for r in results.values() if r is not None])}/{len(models_to_test)} Models")
    
    # Detaillierte Analysen
    create_detailed_comparison_table(results)
    analyze_best_models(results)
    analyze_efficiency(results)
    
    # Speichere Experiment-Ergebnisse
    if save_detailed_results:
        save_experiment_results(experiment_id, results, detailed_results)
        create_research_report(experiment_id, results)
    
    return {
        'experiment_id': experiment_id,
        'results': results,
        'detailed_results': detailed_results if save_detailed_results else None,
        'total_time': total_time
    }

def quick_experiment(epochs=5):
    """Schnelles Experiment für Tests."""
    print("⚡ QUICK MODEL EXPERIMENT (5 Epochen)")
    return run_comprehensive_experiment(epochs=epochs, save_detailed_results=False)

def full_experiment(epochs=20):
    """Vollständiges Experiment für finale Ergebnisse."""
    print("🔬 FULL MODEL EXPERIMENT (20 Epochen)")
    return run_comprehensive_experiment(epochs=epochs, save_detailed_results=True)

def research_experiment(epochs=30):
    """Research-Level Experiment für Masterarbeit."""
    print("🎓 RESEARCH MODEL EXPERIMENT (30 Epochen)")
    return run_comprehensive_experiment(epochs=epochs, save_detailed_results=True)