# direct_training_workflow.py 
# Direkter Workflow mit deinem bestehenden Control Center

import time
import json
from datetime import datetime
from pathlib import Path

# Import dein Control Center (funktioniert bereits!)
from ml_control_center import ml

def quick_model_comparison(architectures=None, epochs=5):
    """
    Schneller Vergleich verschiedener Architekturen.
    Funktioniert direkt mit deinem Control Center!
    """
    
    if architectures is None:
        architectures = ['resnet18', 'efficientnet_b0']
    
    print(f"🎯 QUICK MODEL COMPARISON")
    print(f"📋 Architectures: {', '.join(architectures)}")
    print(f"⏱️  Epochs per model: {epochs}")
    print("=" * 50)
    
    results = {}
    
    for i, arch in enumerate(architectures, 1):
        print(f"\n🚀 TRAINING {i}/{len(architectures)}: {arch.upper()}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Training mit deinem Control Center
            history, exp_id = ml.train(arch, epochs=epochs)
            
            if history:
                # Evaluation direkt danach
                eval_results = ml.evaluate(visualizations=True)
                
                training_time = time.time() - start_time
                
                results[arch] = {
                    'architecture': arch,
                    'experiment_id': exp_id,
                    'training_time_minutes': training_time / 60,
                    'evaluation': eval_results,
                    'status': 'success'
                }
                
                print(f"✅ {arch} RESULTS:")
                print(f"   Accuracy: {eval_results['accuracy']:.4f}")
                print(f"   F1-Score: {eval_results['f1']:.4f}")
                print(f"   Time: {training_time/60:.1f} min")
                
            else:
                print(f"❌ Training failed for {arch}")
                results[arch] = {'status': 'failed', 'error': 'Training returned None'}
                
        except Exception as e:
            print(f"❌ Error with {arch}: {e}")
            results[arch] = {'status': 'failed', 'error': str(e)}
    
    # Zusammenfassung
    print(f"\n📊 COMPARISON SUMMARY")
    print("=" * 40)
    
    successful = [r for r in results.values() if r.get('status') == 'success']
    
    if successful:
        # Sortiere nach F1-Score
        sorted_results = sorted(successful, 
                              key=lambda x: x['evaluation']['f1'], reverse=True)
        
        print(f"🏆 LEADERBOARD (by F1-Score):")
        for i, result in enumerate(sorted_results, 1):
            arch = result['architecture']
            f1 = result['evaluation']['f1']
            acc = result['evaluation']['accuracy']
            time_min = result['training_time_minutes']
            print(f"   {i}. {arch:15} - F1: {f1:.4f} | Acc: {acc:.4f} | Time: {time_min:.1f}m")
        
        # Speichere Ergebnisse
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"model_comparison_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return results
    else:
        print("❌ No successful trainings to compare")
        return results

def progressive_training_study():
    """
    Progressive Studie: Starte mit schnellen Tests, dann ausführlicher.
    """
    
    print(f"🎓 PROGRESSIVE TRAINING STUDY")
    print("=" * 50)
    
    # Phase 1: Quick Tests (3 Epochen)
    print(f"\n📋 PHASE 1: Quick Architecture Test (3 epochs)")
    quick_results = quick_model_comparison(['resnet18', 'efficientnet_b0'], epochs=3)
    
    successful_archs = [arch for arch, result in quick_results.items() 
                       if result.get('status') == 'success']
    
    if not successful_archs:
        print("❌ No successful quick tests. Check your setup.")
        return None
    
    # Phase 2: Best Architecture mit mehr Epochen
    best_arch = max(successful_archs, 
                   key=lambda arch: quick_results[arch]['evaluation']['f1'])
    
    print(f"\n📋 PHASE 2: Extended Training for Best Architecture")
    print(f"🏆 Best from Phase 1: {best_arch}")
    
    print(f"\n🚀 Training {best_arch} with 15 epochs...")
    history, exp_id = ml.train(best_arch, epochs=15)
    
    if history:
        final_results = ml.evaluate(visualizations=True)
        print(f"\n🎉 FINAL RESULTS ({best_arch}):")
        print(f"   Accuracy: {final_results['accuracy']:.4f}")
        print(f"   F1-Score: {final_results['f1']:.4f}")
        
        return {
            'quick_phase': quick_results,
            'best_architecture': best_arch,
            'final_results': final_results,
            'final_experiment_id': exp_id
        }
    else:
        print(f"❌ Extended training failed for {best_arch}")
        return {'quick_phase': quick_results, 'extended_training': 'failed'}

def hyperparameter_quick_test(architecture='resnet18'):
    """
    Schneller Hyperparameter Test mit deinem Control Center.
    """
    
    print(f"🔬 HYPERPARAMETER QUICK TEST: {architecture}")
    print("=" * 45)
    
    # Einfache Parameter-Kombinationen
    param_combinations = [
        {'learning_rate': 0.001, 'batch_size': 32},
        {'learning_rate': 0.01, 'batch_size': 32},
        {'learning_rate': 0.001, 'batch_size': 64}
    ]
    
    results = []
    
    for i, params in enumerate(param_combinations, 1):
        print(f"\n🧪 Test {i}/{len(param_combinations)}: {params}")
        
        try:
            # Training mit Parametern
            history, exp_id = ml.train(architecture, epochs=5, **params)
            
            if history:
                eval_results = ml.evaluate(visualizations=False)
                
                results.append({
                    'parameters': params,
                    'experiment_id': exp_id,
                    'f1_score': eval_results['f1'],
                    'accuracy': eval_results['accuracy']
                })
                
                print(f"   F1: {eval_results['f1']:.4f} | Acc: {eval_results['accuracy']:.4f}")
            else:
                print(f"   ❌ Training failed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if results:
        best_result = max(results, key=lambda x: x['f1_score'])
        print(f"\n🏆 BEST HYPERPARAMETERS:")
        print(f"   Parameters: {best_result['parameters']}")
        print(f"   F1-Score: {best_result['f1_score']:.4f}")
        
        return results
    else:
        print(f"\n❌ No successful hyperparameter tests")
        return []

def model_analysis_dashboard():
    """
    Zeigt Status und Analyse deiner trainierten Models.
    """
    
    print(f"📊 MODEL ANALYSIS DASHBOARD")
    print("=" * 50)
    
    # Control Center Status
    status = ml.status()
    
    print(f"\n💾 REGISTERED MODELS:")
    if hasattr(ml, 'model_registry') and ml.model_registry:
        for i, (model_name, info) in enumerate(ml.model_registry.items(), 1):
            arch = info.get('architecture', 'unknown')
            size_mb = info.get('size_mb', 0)
            last_used = info.get('last_used')
            
            status_icon = "🏆" if model_name == ml.get_best_model() else "📁"
            print(f"   {status_icon} {i}. {model_name}")
            print(f"      Architecture: {arch}")
            print(f"      Size: {size_mb:.1f} MB")
            print(f"      Last used: {last_used or 'Never'}")
    
    print(f"\n🔬 EXPERIMENT HISTORY:")
    if hasattr(ml, 'experiment_history') and ml.experiment_history:
        for exp_id, exp_info in ml.experiment_history.items():
            status = exp_info.get('status', 'unknown')
            arch = exp_info.get('architecture', 'unknown')
            epochs = exp_info.get('epochs', 0)
            
            status_icon = "✅" if status == 'completed' else "❌"
            print(f"   {status_icon} {exp_id}")
            print(f"      {arch} - {epochs} epochs - {status}")
    else:
        print("   No experiments recorded yet")
    
    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"   1. Run: quick_model_comparison() to compare architectures")
    print(f"   2. Run: progressive_training_study() for systematic study")
    print(f"   3. Run: hyperparameter_quick_test() to optimize parameters")

# === CONVENIENCE FUNCTIONS ===

def start_experiments():
    """Starte mit Experimenten - interaktiv."""
    
    print(f"🎯 ML EXPERIMENTS STARTER")
    print("=" * 40)
    print("Choose your approach:")
    print("1. Quick model comparison (5 minutes)")
    print("2. Progressive training study (15 minutes)")
    print("3. Hyperparameter test (10 minutes)")
    print("4. Just show model dashboard")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        return quick_model_comparison()
    elif choice == "2":
        return progressive_training_study()
    elif choice == "3":
        arch = input("Architecture (resnet18/efficientnet_b0): ").strip() or 'resnet18'
        return hyperparameter_quick_test(arch)
    elif choice == "4":
        return model_analysis_dashboard()
    else:
        print("Invalid choice. Running quick comparison...")
        return quick_model_comparison()

# === READY TO USE! ===

if __name__ == "__main__":
    print("""
🎯 DIRECT TRAINING WORKFLOW - READY!

=== QUICK COMMANDS ===
quick_model_comparison()                 # Compare 2-3 architectures
progressive_training_study()             # Systematic study
hyperparameter_quick_test('resnet18')    # Optimize parameters
model_analysis_dashboard()               # Show current status

=== INTERACTIVE ===
start_experiments()                      # Interactive menu

=== DIRECT ML CONTROL CENTER ===
ml.train('resnet18', epochs=10)          # Direct training
ml.evaluate()                           # Direct evaluation
ml.status()                             # Show status
    """)