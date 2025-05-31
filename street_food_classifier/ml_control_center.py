# ml_control_center.py
# Speichere diese Datei im Hauptverzeichnis (neben config.py)

import time
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any

# Imports für Control Center
try:
    from src import StreetFoodClassifier
    from config import Config
    print("✅ Control Center Imports erfolgreich")
except ImportError as e:
    print(f"❌ Import Fehler: {e}")
    print("💡 Stelle sicher dass du im richtigen Verzeichnis bist")
    raise

class MLControlCenter:
    """
    Zentrales Control Center für alle ML-Operations.
    
    Kombiniert:
    - Smart Model Loading mit Architektur-Erkennung
    - Quick Training Functions  
    - Intelligent Model Management
    - Dashboard Integration
    - Experiment Workflows
    """
    
    def __init__(self):
        self.config = Config()
        self.classifier = None
        self.model_registry = {}  # Speichert Model-Informationen
        self.experiment_history = {}  # Speichert alle Experimente
        
        print("🎮 ML CONTROL CENTER INITIALISIERT")
        print("=" * 50)
        self._scan_available_models()
    
    def _scan_available_models(self):
        """Scannt alle verfügbaren Models und erkennt Architekturen."""
        
        models_dir = Path("models/saved_models")
        
        if not models_dir.exists():
            print("📁 Noch keine Models vorhanden")
            return
        
        model_files = list(models_dir.glob("*.pth"))
        
        print(f"🔍 SCANNE {len(model_files)} MODELS...")
        
        for model_file in model_files:
            try:
                architecture = self._detect_architecture(model_file)
                size_mb = model_file.stat().st_size / (1024 * 1024)
                
                self.model_registry[model_file.name] = {
                    'path': str(model_file),
                    'architecture': architecture,
                    'size_mb': size_mb,
                    'last_used': None
                }
                
            except Exception as e:
                print(f"⚠️ Fehler bei {model_file.name}: {e}")
        
        if self.model_registry:
            print(f"✅ {len(self.model_registry)} Models registriert")
        else:
            print("📁 Keine Models gefunden - bereit für Training!")
    
    def _detect_architecture(self, model_path):
        """Erkennt Model-Architektur automatisch."""
        
        try:
            if torch.cuda.is_available():
                checkpoint = torch.load(model_path, weights_only=True)
            else:
                checkpoint = torch.load(model_path, weights_only=True, map_location='cpu')
            
            # State dict extrahieren
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # FC Layer analysieren
            if 'fc.weight' in state_dict:
                fc_features = state_dict['fc.weight'].shape[1]
                
                if fc_features == 512:
                    return 'resnet18'
                elif fc_features == 2048:
                    return 'resnet50'
                elif fc_features == 1280:
                    return 'efficientnet_b0'
            
            # EfficientNet Classifier
            elif 'classifier.weight' in state_dict:
                return 'efficientnet_b0'
            
            return 'resnet18'  # Default
            
        except:
            return 'unknown'
    
    def _setup_classifier(self):
        """Setup Classifier falls nötig."""
        if self.classifier is None:
            self.classifier = StreetFoodClassifier(self.config)
    
    # === TRAINING OPERATIONS ===
    
    def train(self, architecture='resnet18', epochs=20, **kwargs):
        """
        Intelligentes Training mit automatischer Registrierung.
        
        Args:
            architecture: Model-Architektur
            epochs: Anzahl Epochen
            **kwargs: Zusätzliche Parameter (batch_size, learning_rate, etc.)
        """
        
        print(f"\n🚀 TRAINING: {architecture.upper()}")
        print("=" * 40)
        
        self._setup_classifier()
        
        # Config temporär anpassen
        original_config = {}
        for key, value in kwargs.items():
            if hasattr(self.config, key.upper()):
                original_config[key] = getattr(self.config, key.upper())
                setattr(self.config, key.upper(), value)
                print(f"   {key}: {value}")
        
        original_epochs = self.config.EPOCHS
        self.config.EPOCHS = epochs
        
        start_time = time.time()
        experiment_id = f"{architecture}_{epochs}ep_{int(time.time())}"
        
        try:
            # Training durchführen
            history = self.classifier.train(
                architecture=architecture,
                pretrained=True,
                save_results=True
            )
            
            training_time = time.time() - start_time
            
            # Experiment registrieren
            self.experiment_history[experiment_id] = {
                'architecture': architecture,
                'epochs': epochs,
                'training_time': training_time,
                'config': kwargs,
                'timestamp': time.time(),
                'status': 'completed'
            }
            
            print(f"\n🎉 TRAINING ABGESCHLOSSEN!")
            print(f"⏱️  Zeit: {training_time/60:.1f} min")
            print(f"📝 Experiment ID: {experiment_id}")
            
            # Model Registry aktualisieren
            self._scan_available_models()
            
            return history, experiment_id
            
        except Exception as e:
            print(f"❌ Training Fehler: {e}")
            self.experiment_history[experiment_id] = {
                'architecture': architecture,
                'epochs': epochs,
                'error': str(e),
                'status': 'failed'
            }
            return None, experiment_id
        
        finally:
            # Config zurücksetzen
            self.config.EPOCHS = original_epochs
            for key, value in original_config.items():
                setattr(self.config, key.upper(), value)
    
    def quick_train(self, architecture='resnet18', epochs=5):
        """Schnelles Training für Tests."""
        return self.train(architecture, epochs)
    
    # === EVALUATION OPERATIONS ===
    
    def evaluate(self, model_name=None, architecture=None, visualizations=True):
        """
        Intelligente Evaluation mit automatischer Model-Erkennung.
        """
        
        print(f"\n📊 EVALUATION")
        print("=" * 30)
        
        self._setup_classifier()
        
        if model_name:
            # Model aus Registry laden
            if model_name in self.model_registry:
                model_info = self.model_registry[model_name]
                model_path = model_info['path']
                detected_arch = model_info['architecture']
                
                # Verwende erkannte oder angegebene Architektur
                arch = architecture or detected_arch
                
                print(f"📁 Model: {model_name}")
                print(f"🏗️  Architektur: {arch}")
                
                try:
                    self.classifier.load_model(model_path, arch)
                    
                    # Last used aktualisieren
                    self.model_registry[model_name]['last_used'] = time.time()
                    
                except Exception as e:
                    print(f"❌ Model Loading Fehler: {e}")
                    return None
            else:
                print(f"❌ Model '{model_name}' nicht in Registry gefunden")
                print(f"💡 Verfügbare Models: {list(self.model_registry.keys())}")
                return None
        
        # Evaluation durchführen
        try:
            results = self.classifier.evaluate(create_visualizations=visualizations)
            
            print(f"\n📈 ERGEBNISSE:")
            print(f"✅ Accuracy: {results['accuracy']:.4f}")
            print(f"✅ F1-Score: {results['f1']:.4f}")
            print(f"✅ Loss: {results['loss']:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Evaluation Fehler: {e}")
            return None
    
    def quick_evaluate(self, model_name=None):
        """Schnelle Evaluation."""
        return self.evaluate(model_name, visualizations=False)
    
    # === STATUS & INFO ===
    
    def status(self):
        """Zeigt vollständigen Status."""
        
        print(f"\n📊 ML CONTROL CENTER STATUS")
        print("=" * 60)
        
        # System Info
        print(f"🖥️  System: PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        
        # Dataset Info
        try:
            self._setup_classifier()
            data_summary = self.classifier.get_data_summary()
            print(f"📁 Dataset: {data_summary['num_classes']} Klassen, {data_summary['train_samples']:,} Training")
        except:
            print(f"📁 Dataset: Nicht verfügbar")
        
        # Models
        print(f"💾 Models: {len(self.model_registry)} registriert")
        if self.model_registry:
            best_model = self.get_best_model()
            print(f"🏆 Bestes Model: {best_model}")
        
        # Experimente
        print(f"🔬 Experimente: {len(self.experiment_history)} durchgeführt")
        
        return {
            'models': len(self.model_registry),
            'experiments': len(self.experiment_history),
            'classifier_ready': self.classifier is not None
        }
    
    def get_best_model(self):
        """Findet das beste Model basierend auf einer Metrik."""
        
        if not self.model_registry:
            return None
        
        # Sortiere nach Last Used
        sorted_models = sorted(
            self.model_registry.items(),
            key=lambda x: x[1]['last_used'] or 0,
            reverse=True
        )
        
        if sorted_models:
            return sorted_models[0][0]  # Name des Models
        return None


# === INITIALISIERUNG ===

# Erstelle globales Control Center
ml = MLControlCenter()

# === CONVENIENCE FUNCTIONS (für Kompatibilität) ===

def quick_status():
    """Quick Status über Control Center."""
    return ml.status()

def quick_train(architecture='resnet18', epochs=5):
    """Quick Training über Control Center."""
    history, exp_id = ml.train(architecture, epochs)
    return history

def quick_evaluate(model_name=None):
    """Quick Evaluation über Control Center."""
    return ml.evaluate(model_name, visualizations=False)

# === USAGE EXAMPLES ===

print(f"""
🎮 ML CONTROL CENTER BEREIT!

=== BASIC OPERATIONS ===
ml.status()                              # Vollständiger Status
ml.train('resnet18', epochs=10)          # Training
ml.evaluate('best_f1_model.pth')         # Evaluation

=== QUICK FUNCTIONS (kompatibel) ===
quick_status()                           # Quick Status  
quick_train('resnet18', epochs=5)        # Quick Training
quick_evaluate('best_f1_model.pth')      # Quick Evaluation

=== ALLES VON EINEM ORT! ===
""")