# ml_command_dashboard.py
# Professional Command Center Dashboard für dein ML Control Center

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

# Import dein funktionierendes Control Center
from ml_control_center import ml

class MLCommandDashboard:
    """
    Professional Command Center Dashboard für systematische ML-Experimente.
    
    Features:
    - Interactive Experiment Management
    - Automated Model Comparison  
    - Performance Tracking
    - Result Visualization
    - Export & Reporting
    - Automatic Trainer API enhancement
    """
    
    def __init__(self):
        self.ml = ml  # Dein funktionierendes Control Center
        self.experiment_log = []
        self.dashboard_history = {
            'sessions': [],
            'experiments': {},
            'best_models': {},
            'performance_trends': []
        }
        
        # Dashboard Directories
        self.dashboard_dir = Path("outputs/dashboard")
        self.session_dir = self.dashboard_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Store the Configuration method for lazy application
        self._trainer_configured = False
        
        print("🎮 ML COMMAND DASHBOARD INITIALIZED")
        print("=" * 60)
        print(f"📁 Session Directory: {self.session_dir}")
        print("🔧 Trainer API enhancement: READY")
        
    def _ensure_trainer_configuration(self):
        """Stellt sicher, dass der trainer configuration angewendet ist (lazy loading)."""
        
        if not self._trainer_configured:
            # Setup classifier if needed
            if self.ml.classifier is None:
                self.ml._setup_classifier()
            
            # Now apply the fix
            self._configure_trainer_api()
            self._trainer_configured = True
            print("🔧 Trainer API enhancement applied!")
    
    def _configure_trainer_api(self):
        """Wendet den trainer configuration an (nur wenn classifier existiert)."""
        
        import types
        
        def enhanced_train_method(self, architecture='resnet18', pretrained=True, save_results=True, **kwargs):
            """
            Professional training method with correct Trainer API.
            Automatically handles trainer.fit() vs trainer.train() API differences.
            """
            
            try:
                # Setup
                if self.num_classes is None:
                    self.setup_data()
                if self.model is None:
                    self.setup_model(architecture=architecture, pretrained=pretrained)
                
                # Handle epochs correctly
                target_epochs = kwargs.get('epochs', self.config.EPOCHS)
                original_epochs = self.config.EPOCHS
                
                if target_epochs != original_epochs:
                    print(f"🔧 Setting epochs: {original_epochs} -> {target_epochs}")
                    self.config.EPOCHS = target_epochs
                
                # Professional API handling - use trainer.fit()
                print(f"🚀 Training {architecture} for {self.config.EPOCHS} epochs...")
                
                self.training_history = self.trainer.fit(
                    train_loader=self.train_loader,
                    val_loader=self.val_loader
                )
                
                self.is_trained = True
                
                # Create predictor if needed
                if self.predictor is None:
                    _, val_transform = self.data_manager.get_transforms()
                    from src.inference import Predictor
                    self.predictor = Predictor(
                        self.model, self.class_names, self.device, self.config, val_transform
                    )
                
                # Save results if requested
                if save_results:
                    try:
                        self.save_training_results()
                    except Exception as e:
                        print(f"⚠️ Could not save results: {e}")
                
                # Professional visualization
                try:
                    if hasattr(self, 'visualizer') and self.training_history:
                        history_plot = self.visualizer.plot_training_history(
                            self.training_history,
                            save=True,
                            show=False,  # Don't show in dashboard
                            save_name=f"{architecture}_training_history.png"
                        )
                        if history_plot:
                            print(f"📊 Training visualization saved: {history_plot}")
                except Exception as e:
                    print(f"⚠️ Visualization skipped: {e}")
                
                # Restore config
                if target_epochs != original_epochs:
                    self.config.EPOCHS = original_epochs
                
                print("✅ Training completed successfully!")
                return self.training_history
                
            except Exception as e:
                print(f"❌ Training failed: {e}")
                
                # Restore config on error
                if 'original_epochs' in locals():
                    self.config.EPOCHS = original_epochs
                
                raise e
        
        # Apply the professional training method (only if classifier exists)
        if self.ml.classifier is not None:
            self.ml.classifier.train = types.MethodType(enhanced_train_method, self.ml.classifier)
    
    def show_main_menu(self):
        """Zeigt das Hauptmenü des Dashboards."""
        
        print(f"\n🎮 ML COMMAND CENTER DASHBOARD")
        print("=" * 50)
        print("✅ Professional Training API: READY")
        print("")
        print("1. 🔬 Quick Experiment (Single Model)")
        print("2. 📊 Model Comparison Study")
        print("3. 🎯 Hyperparameter Optimization")
        print("4. 📈 Performance Analysis")
        print("5. 🏆 Best Model Tournament")
        print("6. 📋 System Status & Reports")
        print("7. 🔧 Advanced Configuration")
        print("8. 📦 Export Results")
        print("9. ❌ Exit Dashboard")
        
        try:
            choice = input("\n📝 Enter your choice (1-9): ").strip()
            return choice
        except (KeyboardInterrupt, EOFError):
            return "9"
    
    def quick_experiment(self):
        """Führt ein schnelles Experiment durch."""
        
        # Ensure trainer configuration is applied
        self._ensure_trainer_configuration()
        
        print(f"\n🔬 QUICK EXPERIMENT")
        print("=" * 30)
        
        # Architecture Selection
        print("Available architectures:")
        architectures = ['resnet18', 'resnet50', 'efficientnet_b0']
        for i, arch in enumerate(architectures, 1):
            print(f"  {i}. {arch}")
        
        try:
            arch_choice = input("Select architecture (1-3) or enter custom: ").strip()
            
            if arch_choice.isdigit() and 1 <= int(arch_choice) <= 3:
                architecture = architectures[int(arch_choice) - 1]
            else:
                architecture = arch_choice if arch_choice else 'resnet18'
            
            epochs = input(f"Epochs (default: 10): ").strip()
            epochs = int(epochs) if epochs.isdigit() else 10
            
            print(f"\n🚀 Starting Quick Experiment: {architecture} | {epochs} epochs")
            
            start_time = time.time()
            
            # Training
            history, exp_id = self.ml.train(architecture, epochs=epochs)
            
            if history:
                # Evaluation
                eval_results = self.ml.evaluate(visualizations=True)
                
                training_time = time.time() - start_time
                
                # Log Experiment
                experiment_data = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'quick_experiment',
                    'architecture': architecture,
                    'epochs': epochs,
                    'experiment_id': exp_id,
                    'training_time_minutes': training_time / 60,
                    'results': eval_results,
                    'status': 'completed'
                }
                
                self.experiment_log.append(experiment_data)
                
                # Show Results
                print(f"\n🎉 EXPERIMENT COMPLETED!")
                print(f"📊 Results:")
                print(f"   Architecture: {architecture}")
                print(f"   Training Time: {training_time/60:.1f} minutes")
                print(f"   Accuracy: {eval_results['accuracy']:.4f}")
                print(f"   F1-Score: {eval_results['f1']:.4f}")
                print(f"   Experiment ID: {exp_id}")
                
                # Save to session
                self._save_experiment_to_session(experiment_data)
                
                return experiment_data
            else:
                print("❌ Experiment failed")
                return None
                
        except Exception as e:
            print(f"❌ Quick experiment error: {e}")
            return None
    
    def model_comparison_study(self):
        """Führt systematischen Model-Vergleich durch."""
        
        # Ensure trainer configuration is applied
        self._ensure_trainer_configuration()
        
        print(f"\n📊 MODEL COMPARISON STUDY")
        print("=" * 40)
        
        # Configuration
        architectures = ['resnet18', 'resnet50', 'efficientnet_b0']
        
        print("Study Configuration:")
        epochs = input("Epochs per model (default: 8): ").strip()
        epochs = int(epochs) if epochs.isdigit() else 8
        
        include_all = input("Include all architectures? (y/N): ").strip().lower()
        
        if include_all == 'y':
            study_architectures = architectures
        else:
            print("Select architectures (enter numbers separated by commas):")
            for i, arch in enumerate(architectures, 1):
                print(f"  {i}. {arch}")
            
            selection = input("Selection (e.g., 1,3): ").strip()
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                study_architectures = [architectures[i] for i in indices if 0 <= i < len(architectures)]
            except:
                study_architectures = ['resnet18', 'efficientnet_b0']  # Default
        
        print(f"\n🚀 Starting Comparison Study:")
        print(f"   Architectures: {', '.join(study_architectures)}")
        print(f"   Epochs per model: {epochs}")
        
        study_results = {}
        study_start_time = time.time()
        
        for i, architecture in enumerate(study_architectures, 1):
            print(f"\n📊 Training Model {i}/{len(study_architectures)}: {architecture.upper()}")
            print("-" * 50)
            
            model_start_time = time.time()
            
            try:
                # Training
                history, exp_id = self.ml.train(architecture, epochs=epochs)
                
                if history:
                    # Evaluation
                    eval_results = self.ml.evaluate(visualizations=False)  # Save time
                    
                    model_time = time.time() - model_start_time
                    
                    study_results[architecture] = {
                        'experiment_id': exp_id,
                        'training_time_minutes': model_time / 60,
                        'evaluation': eval_results,
                        'status': 'success'
                    }
                    
                    print(f"✅ {architecture}: Acc={eval_results['accuracy']:.4f}, F1={eval_results['f1']:.4f}")
                    
                else:
                    print(f"❌ {architecture}: Training failed")
                    study_results[architecture] = {'status': 'failed'}
                    
            except Exception as e:
                print(f"❌ {architecture}: Error - {e}")
                study_results[architecture] = {'status': 'failed', 'error': str(e)}
        
        # Analysis
        total_study_time = time.time() - study_start_time
        successful_models = {k: v for k, v in study_results.items() if v.get('status') == 'success'}
        
        if successful_models:
            print(f"\n📊 COMPARISON STUDY RESULTS")
            print("=" * 50)
            
            # Ranking by F1-Score
            ranking = sorted(successful_models.items(), 
                           key=lambda x: x[1]['evaluation']['f1'], reverse=True)
            
            print(f"🏆 MODEL RANKING (by F1-Score):")
            for i, (arch, data) in enumerate(ranking, 1):
                eval_data = data['evaluation']
                time_min = data['training_time_minutes']
                print(f"   {i}. {arch:15} - F1: {eval_data['f1']:.4f} | Acc: {eval_data['accuracy']:.4f} | Time: {time_min:.1f}m")
            
            # Best Model
            best_model = ranking[0]
            print(f"\n🥇 BEST MODEL: {best_model[0].upper()}")
            print(f"   F1-Score: {best_model[1]['evaluation']['f1']:.4f}")
            print(f"   Accuracy: {best_model[1]['evaluation']['accuracy']:.4f}")
            print(f"   Training Time: {best_model[1]['training_time_minutes']:.1f} minutes")
            
            # Study Summary
            print(f"\n📈 STUDY SUMMARY:")
            print(f"   Total Study Time: {total_study_time/60:.1f} minutes")
            print(f"   Successful Models: {len(successful_models)}/{len(study_architectures)}")
            
            # Save Study Results
            study_data = {
                'timestamp': datetime.now().isoformat(),
                'type': 'model_comparison_study',
                'architectures': study_architectures,
                'epochs': epochs,
                'total_time_minutes': total_study_time / 60,
                'results': study_results,
                'ranking': [(arch, data['evaluation']) for arch, data in ranking],
                'best_model': best_model[0]
            }
            
            self.experiment_log.append(study_data)
            self._save_experiment_to_session(study_data)
            
            return study_data
        else:
            print("\n❌ No successful models in study")
            return None
    
    def performance_analysis(self):
        """Analysiert Performance-Trends und -Vergleiche."""
        
        print(f"\n📈 PERFORMANCE ANALYSIS")
        print("=" * 35)
        
        if not self.experiment_log:
            print("📝 No experiments logged yet. Run some experiments first!")
            return
        
        # Current Session Analysis
        session_experiments = self.experiment_log
        
        print(f"📊 Current Session Analysis:")
        print(f"   Total Experiments: {len(session_experiments)}")
        
        # Collect all results
        all_results = []
        for exp in session_experiments:
            if exp.get('results'):
                # Single experiment
                all_results.append({
                    'type': exp['type'],
                    'architecture': exp['architecture'],
                    'accuracy': exp['results']['accuracy'],
                    'f1': exp['results']['f1'],
                    'training_time': exp.get('training_time_minutes', 0)
                })
            elif exp.get('results') and isinstance(exp['results'], dict):
                # Study with multiple models
                for arch, data in exp['results'].items():
                    if data.get('status') == 'success':
                        all_results.append({
                            'type': exp['type'],
                            'architecture': arch,
                            'accuracy': data['evaluation']['accuracy'],
                            'f1': data['evaluation']['f1'],
                            'training_time': data.get('training_time_minutes', 0)
                        })
        
        if all_results:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(all_results)
            
            print(f"\n📊 PERFORMANCE STATISTICS:")
            print(f"   Models Trained: {len(df)}")
            print(f"   Average Accuracy: {df['accuracy'].mean():.4f}")
            print(f"   Average F1-Score: {df['f1'].mean():.4f}")
            print(f"   Total Training Time: {df['training_time'].sum():.1f} minutes")
            
            # Architecture Performance
            if len(df['architecture'].unique()) > 1:
                print(f"\n🏗️ ARCHITECTURE PERFORMANCE:")
                arch_stats = df.groupby('architecture').agg({
                    'accuracy': ['mean', 'std', 'count'],
                    'f1': ['mean', 'std'],
                    'training_time': 'mean'
                }).round(4)
                
                for arch in arch_stats.index:
                    acc_mean = arch_stats.loc[arch, ('accuracy', 'mean')]
                    f1_mean = arch_stats.loc[arch, ('f1', 'mean')]
                    time_mean = arch_stats.loc[arch, ('training_time', 'mean')]
                    count = arch_stats.loc[arch, ('accuracy', 'count')]
                    
                    print(f"   {arch:15} - Acc: {acc_mean:.4f} | F1: {f1_mean:.4f} | Avg Time: {time_mean:.1f}m | Runs: {count}")
            
            # Best Performance
            best_f1_idx = df['f1'].idxmax()
            best_model = df.loc[best_f1_idx]
            
            print(f"\n🏆 BEST PERFORMANCE THIS SESSION:")
            print(f"   Architecture: {best_model['architecture']}")
            print(f"   F1-Score: {best_model['f1']:.4f}")
            print(f"   Accuracy: {best_model['accuracy']:.4f}")
            
            # Save analysis
            analysis_data = {
                'timestamp': datetime.now().isoformat(),
                'session_summary': {
                    'total_experiments': len(session_experiments),
                    'total_models': len(df),
                    'average_accuracy': df['accuracy'].mean(),
                    'average_f1': df['f1'].mean(),
                    'total_training_time': df['training_time'].sum(),
                    'best_model': best_model.to_dict()
                },
                'architecture_stats': arch_stats.to_dict() if len(df['architecture'].unique()) > 1 else {}
            }
            
            analysis_file = self.session_dir / "performance_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=2, default=str)
            
            print(f"\n💾 Analysis saved to: {analysis_file}")
            
        else:
            print("📝 No completed experiments with results found.")
    
    def system_status_report(self):
        """Zeigt umfassenden System-Status."""
        
        print(f"\n📋 SYSTEM STATUS & REPORTS")
        print("=" * 40)
        
        # Control Center Status
        print("🎮 ML CONTROL CENTER STATUS:")
        status = self.ml.status()
        
        # Dashboard Status  
        print(f"\n📊 DASHBOARD SESSION STATUS:")
        print(f"   Session Directory: {self.session_dir}")
        print(f"   Experiments This Session: {len(self.experiment_log)}")
        print(f"   Session Duration: {self._get_session_duration()}")
        
        # File System Status
        print(f"\n💾 FILE SYSTEM STATUS:")
        models_dir = Path("models/saved_models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pth"))
            total_size = sum(f.stat().st_size for f in model_files) / (1024**2)  # MB
            print(f"   Saved Models: {len(model_files)} files ({total_size:.1f} MB)")
        else:
            print(f"   Saved Models: No models directory")
        
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            plot_files = list(outputs_dir.rglob("*.png"))
            result_files = list(outputs_dir.rglob("*.json"))
            print(f"   Plots Generated: {len(plot_files)}")
            print(f"   Result Files: {len(result_files)}")
        
        # Recent Experiments
        if self.experiment_log:
            print(f"\n🔬 RECENT EXPERIMENTS:")
            for exp in self.experiment_log[-3:]:  # Last 3
                exp_type = exp.get('type', 'unknown')
                timestamp = exp.get('timestamp', '')[:19]  # Remove microseconds
                
                if exp.get('architecture'):
                    print(f"   • {timestamp} - {exp_type} ({exp['architecture']})")
                else:
                    print(f"   • {timestamp} - {exp_type}")
    
    def export_session_results(self):
        """Exportiert alle Session-Ergebnisse."""
        
        print(f"\n📦 EXPORT SESSION RESULTS")
        print("=" * 35)
        
        if not self.experiment_log:
            print("📝 No experiments to export.")
            return
        
        # Create export data
        export_data = {
            'session_info': {
                'start_time': self.session_dir.name.split('_', 1)[1],
                'export_time': datetime.now().isoformat(),
                'total_experiments': len(self.experiment_log),
                'session_directory': str(self.session_dir)
            },
            'experiments': self.experiment_log,
            'summary': self._create_session_summary()
        }
        
        # Export files
        export_json = self.session_dir / "session_export.json"
        with open(export_json, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        # Create readable report
        report_md = self.session_dir / "session_report.md"
        with open(report_md, 'w') as f:
            f.write(self._create_markdown_report(export_data))
        
        print(f"✅ Session exported:")
        print(f"   📄 JSON: {export_json}")
        print(f"   📋 Report: {report_md}")
        print(f"   📁 Session Dir: {self.session_dir}")
    
    def _save_experiment_to_session(self, experiment_data):
        """Speichert Experiment in Session-Verzeichnis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_file = self.session_dir / f"experiment_{timestamp}.json"
        
        with open(exp_file, 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)
    
    def _get_session_duration(self):
        """Berechnet Session-Dauer."""
        session_time = self.session_dir.name.split('_', 1)[1]
        try:
            start_time = datetime.strptime(session_time, "%Y%m%d_%H%M%S")
            duration = datetime.now() - start_time
            return f"{duration.seconds // 60} minutes"
        except:
            return "Unknown"
    
    def _create_session_summary(self):
        """Erstellt Session-Zusammenfassung."""
        if not self.experiment_log:
            return {}
        
        # Collect metrics
        total_experiments = len(self.experiment_log)
        architectures_tested = set()
        total_training_time = 0
        best_f1 = 0
        best_model = None
        
        for exp in self.experiment_log:
            if exp.get('architecture'):
                architectures_tested.add(exp['architecture'])
            
            if exp.get('training_time_minutes'):
                total_training_time += exp['training_time_minutes']
            
            # Check for best F1
            if exp.get('results') and exp['results'].get('f1', 0) > best_f1:
                best_f1 = exp['results']['f1']
                best_model = exp['architecture']
        
        return {
            'total_experiments': total_experiments,
            'unique_architectures': len(architectures_tested),
            'architectures_tested': list(architectures_tested),
            'total_training_time_minutes': total_training_time,
            'best_f1_score': best_f1,
            'best_model': best_model
        }
    
    def _create_markdown_report(self, export_data):
        """Erstellt Markdown-Report."""
        
        session_info = export_data['session_info']
        summary = export_data['summary']
        
        report = f"""# ML Dashboard Session Report

                ## Session Information
                - **Start Time:** {session_info['start_time']}
                - **Export Time:** {session_info['export_time']}
                - **Total Experiments:** {session_info['total_experiments']}
                
                ## Summary
                - **Unique Architectures:** {summary.get('unique_architectures', 0)}
                - **Architectures Tested:** {', '.join(summary.get('architectures_tested', []))}
                - **Total Training Time:** {summary.get('total_training_time_minutes', 0):.1f} minutes
                - **Best F1-Score:** {summary.get('best_f1_score', 0):.4f}
                - **Best Model:** {summary.get('best_model', 'N/A')}
                
                ## Experiments
                
                """
        
        for i, exp in enumerate(export_data['experiments'], 1):
            exp_type = exp.get('type', 'unknown')
            timestamp = exp.get('timestamp', '')[:19]
            
            report += f"### Experiment {i}: {exp_type}\n"
            report += f"- **Timestamp:** {timestamp}\n"
            
            if exp.get('architecture'):
                report += f"- **Architecture:** {exp['architecture']}\n"
            
            if exp.get('epochs'):
                report += f"- **Epochs:** {exp['epochs']}\n"
            
            if exp.get('results'):
                results = exp['results']
                report += f"- **Accuracy:** {results.get('accuracy', 0):.4f}\n"
                report += f"- **F1-Score:** {results.get('f1', 0):.4f}\n"
            
            if exp.get('training_time_minutes'):
                report += f"- **Training Time:** {exp['training_time_minutes']:.1f} minutes\n"
            
            report += "\n"
        
        return report
    
    def run_dashboard(self):
        """Hauptschleife des Dashboards."""
        
        print("🎮 Welcome to ML Command Center Dashboard!")
        
        while True:
            try:
                choice = self.show_main_menu()
                
                if choice == "1":
                    self.quick_experiment()
                elif choice == "2":
                    self.model_comparison_study()
                elif choice == "3":
                    print("🎯 Hyperparameter Optimization coming soon!")
                elif choice == "4":
                    self.performance_analysis()
                elif choice == "5":
                    print("🏆 Model Tournament coming soon!")
                elif choice == "6":
                    self.system_status_report()
                elif choice == "7":
                    print("🔧 Advanced Configuration coming soon!")
                elif choice == "8":
                    self.export_session_results()
                elif choice == "9":
                    print("\n👋 Exiting ML Dashboard...")
                    self.export_session_results()
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
                
                if choice != "9":
                    input("\n📝 Press Enter to continue...")
                    
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 Dashboard interrupted. Exporting session...")
                self.export_session_results()
                break
        
        print("🎉 ML Dashboard session completed!")


# === CONVENIENCE FUNCTIONS ===

def launch_dashboard():
    """Startet das ML Dashboard mit automatischem Fix."""
    dashboard = MLCommandDashboard()
    dashboard.run_dashboard()
    return dashboard

def quick_experiment():
    """Schnelles Experiment mit automatischem Fix."""
    dashboard = MLCommandDashboard()
    return dashboard.quick_experiment()

def quick_comparison():
    """Schneller Model-Vergleich mit automatischem Fix."""
    dashboard = MLCommandDashboard()
    return dashboard.model_comparison_study()


if __name__ == "__main__":
    print("""
🎮 ML COMMAND CENTER DASHBOARD

=== LAUNCH OPTIONS ===
launch_dashboard()              # Full interactive dashboard with auto-configuration
quick_experiment()              # Single quick experiment
quick_comparison()              # Quick model comparison

=== TROUBLESHOOTING ===
configure_trainer_api_for_existing() # Fix existing ML instances

=== PROFESSIONAL & READY! ===
    """)