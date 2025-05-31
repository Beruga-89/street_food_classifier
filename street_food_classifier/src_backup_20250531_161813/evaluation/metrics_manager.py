"""
Metrics management for evaluation results.

This module provides the MetricsManager class for saving, loading,
and managing evaluation results and metrics.
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Optional

from ..utils import get_logger, save_json, load_json, ensure_dir


class MetricsManager:
    """
    Verwaltet das Speichern und Laden von Metriken und Evaluationsergebnissen.
    
    Diese Klasse bietet:
    - Strukturiertes Speichern von Evaluation-Ergebnissen
    - JSON und Pickle Format Support
    - Automatische Zeitstempel und Metadaten
    - Listing und Vergleich gespeicherter Ergebnisse
    
    Example:
        >>> manager = MetricsManager(config)
        >>> manager.save_evaluation_results(results, class_names, "my_model", "validation")
        >>> saved_results = manager.list_saved_results()
    """
    
    def __init__(self, config):
        """
        Initialisiert den MetricsManager.
        
        Args:
            config: Konfigurationsobjekt mit Pfad-Einstellungen
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Evaluation Results Ordner
        self.results_folder = Path(getattr(config, 'EVALUATION_FOLDER', 'outputs/evaluation_results'))
        ensure_dir(self.results_folder)
        
    def save_evaluation_results(self, 
                              results: Dict, 
                              class_names: List[str],
                              model_name: str = "model",
                              dataset_type: str = "validation",
                              additional_metadata: Optional[Dict] = None) -> str:
        """
        Speichert Evaluationsergebnisse mit korrekter JSON-Serialisierung.
        
        Args:
            results: Ergebnisse von trainer.evaluate()
            class_names: Namen der Klassen
            model_name: Name des Models
            dataset_type: Art des Datasets (train/validation/test)
            additional_metadata: Zusätzliche Metadaten
            
        Returns:
            Pfad zur gespeicherten JSON-Datei
            
        Example:
            >>> results = trainer.evaluate(val_loader)
            >>> path = manager.save_evaluation_results(
            ...     results, class_names, "resnet18", "validation"
            ... )
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{dataset_type}_results_{timestamp}"
        
        # Metadaten sammeln
        metadata = {
            'model_name': model_name,
            'dataset_type': dataset_type,
            'timestamp': timestamp,
            'save_date': datetime.now().isoformat(),
            'class_names': class_names,
            'num_classes': len(class_names),
            'dataset_size': len(results.get('labels', [])),
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Hauptdaten strukturieren
        save_data = {
            'metadata': metadata,
            'metrics': {
                'loss': float(results['loss']),
                'accuracy': float(results['accuracy']),
                'f1': float(results['f1'])
            },
            'predictions': results.get('predictions', []),
            'labels': results.get('labels', [])
        }
        
        # Zusätzliche Metriken falls vorhanden
        for key in ['precision', 'recall']:
            if key in results:
                save_data['metrics'][key] = float(results[key])
        
        # Als JSON speichern
        json_path = self.results_folder / f"{filename}.json"
        try:
            save_json(save_data, json_path, use_numpy_encoder=True)
            self.logger.info(f"Evaluation results saved as JSON: {json_path}")
        except Exception as e:
            self.logger.warning(f"JSON save failed: {e}")
            json_path = None
            
        # Als Pickle speichern (Backup für komplexe Objekte)
        pickle_path = self.results_folder / f"{filename}.pkl"
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(save_data, f)
            self.logger.info(f"Evaluation results saved as Pickle: {pickle_path}")
        except Exception as e:
            self.logger.warning(f"Pickle save failed: {e}")
            pickle_path = None
        
        # Summary ausgeben
        self._print_save_summary(save_data, json_path, pickle_path)
        
        return str(json_path) if json_path else str(pickle_path)
    
    def _print_save_summary(self, data: Dict, json_path: Optional[Path], 
                           pickle_path: Optional[Path]) -> None:
        """Gibt Zusammenfassung der gespeicherten Daten aus."""
        print(f"\n[SAVED] Evaluation Results:")
        print(f"  Model: {data['metadata']['model_name']}")
        print(f"  Dataset: {data['metadata']['dataset_type']}")
        print(f"  Accuracy: {data['metrics']['accuracy']:.4f}")
        print(f"  F1-Score: {data['metrics']['f1']:.4f}")
        print(f"  Samples: {data['metadata']['dataset_size']}")
        
        if json_path:
            print(f"  JSON: {json_path}")
        if pickle_path:
            print(f"  Pickle: {pickle_path}")
    
    def load_evaluation_results(self, file_path: Union[str, Path]) -> Dict:
        """
        Lädt gespeicherte Evaluationsergebnisse.
        
        Args:
            file_path: Pfad zur Datei (.json oder .pkl)
            
        Returns:
            Dictionary mit Evaluationsergebnissen
            
        Example:
            >>> data = manager.load_evaluation_results("results_20241130_143022.json")
            >>> print(f"Model: {data['metadata']['model_name']}")
        """
        file_path = Path(file_path)
        
        # Relativer Pfad zu absolut konvertieren
        if not file_path.is_absolute():
            file_path = self.results_folder / file_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")
        
        # Datei laden basierend auf Extension
        if file_path.suffix == '.json':
            data = load_json(file_path)
        elif file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Arrays sicherstellen (falls aus JSON geladen)
        if 'predictions' in data and isinstance(data['predictions'], list):
            import numpy as np
            data['predictions'] = np.array(data['predictions'])
            data['labels'] = np.array(data['labels'])
        
        # Logging
        metadata = data.get('metadata', {})
        self.logger.info(f"Loaded evaluation results from: {file_path}")
        self.logger.info(f"  Model: {metadata.get('model_name', 'Unknown')}")
        self.logger.info(f"  Dataset: {metadata.get('dataset_type', 'Unknown')}")
        self.logger.info(f"  Timestamp: {metadata.get('timestamp', 'Unknown')}")
        
        return data
    
    def list_saved_results(self, sort_by: str = 'timestamp', 
                          descending: bool = True) -> List[Dict]:
        """
        Listet alle gespeicherten Evaluationsergebnisse auf.
        
        Args:
            sort_by: Sortierung ('timestamp', 'accuracy', 'f1', 'model_name')
            descending: Absteigende Sortierung
            
        Returns:
            Liste von Dictionaries mit Result-Informationen
            
        Example:
            >>> results = manager.list_saved_results(sort_by='accuracy')
            >>> for result in results[:5]:  # Top 5
            ...     print(f"{result['model_name']}: {result['accuracy']:.4f}")
        """
        results = []
        
        # Alle JSON und Pickle Dateien finden
        for pattern in ['*.json', '*.pkl']:
            for file_path in self.results_folder.glob(pattern):
                try:
                    # Nur Metadaten laden für Performance
                    if file_path.suffix == '.json':
                        data = load_json(file_path)
                    else:
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                    
                    metadata = data.get('metadata', {})
                    metrics = data.get('metrics', {})
                    
                    result_info = {
                        'file': str(file_path),
                        'filename': file_path.name,
                        'model_name': metadata.get('model_name', 'Unknown'),
                        'dataset_type': metadata.get('dataset_type', 'Unknown'),
                        'timestamp': metadata.get('timestamp', ''),
                        'save_date': metadata.get('save_date', ''),
                        'accuracy': metrics.get('accuracy', 0.0),
                        'f1': metrics.get('f1', 0.0),
                        'loss': metrics.get('loss', float('inf')),
                        'dataset_size': metadata.get('dataset_size', 0),
                        'num_classes': metadata.get('num_classes', 0)
                    }
                    
                    results.append(result_info)
                    
                except Exception as e:
                    self.logger.warning(f"Error reading {file_path}: {e}")
        
        # Sortieren
        if sort_by in ['accuracy', 'f1']:
            results.sort(key=lambda x: x[sort_by], reverse=descending)
        elif sort_by == 'loss':
            results.sort(key=lambda x: x[sort_by], reverse=not descending)  # Niedrigster Loss ist besser
        else:  # timestamp oder andere
            results.sort(key=lambda x: x.get(sort_by, ''), reverse=descending)
        
        # Summary ausgeben
        self._print_results_summary(results)
        
        return results
    
    def _print_results_summary(self, results: List[Dict]) -> None:
        """Gibt Zusammenfassung der gefundenen Ergebnisse aus."""
        print(f"\n[AVAILABLE RESULTS] Found {len(results)} saved evaluation results:")
        print("-" * 100)
        print(f"{'#':<3} {'Model':<20} {'Dataset':<12} {'Accuracy':<10} {'F1-Score':<10} {'Timestamp':<16}")
        print("-" * 100)
        
        for i, result in enumerate(results[:10], 1):  # Top 10 anzeigen
            print(f"{i:<3} {result['model_name']:<20} {result['dataset_type']:<12} "
                  f"{result['accuracy']:<10.4f} {result['f1']:<10.4f} {result['timestamp']:<16}")
        
        if len(results) > 10:
            print(f"... and {len(results) - 10} more results")
        print("-" * 100)
    
    def delete_result(self, file_path: Union[str, Path]) -> bool:
        """
        Löscht gespeicherte Evaluationsergebnisse.
        
        Args:
            file_path: Pfad zur zu löschenden Datei
            
        Returns:
            True wenn erfolgreich gelöscht
        """
        file_path = Path(file_path)
        
        if not file_path.is_absolute():
            file_path = self.results_folder / file_path
        
        try:
            if file_path.exists():
                file_path.unlink()
                self.logger.info(f"Deleted result file: {file_path}")
                return True
            else:
                self.logger.warning(f"File not found: {file_path}")
                return False
        except Exception as e:
            self.logger.error(f"Error deleting {file_path}: {e}")
            return False
    
    def cleanup_old_results(self, keep_latest: int = 10) -> int:
        """
        Löscht alte Evaluationsergebnisse, behält nur die neuesten.
        
        Args:
            keep_latest: Anzahl der neuesten Ergebnisse zum Behalten
            
        Returns:
            Anzahl gelöschter Dateien
        """
        all_results = self.list_saved_results(sort_by='timestamp', descending=True)
        
        if len(all_results) <= keep_latest:
            self.logger.info(f"No cleanup needed. {len(all_results)} results <= {keep_latest}")
            return 0
        
        to_delete = all_results[keep_latest:]
        deleted_count = 0
        
        for result in to_delete:
            if self.delete_result(result['file']):
                deleted_count += 1
        
        self.logger.info(f"Cleanup completed: {deleted_count} old results deleted")
        return deleted_count
    
    def export_results_summary(self, output_file: Optional[str] = None) -> str:
        """
        Exportiert Zusammenfassung aller Ergebnisse als CSV.
        
        Args:
            output_file: Output-Datei (default: auto-generiert)
            
        Returns:
            Pfad zur exportierten Datei
        """
        import pandas as pd
        
        all_results = self.list_saved_results()
        
        if not all_results:
            raise ValueError("No results found to export")
        
        # DataFrame erstellen
        df = pd.DataFrame(all_results)
        
        # Spalten auswählen und sortieren
        columns = ['model_name', 'dataset_type', 'accuracy', 'f1', 'loss', 
                  'dataset_size', 'timestamp', 'filename']
        df = df[columns]
        
        # Output-Datei
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_summary_{timestamp}.csv"
        
        output_path = self.results_folder / output_file
        df.to_csv(output_path, index=False)
        
        self.logger.info(f"Results summary exported to: {output_path}")
        return str(output_path)