#!/usr/bin/env python3
"""
Dashboard Wrapper - Ruft das evaluation_analysis Modul auf

Usage: python dashboard.py
"""

import sys
from pathlib import Path

# Füge src zum Python Path hinzu
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from visualization.evaluation_analysis import main
    
    if __name__ == "__main__":
        print("🍕 STREET FOOD CLASSIFICATION - DASHBOARD")
        print("=" * 50)
        main()
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Stelle sicher dass src/visualization/evaluation_analysis.py existiert")
except Exception as e:
    print(f"❌ Fehler: {e}")
