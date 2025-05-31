import os
import sys
from pathlib import Path  # WICHTIG: Expliziter Import

def verify_cleanup():
    """Verify that cleanup was successful on Windows."""
    
    print("🔍 VERIFYING CLEANUP SUCCESS (WINDOWS)...")
    print("=" * 45)
    
    success = True
    
    # Check that old files are gone
    problematic_files = [
        "src/visualization/visualizer.py.backup",
        "src\\visualization\\visualizer.py.backup",  # Windows path
        "working_dashboard.py", 
        "paste.txt",
        "paste-2.txt"
    ]
    
    print("📁 Checking removed files...")
    for file_path in problematic_files:
        file_obj = Path(file_path)  # Explizite Path-Nutzung
        if file_obj.exists():
            print(f"❌ FAILED: {file_path} still exists")
            success = False
        else:
            print(f"✅ GOOD: {file_path} removed")
    
    # Check that new visualizer exists
    print("\n📄 Checking new visualizer...")
    new_visualizer = Path("src/visualization/visualizer.py")
    if not new_visualizer.exists():
        print(f"❌ FAILED: New visualizer not found at {new_visualizer}")
        success = False
    else:
        print(f"✅ GOOD: Visualizer file exists")
        
        # Check for ProfessionalVisualizer class
        try:
            with open(new_visualizer, 'r', encoding='utf-8') as f:
                content = f.read()
                if "class ProfessionalVisualizer" in content:
                    print("✅ GOOD: ProfessionalVisualizer class found")
                else:
                    print("❌ FAILED: ProfessionalVisualizer class not found")
                    print("💡 Make sure you replaced the file with the new code")
                    success = False
        except Exception as e:
            print(f"❌ FAILED: Cannot read visualizer file: {e}")
            success = False
    
    # Check Python path setup
    print("\n🐍 Checking Python environment...")
    try:
        import sys
        current_dir = str(Path.cwd())
        if current_dir not in sys.path:
            print("⚠️ WARNING: Current directory not in Python path")
            print("💡 You may need to run: sys.path.insert(0, '.')")
        else:
            print("✅ GOOD: Python path configured correctly")
    except Exception as e:
        print(f"⚠️ WARNING: Python path check failed: {e}")
    
    # Check imports
    print("\n📦 Testing critical imports...")
    try:
        from pathlib import Path
        print("✅ GOOD: pathlib import works")
    except ImportError:
        print("❌ FAILED: pathlib import failed")
        success = False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ GOOD: matplotlib import works")
    except ImportError:
        print("❌ FAILED: matplotlib import failed")
        print("💡 Install with: pip install matplotlib")
        success = False
    
    print("\n" + "="*45)
    if success:
        print("🎉 CLEANUP VERIFICATION SUCCESSFUL!")
        print("📋 Next steps:")
        print("  1. Test imports: python test_imports_windows.py")
        print("  2. Test training: python -c \"from ml_control_center import ml; ml.train('resnet18', epochs=1)\"")
    else:
        print("❌ CLEANUP VERIFICATION FAILED!")
        print("💡 Please check the failed items above")
    
    return success

def create_test_import_script():
    """Create a Windows-specific import test script."""
    
    test_script = '''# test_imports_windows.py
# Test script for Windows import verification

import sys
from pathlib import Path

# Add current directory to path
current_dir = str(Path.cwd())
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_imports():
    """Test all critical imports for Windows."""
    
    print("🧪 TESTING IMPORTS (WINDOWS)...")
    print("=" * 35)
    
    tests = [
        ("config", "from config import Config"),
        ("visualizer", "from src.visualization import ProfessionalVisualizer"),
        ("classifier", "from src import StreetFoodClassifier"),
        ("control center", "from ml_control_center import ml")
    ]
    
    success_count = 0
    
    for name, import_code in tests:
        try:
            exec(import_code)
            print(f"✅ {name}: OK")
            success_count += 1
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")
        except Exception as e:
            print(f"⚠️ {name}: ERROR - {e}")
    
    print(f"\\n📊 Results: {success_count}/{len(tests)} imports successful")
    
    if success_count == len(tests):
        print("🎉 ALL IMPORTS SUCCESSFUL!")
        print("🚀 Ready for training!")
        return True
    else:
        print("❌ Some imports failed")
        print("💡 Check your file replacements")
        return False

if __name__ == "__main__":
    test_imports()
'''
    
    with open("test_imports_windows.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print("✅ Created: test_imports_windows.py")

if __name__ == "__main__":
    success = verify_cleanup()
    
    if success:
        create_test_import_script()
    
    print("\n📞 Press Enter to continue...")
    input()
    
    sys.exit(0 if success else 1)