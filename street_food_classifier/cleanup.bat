@echo off
REM WINDOWS BATCH CLEANUP SCRIPT - cleanup.bat
REM Doppelklick zum Ausführen

echo 🚀 STARTING WINDOWS CODE CLEANUP
echo =================================

REM SCHRITT 1: Backup erstellen
echo 📦 Step 1: Creating safety backup...
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,8%_%dt:~8,6%"
set "backupDir=src_backup_%timestamp%"

if exist "src" (
    xcopy "src" "%backupDir%\" /E /I /H /Y >nul 2>&1
    echo ✅ Backup created: %backupDir%
) else (
    echo ❌ src directory not found!
    pause
    exit /b 1
)

REM SCHRITT 2: Problematische Dateien entfernen
echo 🧹 Step 2: Removing problematic files...

if exist "src\visualization\visualizer.py.backup" (
    del "src\visualization\visualizer.py.backup"
    echo   ✅ Removed: visualizer.py.backup
)

if exist "working_dashboard.py" (
    del "working_dashboard.py"
    echo   ✅ Removed: working_dashboard.py
)

if exist "paste.txt" (
    del "paste.txt"
    echo   ✅ Removed: paste.txt
)

if exist "paste-2.txt" (
    del "paste-2.txt"
    echo   ✅ Removed: paste-2.txt
)

REM SCHRITT 3: Checkpoint-Ordner bereinigen
echo 🧹 Step 3: Cleaning checkpoint directories...
for /d /r . %%d in (.ipynb_checkpoints) do (
    if exist "%%d" (
        rmdir /s /q "%%d"
        echo   ✅ Removed: %%d
    )
)

for /d /r . %%d in (__pycache__) do (
    if exist "%%d" (
        rmdir /s /q "%%d"
        echo   ✅ Removed: %%d
    )
)

REM SCHRITT 4: Aktuelle Visualizer sichern
echo 📋 Step 4: Backing up current visualizer...
if exist "src\visualization\visualizer.py" (
    copy "src\visualization\visualizer.py" "src\visualization\visualizer_old_%timestamp%.py" >nul
    echo ✅ Current visualizer backed up
) else (
    echo ❌ Visualizer file not found!
)

echo.
echo 🎯 CLEANUP COMPLETED!
echo =====================
echo.
echo NEXT STEPS:
echo 1. 📝 Replace src\visualization\visualizer.py with new ProfessionalVisualizer
echo 2. 📝 Update src\street_food_classifier.py methods  
echo 3. 🧪 Run verification script
echo.
echo 📞 Ready for manual file replacement!
echo.
pause