@echo off
cd /d "%~dp0"
echo Starting PyArt Launcher...
echo Please ensure you have installed all requirements: pip install -r requirements.txt
python launcher.py
if %errorlevel% neq 0 (
    echo.
    echo Error: Python or required packages not found.
    echo Please install Python 3.8+ and run: pip install -r requirements.txt
)
pause
