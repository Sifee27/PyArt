@echo off
REM PyArt Setup Script for Windows
echo ====================================
echo       PyArt Installation Setup
echo ====================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

python --version
echo ✓ Python found

echo.
echo [2/3] Installing required packages...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo ✓ Packages installed successfully

echo.
echo [3/3] Testing installation...
python test_installation.py
if %errorlevel% neq 0 (
    echo ERROR: Installation test failed
    pause
    exit /b 1
)

echo.
echo ====================================
echo     Setup Complete! 🎉
echo ====================================
echo.
echo You can now run PyArt using:
echo   1. python launcher.py
echo   2. Double-click run_pyart.bat
echo.
pause
