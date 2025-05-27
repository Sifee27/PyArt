#!/bin/bash
# PyArt Setup Script for Unix/Linux/macOS

echo "===================================="
echo "      PyArt Installation Setup"
echo "===================================="
echo

cd "$(dirname "$0")"

echo "[1/3] Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    $PYTHON_CMD --version
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    $PYTHON_CMD --version
else
    echo "ERROR: Python is not installed or not in PATH"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✓ Python found"

echo
echo "[2/3] Installing required packages..."
$PYTHON_CMD -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install packages"
    echo "Please check your internet connection and try again"
    exit 1
fi

echo "✓ Packages installed successfully"

echo
echo "[3/3] Testing installation..."
$PYTHON_CMD test_installation.py
if [ $? -ne 0 ]; then
    echo "ERROR: Installation test failed"
    exit 1
fi

echo
echo "===================================="
echo "     Setup Complete! 🎉"
echo "===================================="
echo
echo "You can now run PyArt using:"
echo "  1. $PYTHON_CMD launcher.py"
echo "  2. ./run_pyart.sh"
echo

# Make shell scripts executable
chmod +x run_pyart.sh
chmod +x run_pyart_voice.sh
chmod +x run_converter.sh

echo "Shell scripts have been made executable."
echo
read -p "Press any key to continue..."
