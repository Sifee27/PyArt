#!/bin/bash
# PyArt Launch Script for Unix/Linux/macOS

cd "$(dirname "$0")"

echo "Starting PyArt Launcher..."
echo "Please ensure you have installed all requirements: pip install -r requirements.txt"

# Check if Python is available
if command -v python3 &> /dev/null; then
    python3 launcher.py
elif command -v python &> /dev/null; then
    python launcher.py
else
    echo "Error: Python not found. Please install Python 3.8+ and try again."
    exit 1
fi

if [ $? -ne 0 ]; then
    echo ""
    echo "Error: Python or required packages not found."
    echo "Please install Python 3.8+ and run: pip install -r requirements.txt"
fi

read -p "Press any key to continue..."
