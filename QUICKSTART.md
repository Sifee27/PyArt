# PyArt - Quick Start Guide

## 🚀 Getting Started

### Option 1: Use the Launcher (Recommended)
```bash
python launcher.py
```
Or double-click `run_pyart.bat` on Windows

### Option 2: Direct Launch
```bash
# Main application (requires webcam)
python main.py

# File converter (for images and videos)
python file_converter.py

# Demo mode (no webcam needed)
python demo.py

# Test installation
python test_installation.py
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `SPACE` | Cycle through effects |
| `S` | Save snapshot |
| `+/-` | Adjust effect intensity |
| `R` | Reset to original |
| `H` | Toggle help overlay |
| `Q/ESC` | Quit application |
| `1-9` | Select specific effect |

### Gesture Controls
- **👍 Thumbs Up**: Increase ASCII detail level
- **👎 Thumbs Down**: Decrease ASCII detail level
- **✊ Closed Fist**: Capture a snapshot

## 🖼️ File Converter

The File Converter allows you to transform any image or video into ASCII art.

### How to Use
1. Launch the file converter: `python file_converter.py`
2. Click "Open Image" or "Open Video" to select a file
3. Choose an ASCII effect from the dropdown menu
4. Adjust the ASCII detail level using the slider
5. Click "Process" to apply the effect
6. Save the result using the "Save" button
7. For videos, click "Process Video" to create a full ASCII video

## 🎨 Available Effects

### Original Effect
1. **Original** - Unmodified webcam feed

### ASCII Art Effects (NEW!)
2. **ASCII Simple** - Clean ASCII art using basic characters
3. **ASCII Detailed** - Complex ASCII art with extended character set
4. **ASCII Blocks** - Block-based ASCII art using Unicode blocks
5. **ASCII Color** - ASCII art with enhanced color saturation
6. **ASCII Inverted** - ASCII art with inverted brightness mapping
7. **ASCII Psychedelic** - ASCII art with shifting psychedelic colors
8. **ASCII Rainbow** - ASCII art with rainbow color mapping

### Classic Visual Effects
9. **Color Inversion** - Inverts all colors for artistic effect
10. **Pixelation** - Creates retro pixel art appearance
11. **Edge Detection** - Highlights edges and contours
12. **Psychedelic** - Colorful shifting patterns and hues
13. **Blur** - Artistic Gaussian blur effect
14. **Posterize** - Reduces color depth for poster-like effect
15. **HSV Shift** - Dynamic color space manipulation
16. **Kaleidoscope** - Creates symmetrical mandala patterns

## 📁 Project Structure

```
PyArt/
├── main.py              # Main application
├── launcher.py          # Launcher with menu options
├── demo.py             # Demo mode (no webcam needed)
├── test_installation.py # Installation verification
├── run_pyart.bat       # Windows batch launcher
├── config.json         # Configuration settings
├── requirements.txt    # Python dependencies
├── README.md          # Main documentation
├── QUICKSTART.md      # This file
├── saved_images/      # Screenshot storage
└── src/               # Source code modules
    ├── camera.py      # Webcam handling
    ├── effects.py     # Visual effects
    ├── ui.py         # User interface
    └── utils.py      # Utility functions
```

## 🔧 Configuration

Edit `config.json` to customize:
- Camera index (if multiple cameras)
- Resolution settings
- Default effect and intensity
- Save directory

## 📸 Saving Images

- Press `S` to save current frame
- Images saved to `saved_images/` folder
- Automatic timestamp naming
- PNG format for best quality

## 🆘 Troubleshooting

### Camera Issues
- **No camera detected**: Check webcam connection and permissions
- **Poor performance**: Try lower resolution in config.json
- **Multiple cameras**: Change camera_index in config.json

### Effects Issues
- **Effects not visible**: Adjust intensity with +/- keys
- **Application crashes**: Run test_installation.py to check setup
- **Slow performance**: Close other applications using camera

### Installation Issues
```bash
# Reinstall dependencies
pip install --upgrade opencv-python numpy pillow

# Test installation
python test_installation.py
```

## 🎯 Tips for Best Results

1. **Lighting**: Ensure good, even lighting for clear effects
2. **Background**: Try different backgrounds for varied effects
3. **Movement**: Some effects respond well to movement
4. **Intensity**: Experiment with different intensity levels
5. **Combinations**: Try switching effects rapidly for dynamic art

## 🌟 Have Fun!

PyArt is designed for creative exploration. Try different combinations of effects, lighting conditions, and movements to create unique digital art!
