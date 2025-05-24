# PyArt - Interactive Webcam Art Application

![PyArt Demo](https://github.com/username/PyArt/raw/main/docs/demo.gif)

PyArt is a Python-based interactive art application that uses your computer's webcam to create real-time visual effects and digital art, with a special focus on ASCII art generation and gesture controls.

## ✨ Features

- **Live Webcam Feed**: Real-time video capture and processing
- **ASCII Art**: Transform your webcam feed into ASCII art with various styles
- **16 Visual Effects**: Including ASCII variants and classic visual effects
- **Hand Gesture Controls**: Use thumbs up/down to adjust ASCII detail level and fist gesture to capture photos
- **Image & Video Converter**: Convert any image or video file to ASCII art
- **Interactive Controls**: Keyboard shortcuts to switch effects and adjust parameters
- **Snapshot Capture**: Save your favorite moments as image files
- **Modular Design**: Easy to extend with new effects

## 🖼️ Gallery

<table>
  <tr>
    <td><img src="https://github.com/username/PyArt/raw/main/docs/ascii_simple.png" width="200" alt="ASCII Simple"></td>
    <td><img src="https://github.com/username/PyArt/raw/main/docs/ascii_detailed.png" width="200" alt="ASCII Detailed"></td>
    <td><img src="https://github.com/username/PyArt/raw/main/docs/ascii_blocks.png" width="200" alt="ASCII Blocks"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/username/PyArt/raw/main/docs/edge_detection.png" width="200" alt="Edge Detection"></td>
    <td><img src="https://github.com/username/PyArt/raw/main/docs/psychedelic.png" width="200" alt="Psychedelic"></td>
    <td><img src="https://github.com/username/PyArt/raw/main/docs/kaleidoscope.png" width="200" alt="Kaleidoscope"></td>
  </tr>
</table>

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/username/PyArt.git
   cd PyArt
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

There are multiple ways to run PyArt:

### Using the Launcher (Recommended)

```bash
# On Windows, you can double-click run_pyart.bat or run:
python launcher.py
```

### Direct Launch

```bash
# Main application (requires webcam)
python main.py

# Demo mode (no webcam needed)
python demo.py

# Test installation
python test_installation.py
```

## 🕹️ Controls

| Key | Action |
|-----|--------|
| `SPACE` | Cycle through effects |
| `S` | Save snapshot |
| `+/-` | Adjust effect intensity |
| `G` | Toggle gesture controls |
| `D` | Toggle debug overlay |
| `H` | Show help |
| `R` | Reset to original |
| `Q/ESC` | Quit |
| `1-9` | Select specific effects |

### Gesture Controls

- **👍 Thumbs Up**: Increase ASCII detail level
- **👎 Thumbs Down**: Decrease ASCII detail level
- **✊ Closed Fist**: Capture a snapshot

## 🖼️ File Converter

PyArt now includes a dedicated file converter that allows you to transform any image or video into ASCII art!

### Features

- **Image Conversion**: Convert any image to ASCII art with 8 different styles
- **Video Processing**: Transform videos into ASCII art videos
- **Adjustable Detail**: Fine-tune the ASCII detail level for optimal results
- **Batch Processing**: Process multiple files with the same settings
- **Preview Mode**: See the results before saving

### Usage

```bash
# Launch the file converter directly
python file_converter.py

# Or select option 2 from the launcher menu
python launcher.py
```

## 🎨 Available Effects

### Original
- **Original** - Unmodified webcam feed

### ASCII Art Effects
- **ASCII Simple** - Clean ASCII art using basic characters
- **ASCII Detailed** - Higher resolution ASCII with extensive character set
- **ASCII Blocks** - Block-based ASCII using Unicode block characters
- **ASCII Color** - ASCII with enhanced colors
- **ASCII Inverted** - Inverted brightness mapping for ASCII
- **ASCII Psychedelic** - ASCII with psychedelic color effects
- **ASCII Rainbow** - ASCII with rainbow color mapping

### Classic Effects
- **Color Inversion** - Inverts all colors
- **Pixelation** - Creates a pixelated art effect
- **Edge Detection** - Highlights edges and contours
- **Psychedelic** - Colorful shifting palette
- **Blur** - Artistic blur effect
- **Posterize** - Reduces color depth for poster-like effect
- **HSV Shift** - Shifts hue, saturation, and value
- **Kaleidoscope** - Creates symmetrical patterns

## 📋 Requirements

- Python 3.7+
- Webcam
- OpenCV
- NumPy
- Pillow
- Pygame (for UI interactions)

## 📁 Project Structure

```
PyArt/
├── main.py                 # Main application entry point
├── demo.py                 # Demo mode for testing effects
├── launcher.py             # User-friendly launcher
├── run_pyart.bat           # Windows launcher script
├── config.json             # Configuration settings
├── requirements.txt        # Python dependencies
├── saved_images/           # Directory for saved images
└── src/                    # Source code modules
    ├── camera.py           # Webcam capture
    ├── effects.py          # Visual effects
    ├── gesture_detector.py # Hand gesture detection
    ├── ui.py               # User interface
    └── utils.py            # Utility functions
```

## 👐 Hand Gesture Detection

PyArt includes a custom hand gesture detection system built with OpenCV. The system:

1. Detects skin tones in the webcam feed
2. Identifies hand contours and analyzes their shape
3. Recognizes three key gestures:
   - Thumbs up (increases ASCII detail)
   - Thumbs down (decreases ASCII detail)
   - Closed fist (captures a snapshot)

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for their excellent computer vision library
- Python community for the tools that made this project possible
│   └── utils.py        # Utility functions
├── saved_images/       # Directory for saved snapshots
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Contributing

Feel free to add new effects by extending the `effects.py` module!

## License

This project is open source and available under the MIT License.
