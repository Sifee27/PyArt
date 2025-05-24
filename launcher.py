"""
PyArt Launcher
Simple launcher script for PyArt with different options
"""

import sys
import subprocess
from src.utils import check_webcam_availability, print_system_info


def show_banner():
    """Display PyArt banner"""
    banner = """
    ╔═══════════════════════════════════════╗
    ║             🎨 PyArt 🎨              ║
    ║    Interactive Webcam Art Studio      ║
    ╚═══════════════════════════════════════╝
    """
    print(banner)


def check_system():
    """Check system requirements"""
    print("Checking system requirements...")
    
    try:
        import cv2
        import numpy as np
        print(f"✓ OpenCV: {cv2.__version__}")
        print(f"✓ NumPy: {np.__version__}")
        
        cameras = check_webcam_availability()
        if cameras:
            print(f"✓ Cameras available: {cameras}")
        else:
            print("⚠ No cameras detected")
        
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False


def main():
    """Main launcher function"""
    show_banner()
    
    if not check_system():
        print("\nPlease install missing dependencies:")
        print("pip install opencv-python numpy pillow")
        return
    
    print("\nChoose an option:")
    print("1. 🎥 Launch PyArt (webcam required)")
    print("2. 🖼️  Run File Converter (convert images/videos to ASCII)")
    print("3. 🎬 Run Effects Demo (no webcam needed)")
    print("4. 📊 Run Installation Test")
    print("5. 🔧 Show System Info")
    print("6. 📚 View Help")
    print("7. 🚪 Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                print("Launching PyArt...")
                subprocess.run([sys.executable, "main.py"])
                break
                
            elif choice == '2':
                print("Running Effects Demo...")
                subprocess.run([sys.executable, "demo.py"])
                break
                
            elif choice == '3':
                print("Running Installation Test...")
                subprocess.run([sys.executable, "test_installation.py"])
                
            elif choice == '4':
                print_system_info()
                
            elif choice == '5':
                show_help()
                
            elif choice == '6':
                print("Goodbye! 👋")
                break
                
            else:
                print("Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


def show_help():
    """Show help information"""
    help_text = """
    🎨 PyArt Help 🎨
    
    MAIN APPLICATION CONTROLS:
    ┌─────────────────────────────────────┐
    │ SPACE       - Cycle through effects │
    │ S           - Save snapshot         │
    │ +/-         - Adjust intensity      │
    │ R           - Reset to original     │
    │ H           - Toggle help overlay   │
    │ Q/ESC       - Quit application      │
    │ 1-9         - Select effect directly│
    └─────────────────────────────────────┘
    
    🆕 NEW ASCII ART EFFECTS:
    1. Original      - No effects applied
    2. ASCII Simple  - Clean ASCII text art
    3. ASCII Detail  - Complex ASCII characters
    4. ASCII Blocks  - Unicode block art
    5. ASCII Color   - Enhanced color ASCII
    6. ASCII Invert  - Inverted brightness
    7. ASCII Psyche  - Shifting color ASCII
    8. ASCII Rainbow - Rainbow colored ASCII
    
    📺 CLASSIC EFFECTS:
    9. Color Invert  - Inverts all colors
    10. Pixelation   - Retro pixel art effect
    11. Edge Detect  - Highlights edges
    12. Psychedelic  - Colorful shifting patterns
    13. Blur         - Artistic blur effect
    14. Posterize    - Reduced color depth
    15. HSV Shift    - Color space manipulation
    16. Kaleidoscope - Symmetrical patterns
    
    💡 ASCII ART TIPS:
    • Higher intensity = smaller characters (more detail)
    • Lower intensity = larger characters (more readable)
    • ASCII effects work great in good lighting
    • Try ASCII Rainbow for dynamic color effects
    
    TROUBLESHOOTING:
    • No camera detected: Check webcam connection
    • Poor performance: Reduce window size
    • Effects not working: Try different intensity
    """
    print(help_text)


if __name__ == "__main__":
    main()
