"""
UI module for PyArt
Handles user interface, keyboard input, and display management
"""

import cv2
import numpy as np
from typing import Optional, Callable, Dict, Any


class UserInterface:
    """Manages user interface and input handling for PyArt"""
    
    def __init__(self, window_name: str = "PyArt - Interactive Webcam Art"):
        """
        Initialize the user interface
        
        Args:
            window_name: Name of the display window
        """
        self.window_name = window_name
        self.window_created = False
        self.show_help = False
        self.show_info = True
        self.info_timer = 0
        
        # Key mappings
        self.key_mappings = {
            ord(' '): 'cycle_effect',
            ord('s'): 'save_snapshot',
            ord('S'): 'save_snapshot',
            ord('+'): 'increase_intensity',
            ord('='): 'increase_intensity',
            ord('-'): 'decrease_intensity',
            ord('_'): 'decrease_intensity',
            ord('r'): 'reset',
            ord('R'): 'reset',            ord('h'): 'toggle_help',
            ord('H'): 'toggle_help',
            ord('g'): 'toggle_gesture',
            ord('G'): 'toggle_gesture',
            ord('d'): 'toggle_debug',
            ord('D'): 'toggle_debug',
            ord('q'): 'quit',
            ord('Q'): 'quit',
            27: 'quit',  # ESC key
                }
        
        # Number keys for direct effect selection (1-9, then 0 for effect 10)
        for i in range(1, 10):
            self.key_mappings[ord(str(i))] = f'select_effect_{i-1}'
        self.key_mappings[ord('0')] = 'select_effect_9'  # 0 = effect 10
        
        # Letter keys for effects 11-16
        self.key_mappings[ord('a')] = 'select_effect_10'  # A = Color Inversion
        self.key_mappings[ord('b')] = 'select_effect_11'  # B = Pixelation  
        self.key_mappings[ord('c')] = 'select_effect_12'  # C = Edge Detection
        self.key_mappings[ord('d')] = 'select_effect_13'  # D = Psychedelic
        self.key_mappings[ord('e')] = 'select_effect_14'  # E = Blur
        self.key_mappings[ord('f')] = 'select_effect_15'  # F = Posterize
        # G and above reserved for future effects
    
    def create_window(self):
        """Create the display window"""
        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self.window_created = True
    
    def display_frame(self, frame: np.ndarray, current_effect: str, intensity: float, 
                     effect_names: list, gesture_mode: bool = False, ascii_detail: float = 1.0):
        """
        Display frame with UI overlay information
        
        Args:
            frame: Video frame to display
            current_effect: Name of currently active effect
            intensity: Current effect intensity
            effect_names: List of all available effects
            gesture_mode: Whether gesture control is enabled
            ascii_detail: Current ASCII detail level
        """
        if not self.window_created:
            self.create_window()
        
        display_frame = frame.copy()
        
        # Add UI overlays
        if self.show_info or self.info_timer > 0:
            self._draw_info_overlay(display_frame, current_effect, intensity, effect_names, 
                                   gesture_mode, ascii_detail)
            if self.info_timer > 0:
                self.info_timer -= 1
        
        if self.show_help:
            self._draw_help_overlay(display_frame)
        
        cv2.imshow(self.window_name, display_frame)
    
    def _draw_info_overlay(self, frame: np.ndarray, current_effect: str, 
                          intensity: float, effect_names: list, gesture_mode: bool = False,
                          ascii_detail: float = 1.0):
        """Draw information overlay on the frame"""
        height, width = frame.shape[:2]
        
        # Semi-transparent background for text
        overlay = frame.copy()
        
        # Effect name
        effect_text = f"Effect: {current_effect.replace('_', ' ').title()}"
        cv2.putText(overlay, effect_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Intensity
        intensity_text = f"Intensity: {intensity:.1f}"
        cv2.putText(overlay, intensity_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ASCII detail level (only show for ASCII effects)
        if 'ascii' in current_effect.lower():
            ascii_text = f"ASCII Detail: {ascii_detail:.1f}"
            cv2.putText(overlay, ascii_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset = 120
        else:
            y_offset = 90
        
        # Effect number indicator
        try:
            effect_index = effect_names.index(current_effect) + 1
            number_text = f"[{effect_index}/{len(effect_names)}]"
            cv2.putText(overlay, number_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        except ValueError:
            pass
        
        # Gesture mode indicator
        if gesture_mode:
            gesture_text = "Gestures: ON (👍↑detail 👎↓detail ✊photo)"
            cv2.putText(overlay, gesture_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_offset += 25
        
        # Controls hint
        controls_text = "H:help G:gestures D:debug Q:quit"
        cv2.putText(overlay, controls_text, (10, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    def _draw_help_overlay(self, frame: np.ndarray):
        """Draw help overlay with all controls"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent background
        overlay = np.zeros_like(frame)
        cv2.rectangle(overlay, (0, 0), (400, height), (0, 0, 0), -1)
        
        help_text = [
            "PyArt ASCII Art Controls:",
            "",
            "SPACE - Cycle effects",
            "S - Save snapshot",
            "+/- - Adjust intensity",
            "R - Reset to original",
            "H - Toggle this help",
            "G - Toggle gestures",
            "D - Toggle debug overlay",
            "Q/ESC - Quit",
            "",
            "Gesture Controls:",
            "👍 Thumbs Up - Increase ASCII detail",
            "👎 Thumbs Down - Decrease ASCII detail", 
            "✊ Closed Fist - Capture photo",
            "",
            "ASCII Effects (1-8):",
            "1. Original",
            "2. ASCII Simple",
            "3. ASCII Detailed", 
            "4. ASCII Blocks",
            "5. ASCII Color",
            "6. ASCII Inverted",
            "7. ASCII Psychedelic",
            "8. ASCII Rainbow",
            "",
            "Classic Effects (9,0,A-F):",
            "9. Color Inversion",
            "0. Pixelation",
            "A. Edge Detection",
            "B. Psychedelic",
            "C. Blur",
            "D. Posterize"
        ]
        
        y_start = 30
        for i, line in enumerate(help_text):
            y_pos = y_start + i * 25
            if y_pos < height - 20:
                color = (255, 255, 0) if line.endswith(":") else (255, 255, 255)
                font_scale = 0.6 if line.endswith(":") else 0.5
                cv2.putText(overlay, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        
        # Blend with frame
        cv2.addWeighted(frame, 0.3, overlay, 0.7, 0, frame)
    
    def handle_input(self) -> Optional[str]:
        """
        Handle keyboard input and return action command
        
        Returns:
            str: Action command or None if no action
        """
        key = cv2.waitKey(1) & 0xFF
        
        if key == 255:  # No key pressed
            return None
        
        return self.key_mappings.get(key, None)
    
    def show_info_temporarily(self, duration: int = 60):
        """Show info overlay temporarily (for duration frames)"""
        self.info_timer = duration
    
    def cleanup(self):
        """Clean up UI resources"""
        if self.window_created:
            cv2.destroyAllWindows()
            self.window_created = False
