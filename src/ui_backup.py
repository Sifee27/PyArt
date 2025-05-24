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
            ord('_'): 'decrease_intensity',            ord('r'): 'reset',
            ord('R'): 'reset',
            ord('h'): 'toggle_help',
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
        """Draw modern, clean information overlay on the frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Modern glass-like panel design
        panel_width = 340
        panel_height = 200 if 'ascii' in current_effect.lower() else 160
        
        # Glass panel background with gradient effect
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (15, 15, 15), -1)
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (80, 80, 80), 2)
        cv2.rectangle(overlay, (12, 12), (panel_width - 2, panel_height - 2), (40, 40, 40), 1)
        
        # Title bar with accent
        cv2.rectangle(overlay, (10, 10), (panel_width, 45), (25, 25, 25), -1)
        cv2.rectangle(overlay, (10, 45), (panel_width, 47), (0, 120, 255), -1)
        
        # Effect name with modern typography
        effect_display = current_effect.replace('_', ' ').title()
        cv2.putText(overlay, f"PYART - {effect_display}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_pos = 75
        
        # Intensity section with enhanced visual bar
        cv2.putText(overlay, "Intensity", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(overlay, f"{intensity:.1f}", (280, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Modern progress bar for intensity
        bar_x, bar_y = 20, y_pos + 8
        bar_bg_width = 250
        bar_height = 8
        
        # Background bar
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_bg_width, bar_y + bar_height), (40, 40, 40), -1)
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_bg_width, bar_y + bar_height), (80, 80, 80), 1)
        
        # Filled portion with gradient-like effect
        filled_width = int(bar_bg_width * min(intensity / 2.0, 1.0))
        if filled_width > 0:
            # Main bar
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 255), -1)
            # Highlight effect
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + filled_width, bar_y + 3), (100, 255, 255), -1)
        
        y_pos += 35
        
        # ASCII detail level (only for ASCII effects)
        if 'ascii' in current_effect.lower():
            cv2.putText(overlay, "ASCII Detail", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(overlay, f"{ascii_detail:.1f}", (280, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # ASCII detail progress bar
            bar_y = y_pos + 8
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_bg_width, bar_y + bar_height), (40, 40, 40), -1)
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_bg_width, bar_y + bar_height), (80, 80, 80), 1)
            
            detail_width = int(bar_bg_width * min(ascii_detail / 2.0, 1.0))
            if detail_width > 0:
                cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + detail_width, bar_y + bar_height), (255, 165, 0), -1)
                cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + detail_width, bar_y + 3), (255, 200, 100), -1)
            
            y_pos += 35
        
        # Effect counter with modern styling
        try:
            effect_index = effect_names.index(current_effect) + 1
            cv2.putText(overlay, f"Effect {effect_index} of {len(effect_names)}", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            y_pos += 25
        except ValueError:
            pass
        
        # Gesture status with modern indicator
        if gesture_mode:
            # Status indicator background
            cv2.rectangle(overlay, (20, y_pos - 15), (140, y_pos + 5), (0, 80, 0), -1)
            cv2.rectangle(overlay, (20, y_pos - 15), (140, y_pos + 5), (0, 150, 0), 1)
            
            # Status text
            cv2.putText(overlay, "GESTURES", (30, y_pos - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            # Animated status dot
            cv2.circle(overlay, (120, y_pos - 7), 4, (0, 255, 0), -1)
            cv2.circle(overlay, (120, y_pos - 7), 6, (0, 200, 0), 1)
        else:
            cv2.putText(overlay, "Gestures: OFF", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        
        # Modern bottom control bar
        controls_height = 35
        control_y = height - controls_height
        
        # Gradient background for controls
        cv2.rectangle(overlay, (0, control_y), (width, height), (8, 8, 8), -1)
        cv2.rectangle(overlay, (0, control_y), (width, control_y + 2), (0, 120, 255), -1)
        
        # Control shortcuts with better spacing
        controls = [
            ("H", "Help"),
            ("G", "Gestures"), 
            ("D", "Debug"),
            ("S", "Save"),
            ("SPACE", "Effects"),
            ("Q", "Quit")
        ]
        
        x_offset = 15
        for key, action in controls:
            # Key highlight
            key_width = len(key) * 8 + 8
            cv2.rectangle(overlay, (x_offset - 2, control_y + 8), 
                         (x_offset + key_width, control_y + 22), (40, 40, 40), -1)
            cv2.rectangle(overlay, (x_offset - 2, control_y + 8), 
                         (x_offset + key_width, control_y + 22), (80, 80, 80), 1)
            
            cv2.putText(overlay, key, (x_offset + 2, control_y + 19), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            cv2.putText(overlay, action, (x_offset + key_width + 8, control_y + 19), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
            
            x_offset += key_width + len(action) * 6 + 25
        
        # Apply glass effect with enhanced blending
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
      def _draw_help_overlay(self, frame: np.ndarray):
        """Draw modern help overlay with all controls"""
        height, width = frame.shape[:2]
        
        # Create modern help panel
        overlay = frame.copy()
        panel_width = 450
        panel_height = height - 20
        
        # Glass panel background
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (8, 8, 8), -1)
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (80, 80, 80), 2)
        cv2.rectangle(overlay, (12, 12), (panel_width - 2, panel_height - 2), (30, 30, 30), 1)
        
        # Title bar
        cv2.rectangle(overlay, (10, 10), (panel_width, 50), (20, 20, 20), -1)
        cv2.rectangle(overlay, (10, 50), (panel_width, 52), (0, 120, 255), -1)
        cv2.putText(overlay, "PYART - HELP & CONTROLS", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Help sections with modern formatting
        sections = [
            ("Basic Controls:", [
                "SPACE - Cycle through effects",
                "S - Save current frame as image",
                "+/- - Adjust effect intensity",
                "R - Reset to original view",
                "H - Toggle this help panel",
                "G - Toggle gesture controls",
                "D - Toggle debug overlay",
                "Q/ESC - Exit PyArt"
            ]),
            ("Gesture Controls:", [
                "👍 Thumbs Up - Increase ASCII detail level",
                "👎 Thumbs Down - Decrease ASCII detail level", 
                "✊ Closed Fist - Capture and save photo"
            ]),
            ("ASCII Effects (1-8):", [
                "1. Original Camera View",
                "2. ASCII Simple (Basic characters)",
                "3. ASCII Detailed (Enhanced density)", 
                "4. ASCII Blocks (Block patterns)",
                "5. ASCII Color (Colored characters)",
                "6. ASCII Inverted (Negative ASCII)",
                "7. ASCII Psychedelic (Color effects)",
                "8. ASCII Rainbow (Multi-color)"
            ]),
            ("Classic Effects (9,0,A-F):", [
                "9. Color Inversion",
                "0. Pixelation",
                "A. Edge Detection",
                "B. Psychedelic Colors",
                "C. Motion Blur",
                "D. Color Posterize"
            ])
        ]
        
        y_pos = 70
        for section_title, items in sections:
            # Section header with accent
            cv2.rectangle(overlay, (20, y_pos - 5), (panel_width - 20, y_pos + 15), (25, 25, 25), -1)
            cv2.putText(overlay, section_title, (25, y_pos + 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_pos += 30
            
            # Section items
            for item in items:
                if y_pos < panel_height - 30:
                    cv2.putText(overlay, f"  {item}", (30, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
                    y_pos += 22
            
            y_pos += 10  # Extra spacing between sections
        
        # Close instruction at bottom
        cv2.rectangle(overlay, (20, panel_height - 35), (panel_width - 20, panel_height - 15), (40, 40, 40), -1)
        cv2.putText(overlay, "Press H again to close help", (25, panel_height - 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Apply glass effect
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
    
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
