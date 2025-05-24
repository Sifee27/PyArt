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
            ord('r'): 'reset', # General reset, might be different from reset_view
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
            
            # Voice command toggle
            ord('w'): 'toggle_voice',
            ord('W'): 'toggle_voice',
            
            # Recording
            ord('v'): 'toggle_recording',
            ord('V'): 'toggle_recording',
            
            # ASCII detail control
            ord('['): 'decrease_ascii_detail',
            ord(']'): 'increase_ascii_detail',
            
            # Navigation keys (more intuitive for most users)
            ord('.'): 'cycle_effect',  # Right arrow
            ord(','): 'previous_effect',  # Left arrow
            
            # Zoom and pan
            ord('z'): 'increase_zoom',
            ord('Z'): 'decrease_zoom',
            ord('a'): 'pan_left',
            ord('A'): 'pan_left',
            ord('d'): 'pan_right',
            ord('D'): 'pan_right',
            ord('w'): 'pan_up',
            ord('W'): 'pan_up',
            ord('s'): 'pan_down',
            ord('S'): 'pan_down',
            ord('x'): 'reset_view',
            ord('X'): 'reset_view',
            
            # Mirror controls
            ord('m'): 'toggle_mirror_h',
            ord('M'): 'toggle_mirror_v',
            
            # Color themes
            ord('t'): 'next_theme',
            ord('T'): 'previous_theme',
            
            # Blend controls
            ord('['): 'decrease_blend',
            ord(']'): 'increase_blend',
            
            # FPS controls
            ord(','): 'decrease_fps',
            ord('.'): 'increase_fps',
            
            # Blend mode toggle
            ord('l'): 'toggle_blend',
            ord('L'): 'toggle_blend',

            # Original Effect Selection Keys (some will be overridden)
            ord('a'): 'select_effect_0',   # A = Original (Safe)
            ord('b'): 'select_effect_1',   # B = ASCII Simple (Conflict with start_burst)
            ord('c'): 'select_effect_2',   # C = ASCII Detailed (Conflict with clear_drawing)
            # D is toggle_debug
            ord('e'): 'select_effect_3',   # E = ASCII Blocks (Conflict with cycle_face_effect)
            ord('f'): 'select_effect_4',   # F = ASCII Color (Conflict with toggle_face_tracking)

            # --- COOL FEATURES & OTHER NEW FEATURES ---
            # These may overwrite previous key assignments.
            # The last assignment in the dict literal for a given key wins.

            ord('b'): 'start_burst',          # B for Burst (overwrites select_effect_1)
            ord('B'): 'start_burst',

            ord('x'): 'toggle_timelapse',    # X for Time-lapse (overwrites original reset_view binding if it was 'x')
            ord('X'): 'toggle_timelapse',    
            
            ord('k'): 'toggle_drawing',      # K for Drawing mode
            ord('K'): 'toggle_drawing',
            ord('n'): 'toggle_night_vision', # N for Night vision
            ord('N'): 'toggle_night_vision',
            ord('p'): 'toggle_split_screen', # P for Split screen
            ord('P'): 'toggle_split_screen',
            
            ord('c'): 'clear_drawing',       # C for Clear drawing (overwrites select_effect_2)
            ord('C'): 'change_drawing_color',# Shift+C for Change drawing color
            
            ord('j'): 'adjust_timelapse_speed', 
            ord('J'): 'adjust_timelapse_speed',

            # Motion Trails and Face Tracking (Definitive assignments)
            ord('m'): 'toggle_motion_trails', # M for Motion (overwrites toggle_mirror_h)
            ord('M'): 'toggle_motion_trails', # Shift+M for Motion (overwrites toggle_mirror_v)
            
            ord('f'): 'toggle_face_tracking', # F for Face (overwrites select_effect_4)
            ord('F'): 'toggle_face_tracking',
            
            ord('e'): 'cycle_face_effect',    # E for Effect (overwrites select_effect_3)
            ord('E'): 'cycle_face_effect',
            
            ord('o'): 'next_face_emoji',      # O for next emoji (does not conflict)
            ord('O'): 'next_face_emoji',
        }
        
        # Re-assign any overridden effect selection keys to number keys
        if 'select_effect_0' not in self.key_mappings.values(): # Original (original 'a')
            self.key_mappings[ord('4')] = 'select_effect_0'
        if 'select_effect_1' not in self.key_mappings.values(): # ASCII Simple (original 'b')
            self.key_mappings[ord('5')] = 'select_effect_1' 
        if 'select_effect_2' not in self.key_mappings.values(): # ASCII Detailed (original 'c')
            self.key_mappings[ord('6')] = 'select_effect_2' 
        if 'select_effect_3' not in self.key_mappings.values(): # ASCII Blocks (original 'e')
            self.key_mappings[ord('7')] = 'select_effect_3' 
        if 'select_effect_4' not in self.key_mappings.values(): # ASCII Color (original 'f')
            self.key_mappings[ord('8')] = 'select_effect_4'
    
    def create_window(self):
        """Create the display window"""
        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self.window_created = True
    
    def display_frame(self, frame: np.ndarray, current_effect: str, intensity: float, 
                     effect_names: list, gesture_mode: bool = False, voice_mode: bool = False,
                     ascii_detail: float = 1.0):
        """
        Display frame with UI overlay information
        
        Args:
            frame: Video frame to display
            current_effect: Name of currently active effect
            intensity: Current effect intensity
            effect_names: List of all available effects
            gesture_mode: Whether gesture control is enabled
            voice_mode: Whether voice command control is enabled
            ascii_detail: Current ASCII detail level
        """
        if not self.window_created:
            self.create_window()
        
        display_frame = frame.copy()
        
        # Add UI overlays
        if self.show_info or self.info_timer > 0:
            self._draw_info_overlay(display_frame, current_effect, intensity, effect_names, 
                                   gesture_mode, voice_mode, ascii_detail)
            if self.info_timer > 0:
                self.info_timer -= 1
        
        if self.show_help:
            self._draw_help_overlay(display_frame)
        
        cv2.imshow(self.window_name, display_frame)
    
    def _draw_info_overlay(self, frame: np.ndarray, current_effect: str, 
                          intensity: float, effect_names: list, gesture_mode: bool = False,
                          voice_mode: bool = False, ascii_detail: float = 1.0):
        """Draw modern, clean information overlay on the frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Modern glass-like panel design
        panel_width = 340
        panel_height = 200 if 'ascii' in current_effect.lower() else 160
        
        # Add panel height for voice command status if enabled
        if voice_mode:
            panel_height += 25
        
        # Panel position - anchored at bottom left
        panel_x = 20
        panel_y = height - panel_height - 20
        
        # Draw panel background with rounded corners
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     (30, 30, 30), -1)
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     (200, 200, 200), 1)
        
        # Convert effect name to display format
        effect_display = current_effect.replace('_', ' ').title()
        
        # Draw current effect name
        cv2.putText(overlay, effect_display, (panel_x + 20, panel_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Show effect index
        y_pos = panel_y + 80
        try:
            effect_index = effect_names.index(current_effect) + 1
            cv2.putText(overlay, f"Effect {effect_index} of {len(effect_names)}", (panel_x + 20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        except ValueError:
            # Effect not in list (could be a special effect)
            pass
        
        # Draw intensity slider
        y_pos = panel_y + 110
        cv2.putText(overlay, "Intensity:", (panel_x + 20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Intensity value
        cv2.putText(overlay, f"{intensity:.1f}", (panel_x + 280, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Intensity bar
        bar_y = y_pos + 10
        bar_bg_width = 200
        cv2.rectangle(overlay, (panel_x + 20, bar_y), (panel_x + 20 + bar_bg_width, bar_y + 10), 
                     (70, 70, 70), -1)
        
        # Filled part
        filled_width = int(bar_bg_width * min(intensity / 2.0, 1.0))
        if filled_width > 0:
            cv2.rectangle(overlay, (panel_x + 20, bar_y), (panel_x + 20 + filled_width, bar_y + 10), 
                         (0, 180, 255), -1)
        
        # ASCII detail slider if applicable
        if 'ascii' in current_effect.lower():
            y_pos = panel_y + 150
            cv2.putText(overlay, "ASCII Detail:", (panel_x + 20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Detail value
            cv2.putText(overlay, f"{ascii_detail:.1f}", (panel_x + 280, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Detail bar
            bar_y = y_pos + 10
            cv2.rectangle(overlay, (panel_x + 20, bar_y), (panel_x + 20 + bar_bg_width, bar_y + 10), 
                         (70, 70, 70), -1)
            
            # Filled part
            detail_width = int(bar_bg_width * min(ascii_detail / 2.0, 1.0))
            if detail_width > 0:
                cv2.rectangle(overlay, (panel_x + 20, bar_y), (panel_x + 20 + detail_width, bar_y + 10), 
                             (120, 180, 0), -1)
        
        # Control mode indicators at bottom of panel
        y_pos = panel_y + panel_height - 15
        
        # Gesture control indicator
        if gesture_mode:
            cv2.putText(overlay, "Gesture: ON", (panel_x + 20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        else:
            cv2.putText(overlay, "Gesture: OFF", (panel_x + 20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        
        # Voice command indicator
        if voice_mode:
            y_pos = panel_y + panel_height - 15
            cv2.putText(overlay, "Voice: ON", (panel_x + 150, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
            # Add active indicator dot
            cv2.circle(overlay, (panel_x + 120, y_pos - 7), 6, (0, 200, 0), 1)
        else:
            cv2.putText(overlay, "Voice: OFF", (panel_x + 150, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        
        # Apply the overlay with alpha blending
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    
    def _draw_help_overlay(self, frame: np.ndarray):
        """Draw help information overlay"""
        height, width = frame.shape[:2]
        overlay = np.zeros_like(frame)
        
        # Semi-transparent background
        cv2.rectangle(overlay, (0, 0), (width, height), (30, 30, 30), -1)
        
        # Title
        cv2.putText(overlay, "PyArt - Interactive Webcam Art", (30, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Help content
        help_text = [
            "KEYBOARD CONTROLS:",
            " ",
            "SPACEBAR - Cycle through effects",
            "S - Save snapshot",
            "V - Toggle video recording",
            "+/- - Adjust effect intensity",
            "[/] - Adjust ASCII detail level",
            "H - Toggle this help screen",
            "G - Toggle gesture control",
            "W - Toggle voice commands",
            "D - Toggle debug overlay",
            "Q/ESC - Quit application",
            " ",
            "COOL FEATURES:",
            "N - Toggle night vision",
            "M - Toggle motion trails",
            "F - Toggle face tracking",
            "E - Cycle face effects",
            "K - Toggle drawing mode",
            "C - Clear drawing",
            "P - Toggle split screen",
            "X - Toggle timelapse",
            " ",
            "VOICE COMMANDS:",
            "'next effect' - Change to next effect",
            "'take photo' - Capture snapshot",
            "'zoom in/out' - Adjust zoom level",
            "'night vision' - Toggle night mode",
            "'motion trails' - Toggle motion trails",
            "'face tracking' - Toggle face detection",
            " ",
            "GESTURE CONTROLS:",
            "Thumbs Up - Increase detail",
            "Thumbs Down - Decrease detail",
            "Fist - Take photo"
        ]
        
        y_offset = 100
        for line in help_text:
            if line.endswith(":"):
                # Section headers
                cv2.putText(overlay, line, (30, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 1)
                y_offset += 30
            else:
                # Regular help text
                cv2.putText(overlay, line, (30, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                y_offset += 30
        
        # Version and attribution
        cv2.putText(overlay, "PyArt v1.2 - Press H to hide help", (30, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # Apply overlay with alpha blending
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    def handle_input(self) -> Optional[str]:
        """
        Handle keyboard input
        
        Returns:
            Optional[str]: Action to perform, or None if no action
        """
        key = cv2.waitKey(1) & 0xFF
        
        if key in self.key_mappings:
            return self.key_mappings[key]
        
        return None
    
    def show_info_temporarily(self, frames: int = 90):
        """
        Show information overlay temporarily
        
        Args:
            frames: Number of frames to show info
        """
        self.info_timer = frames
