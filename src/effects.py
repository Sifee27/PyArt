"""
Effects module for PyArt
Contains all visual effects and filters that can be applied to video frames
"""

import cv2
import numpy as np
import math
from typing import Dict, Callable, Any


class EffectProcessor:
    """Processes and applies visual effects to video frames"""
    
    def __init__(self):
        """Initialize the effect processor with default parameters"""
        self.intensity = 1.0
        self.ascii_detail = 1.0  # ASCII detail level (0.0 to 2.0)
        self.effect_params = {}
        self.frame_counter = 0
        # ASCII character sets for different density levels
        self.ascii_chars_simple = "@%#*+=-:. "
        self.ascii_chars_detailed = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
        self.ascii_chars_blocks = "█▉▊▋▌▍▎▏ "
        
        # Register all available effects
        self.effects: Dict[str, Callable] = {
            'original': self.original,
            'ascii_simple': self.ascii_simple,
            'ascii_detailed': self.ascii_detailed,
            'ascii_blocks': self.ascii_blocks,
            'ascii_color': self.ascii_color,
            'ascii_inverted': self.ascii_inverted,
            'ascii_psychedelic': self.ascii_psychedelic,
            'ascii_rainbow': self.ascii_rainbow,
            'color_inversion': self.color_inversion,
            'pixelation': self.pixelation,
            'edge_detection': self.edge_detection,
            'psychedelic': self.psychedelic,
            'blur': self.blur,
            'posterize': self.posterize,
            'hsv_shift': self.hsv_shift,
            'kaleidoscope': self.kaleidoscope
        }
    
    def get_effect_names(self) -> list:
        """Get list of all available effect names"""
        return list(self.effects.keys())
    
    def apply_effect(self, frame: np.ndarray, effect_name: str) -> np.ndarray:
        """
        Apply the specified effect to a frame
        
        Args:
            frame: Input video frame
            effect_name: Name of the effect to apply
            
        Returns:
            Processed frame with effect applied
        """
        self.frame_counter += 1
        
        if effect_name not in self.effects:
            return frame
        
        try:
            return self.effects[effect_name](frame)
        except Exception as e:
            print(f"Error applying effect {effect_name}: {e}")
            return frame
    
    def set_intensity(self, intensity: float):
        """Set the intensity parameter for effects (0.0 to 2.0)"""
        self.intensity = max(0.0, min(2.0, intensity))
    
    def set_ascii_detail(self, detail: float):
        """Set the ASCII detail level parameter (0.0 to 2.0)"""
        self.ascii_detail = max(0.0, min(2.0, detail))
    
    def original(self, frame: np.ndarray) -> np.ndarray:
        """Return the original frame without modifications"""
        return frame
    
    def color_inversion(self, frame: np.ndarray) -> np.ndarray:
        """Invert all colors in the frame"""
        return 255 - frame
    
    def pixelation(self, frame: np.ndarray) -> np.ndarray:
        """Apply pixelation effect"""
        # Calculate pixel size based on intensity
        pixel_size = max(1, int(20 * self.intensity))
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Resize down then up to create pixelation
        small_frame = cv2.resize(frame, (width // pixel_size, height // pixel_size), 
                                interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small_frame, (width, height), interpolation=cv2.INTER_NEAREST)
    
    def edge_detection(self, frame: np.ndarray) -> np.ndarray:
        """Apply edge detection effect"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection with intensity-based thresholds
        low_threshold = max(10, int(50 * (2.0 - self.intensity)))
        high_threshold = max(20, int(150 * (2.0 - self.intensity)))
        
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Convert back to 3-channel for consistency
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def psychedelic(self, frame: np.ndarray) -> np.ndarray:
        """Apply psychedelic color effect"""
        # Convert to HSV for easier color manipulation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create time-based offset for animation
        time_offset = self.frame_counter * 5 * self.intensity
        
        # Shift hue based on position and time
        h, s, v = cv2.split(hsv)
        h = (h.astype(np.float32) + time_offset) % 180
        
        # Boost saturation
        s = np.clip(s.astype(np.float32) * (1 + self.intensity), 0, 255)
        
        # Recombine and convert back
        hsv_modified = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v])
        return cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2BGR)
    
    def blur(self, frame: np.ndarray) -> np.ndarray:
        """Apply blur effect"""
        kernel_size = max(1, int(15 * self.intensity))
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    def posterize(self, frame: np.ndarray) -> np.ndarray:
        """Apply posterization effect (reduce color levels)"""
        # Calculate number of levels based on intensity (fewer levels = more posterized)
        levels = max(2, int(16 - 14 * self.intensity))
        
        # Reduce color depth
        factor = 256.0 / levels
        posterized = (frame / factor).astype(np.uint8) * factor
        return posterized.astype(np.uint8)
    
    def hsv_shift(self, frame: np.ndarray) -> np.ndarray:
        """Apply HSV color space shift"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Shift hue based on intensity and time
        hue_shift = int(self.intensity * 60 + self.frame_counter * 2) % 180
        h = (h.astype(np.float32) + hue_shift) % 180
        
        hsv_shifted = cv2.merge([h.astype(np.uint8), s, v])
        return cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2BGR)
    
    def kaleidoscope(self, frame: np.ndarray) -> np.ndarray:
        """Apply kaleidoscope effect"""
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Create kaleidoscope by copying and rotating segments
        result = frame.copy()
        
        # Number of segments based on intensity
        segments = max(3, int(6 + 6 * self.intensity))
        angle_step = 360 / segments
        
        for i in range(segments):
            angle = i * angle_step + self.frame_counter * self.intensity
            # Rotate and blend segment
            rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            rotated = cv2.warpAffine(frame, rotation_matrix, (width, height))
            result = cv2.addWeighted(result, 0.7, rotated, 0.3, 0)
        
        return result
    
    def _frame_to_ascii_array(self, frame: np.ndarray, char_set: str, 
                            char_width: int, char_height: int):
        """Convert frame to ASCII character array with colors"""
        height, width = frame.shape[:2]
        
        # Calculate grid dimensions
        grid_width = width // char_width
        grid_height = height // char_height
        
        ascii_chars = []
        colors = []
        
        for y in range(grid_height):
            char_row = []
            color_row = []
            
            for x in range(grid_width):
                # Get region
                y1, y2 = y * char_height, min((y + 1) * char_height, height)
                x1, x2 = x * char_width, min((x + 1) * char_width, width)
                region = frame[y1:y2, x1:x2]
                
                # Calculate average brightness and color
                if region.size > 0:
                    avg_color = np.mean(region, axis=(0, 1))
                    brightness = np.mean(avg_color)
                    
                    # Map brightness to character
                    char_index = int((brightness / 255.0) * (len(char_set) - 1))
                    char_index = max(0, min(len(char_set) - 1, char_index))
                    
                    char_row.append(char_set[char_index])
                    color_row.append((int(avg_color[0]), int(avg_color[1]), int(avg_color[2])))
                else:
                    char_row.append(' ')
                    color_row.append((0, 0, 0))
            
            ascii_chars.append(char_row)
            colors.append(color_row)
        
        return ascii_chars, colors, (grid_width, grid_height)
    
    def _ascii_to_image(self, ascii_chars, colors, target_size, 
                       char_width, char_height, font_scale):
        """Convert ASCII array to image"""
        width, height = target_size
        ascii_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some spacing
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        
        for y, (char_row, color_row) in enumerate(zip(ascii_chars, colors)):
            for x, (char, color) in enumerate(zip(char_row, color_row)):
                if char != ' ':
                    pos_x = x * char_width + 2
                    pos_y = y * char_height + char_height - 2
                    
                    if pos_y < height and pos_x < width:
                        cv2.putText(ascii_image, char, (pos_x, pos_y), 
                                  font, font_scale, color, thickness)
        
        return ascii_image
    
    def ascii_simple(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to simple ASCII art"""
        # Calculate character size based on ASCII detail level
        char_size = max(4, int(12 - 8 * self.ascii_detail))
        
        ascii_chars, colors, _ = self._frame_to_ascii_array(
            frame, self.ascii_chars_simple, char_size, char_size
        )
        
        return self._ascii_to_image(ascii_chars, colors, 
                                   (frame.shape[1], frame.shape[0]), 
                                   char_size, char_size, 0.3 + 0.3 * self.ascii_detail)
    
    def ascii_detailed(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to detailed ASCII art with more characters"""
        char_size = max(6, int(16 - 10 * self.ascii_detail))
        
        ascii_chars, colors, _ = self._frame_to_ascii_array(
            frame, self.ascii_chars_detailed, char_size, char_size
        )
        
        return self._ascii_to_image(ascii_chars, colors, 
                                   (frame.shape[1], frame.shape[0]), 
                                   char_size, char_size, 0.25 + 0.25 * self.ascii_detail)
    
    def ascii_blocks(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to block-based ASCII art"""
        char_size = max(4, int(10 - 6 * self.ascii_detail))
        
        ascii_chars, colors, _ = self._frame_to_ascii_array(
            frame, self.ascii_chars_blocks, char_size, char_size
        )
        
        return self._ascii_to_image(ascii_chars, colors, 
                                   (frame.shape[1], frame.shape[0]), 
                                   char_size, char_size, 0.4 + 0.4 * self.ascii_detail)
    
    def ascii_color(self, frame: np.ndarray) -> np.ndarray:
        """ASCII art with enhanced colors"""
        char_size = max(6, int(14 - 8 * self.ascii_detail))
        
        ascii_chars, colors, _ = self._frame_to_ascii_array(
            frame, self.ascii_chars_simple, char_size, char_size
        )
        
        # Enhance colors
        enhanced_colors = []
        for color_row in colors:
            enhanced_row = []
            for color in color_row:
                # Boost saturation
                hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
                hsv_color[1] = min(255, int(hsv_color[1] * (1 + self.ascii_detail)))
                enhanced_bgr = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]
                enhanced_row.append((int(enhanced_bgr[0]), int(enhanced_bgr[1]), int(enhanced_bgr[2])))
            enhanced_colors.append(enhanced_row)
        
        return self._ascii_to_image(ascii_chars, enhanced_colors, 
                                   (frame.shape[1], frame.shape[0]), 
                                   char_size, char_size, 0.3 + 0.3 * self.ascii_detail)
    
    def ascii_inverted(self, frame: np.ndarray) -> np.ndarray:
        """ASCII art with inverted brightness mapping"""
        char_size = max(6, int(14 - 8 * self.ascii_detail))
        
        # Invert the character set
        inverted_chars = self.ascii_chars_simple[::-1]
        
        ascii_chars, colors, _ = self._frame_to_ascii_array(
            frame, inverted_chars, char_size, char_size
        )
        
        return self._ascii_to_image(ascii_chars, colors, 
                                   (frame.shape[1], frame.shape[0]), 
                                   char_size, char_size, 0.3 + 0.3 * self.ascii_detail)
    
    def ascii_psychedelic(self, frame: np.ndarray) -> np.ndarray:
        """ASCII art with psychedelic color effects"""
        char_size = max(6, int(14 - 8 * self.ascii_detail))
        
        ascii_chars, colors, _ = self._frame_to_ascii_array(
            frame, self.ascii_chars_simple, char_size, char_size
        )
        
        # Apply psychedelic color transformation
        psychedelic_colors = []
        time_factor = self.frame_counter * 0.1 * self.ascii_detail
        
        for y, color_row in enumerate(colors):
            psychedelic_row = []
            for x, color in enumerate(color_row):
                # Create shifting colors based on position and time
                hue_shift = (np.sin(time_factor + x * 0.1) * 60 + 
                           np.cos(time_factor + y * 0.1) * 60) % 360
                
                # Convert to HSV and apply shift
                gray_val = int(0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0])
                hue = int(hue_shift) % 180
                hsv_color = np.array([hue, 255, gray_val], dtype=np.uint8)
                psychedelic_bgr = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]
                psychedelic_row.append((int(psychedelic_bgr[0]), int(psychedelic_bgr[1]), int(psychedelic_bgr[2])))
            
            psychedelic_colors.append(psychedelic_row)
        
        return self._ascii_to_image(ascii_chars, psychedelic_colors, 
                                   (frame.shape[1], frame.shape[0]), 
                                   char_size, char_size, 0.3 + 0.3 * self.ascii_detail)
    
    def ascii_rainbow(self, frame: np.ndarray) -> np.ndarray:
        """ASCII art with rainbow color mapping"""
        char_size = max(6, int(14 - 8 * self.ascii_detail))
        
        ascii_chars, colors, _ = self._frame_to_ascii_array(
            frame, self.ascii_chars_simple, char_size, char_size
        )
        
        # Create rainbow color mapping
        rainbow_colors = []
        
        for y, color_row in enumerate(colors):
            rainbow_row = []
            for x, color in enumerate(color_row):
                # Map position to rainbow hue
                hue = int((x + y + self.frame_counter * 0.5) * self.ascii_detail * 10) % 180
                
                # Get brightness from original color
                gray_val = int(0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0])
                
                # Create rainbow color with original brightness
                hsv_color = np.array([hue, 255, gray_val], dtype=np.uint8)
                rainbow_bgr = cv2.cvtColor(np.uint8([[hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]
                rainbow_row.append((int(rainbow_bgr[0]), int(rainbow_bgr[1]), int(rainbow_bgr[2])))
            
            rainbow_colors.append(rainbow_row)
        
        return self._ascii_to_image(ascii_chars, rainbow_colors, 
                                   (frame.shape[1], frame.shape[0]), 
                                   char_size, char_size, 0.3 + 0.3 * self.ascii_detail)
