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
        # New ASCII grid using only regular English characters
        self.ascii_chars_grid = "#@8&WBMmwqpdbgXA$%Z0QOLCIUY[]}{?/\\+=-_~<>i!;:,\"^`'. "
        
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
            'ascii_grid': self.ascii_grid,  # Add new ASCII grid effect
            'color_inversion': self.color_inversion,
            'pixelation': self.pixelation,
            'edge_detection': self.edge_detection,
            'psychedelic': self.psychedelic,
            'blur': self.blur,
            'posterize': self.posterize,
            'hsv_shift': self.hsv_shift,
            'kaleidoscope': self.kaleidoscope,
            'matrix_rain': self.matrix_rain,
            'thermal_vision': self.thermal_vision,
            'glitch_art': self.glitch_art,
            'oil_painting': self.oil_painting,
            'retro_crt': self.retro_crt,
            'neon_glow': self.neon_glow,
            'watercolor': self.watercolor
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
    
    def matrix_rain(self, frame: np.ndarray) -> np.ndarray:
        """Apply Matrix-style digital rain effect"""
        height, width = frame.shape[:2]
        
        # Convert to green-tinted image
        green_frame = frame.copy()
        green_frame[:, :, 0] = green_frame[:, :, 0] * 0.2  # Reduce blue
        green_frame[:, :, 2] = green_frame[:, :, 2] * 0.2  # Reduce red
        green_frame[:, :, 1] = np.clip(green_frame[:, :, 1] * 1.5, 0, 255)  # Enhance green
        
        # Create digital rain overlay
        rain_overlay = np.zeros_like(frame)
        
        # Parameters based on intensity
        num_streams = int(30 + 50 * self.intensity)
        char_size = max(8, int(20 - 10 * self.intensity))
        
        # Matrix characters
        matrix_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        
        for i in range(num_streams):
            x = int((i * width) / num_streams)
            # Stream falls based on time
            y_offset = (self.frame_counter * 3 + i * 20) % (height + 200)
            
            for j in range(0, height, char_size):
                y = j + y_offset - 200
                if 0 <= y < height:
                    # Fade effect - characters are brighter at the "head" of the stream
                    alpha = max(0, 1.0 - (j / (height * 0.7)))
                    char = matrix_chars[(i + j + self.frame_counter) % len(matrix_chars)]
                    
                    # Draw character
                    color = (0, int(255 * alpha), 0)  # Green
                    cv2.putText(rain_overlay, char, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, color, 1)
        
        # Blend with original
        return cv2.addWeighted(green_frame, 0.7, rain_overlay, 0.3, 0)
    
    def thermal_vision(self, frame: np.ndarray) -> np.ndarray:
        """Apply thermal imaging effect"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thermal colormap
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        # Add some blur for thermal effect
        thermal = cv2.GaussianBlur(thermal, (5, 5), 0)
        
        # Enhance based on intensity
        if self.intensity > 1.0:
            # Increase contrast
            thermal = cv2.addWeighted(thermal, self.intensity, thermal, 0, -50)
        
        return thermal
    
    def glitch_art(self, frame: np.ndarray) -> np.ndarray:
        """Apply digital glitch art effect"""
        result = frame.copy()
        height, width = frame.shape[:2]
        
        # Digital noise based on intensity
        noise_level = int(20 * self.intensity)
        
        # RGB channel shifting
        shift_amount = int(5 * self.intensity)
        if shift_amount > 0:
            # Shift red channel
            result[:-shift_amount, :, 2] = frame[shift_amount:, :, 2]
            # Shift blue channel
            result[shift_amount:, :, 0] = frame[:-shift_amount, :, 0]
        
        # Random horizontal line displacement
        num_glitches = int(10 * self.intensity)
        for _ in range(num_glitches):
            y = np.random.randint(0, height)
            glitch_height = np.random.randint(1, 10)
            displacement = np.random.randint(-20, 20)
            
            if y + glitch_height < height:
                line = result[y:y+glitch_height, :].copy()
                # Shift the line horizontally
                if displacement > 0:
                    result[y:y+glitch_height, displacement:] = line[:, :-displacement]
                elif displacement < 0:
                    result[y:y+glitch_height, :displacement] = line[:, -displacement:]
        
        # Add digital noise
        noise = np.random.randint(0, noise_level, (height, width, 3), dtype=np.uint8)
        result = cv2.add(result, noise)
        
        return result
    
    def oil_painting(self, frame: np.ndarray) -> np.ndarray:
        """Apply oil painting artistic effect"""
        # Bilateral filter for smooth regions while preserving edges
        smooth = cv2.bilateralFilter(frame, 15, 80, 80)
        
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 7, 7)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine smooth regions with edges
        oil_effect = cv2.bitwise_and(smooth, edges)
        
        # Enhance saturation for oil painting look
        hsv = cv2.cvtColor(oil_effect, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.3)  # Increase saturation
        oil_effect = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return oil_effect
    
    def retro_crt(self, frame: np.ndarray) -> np.ndarray:
        """Apply retro CRT monitor effect"""
        height, width = frame.shape[:2]
        result = frame.copy()
        
        # Add scanlines
        for y in range(0, height, 3):
            result[y:y+1, :] = result[y:y+1, :] * 0.8
        
        # Add slight green tint
        result[:, :, 1] = np.clip(result[:, :, 1] * 1.1, 0, 255)
        
        # Add vignette effect
        center_x, center_y = width // 2, height // 2
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                vignette_factor = 1.0 - (distance / max_distance) * 0.3
                result[y, x] = result[y, x] * vignette_factor
          # Add slight blur for CRT softness
        result = cv2.GaussianBlur(result, (3, 3), 0)
        
        return result
    
    def neon_glow(self, frame: np.ndarray) -> np.ndarray:
        """Apply neon glow effect"""
        # Edge detection for neon outline
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Create multiple glow layers
        glow_layers = []
        for blur_size in [3, 7, 15, 31]:
            glow = cv2.GaussianBlur(edges, (blur_size, blur_size), 0)
            glow = cv2.cvtColor(glow, cv2.COLOR_GRAY2BGR)
            glow_layers.append(glow)
        
        # Combine glow layers with different colors
        neon_overlay = np.zeros_like(frame, dtype=np.float32)
        colors = [(255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0)]  # Magenta, Cyan, Yellow, Green
        
        for i, (glow, color) in enumerate(zip(glow_layers, colors)):
            colored_glow = np.zeros_like(frame, dtype=np.float32)
            for c in range(3):
                colored_glow[:, :, c] = glow[:, :, 0].astype(np.float32) * (color[c] / 255.0)
            neon_overlay = neon_overlay + colored_glow
        
        # Darken original image
        dark_frame = frame.astype(np.float32) * 0.3
          # Combine with glow
        result = cv2.addWeighted(dark_frame.astype(np.uint8), 0.7, neon_overlay.astype(np.uint8), 0.3, 0)
        
        return result.astype(np.uint8)
    
    def watercolor(self, frame: np.ndarray) -> np.ndarray:
        """Apply watercolor painting effect"""
        # Multiple bilateral filters for watercolor smoothness
        watercolor = frame.copy()
        for _ in range(3):
            watercolor = cv2.bilateralFilter(watercolor, 9, 75, 75)
        
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 7, 7)
        
        # Convert edges to 3-channel
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Reduce edge intensity for watercolor look - ensure proper dtype
        edges = (edges.astype(np.float32) * 0.7).astype(np.uint8)
        
        # Combine watercolor and edges
        result = cv2.bitwise_and(watercolor, edges)
        
        # Slightly desaturate for watercolor effect
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.8  # Reduce saturation
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
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
    
    def ascii_grid(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to ASCII art in a grid pattern using regular characters"""
        char_size = max(4, int(12 - 8 * self.ascii_detail))
        
        ascii_chars, colors, _ = self._frame_to_ascii_array(
            frame, self.ascii_chars_grid, char_size, char_size
        )
        
        # Create blank image
        height, width = frame.shape[:2]
        grid_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Render ASCII characters line by line
        for y, (char_row, color_row) in enumerate(zip(ascii_chars, colors)):
            for x, (char, color) in enumerate(zip(char_row, color_row)):
                if char != ' ':
                    # Calculate position with some padding
                    pos_x = x * char_size + 2
                    pos_y = y * char_size + char_size - 2
                    
                    if pos_y < height and pos_x < width:
                        cv2.putText(grid_image, char, (pos_x, pos_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return grid_image
