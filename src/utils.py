"""
Utility functions for PyArt
Contains helper functions for file operations, image processing, and other utilities
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
import json


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure that a directory exists, create it if it doesn't
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False


def save_image(frame: np.ndarray, directory: str = "saved_images", 
               prefix: str = "pyart_snapshot") -> Optional[str]:
    """
    Save a frame as an image file with timestamp
    
    Args:
        frame: Image frame to save
        directory: Directory to save the image in
        prefix: Prefix for the filename
        
    Returns:
        str: Full path to saved file, or None if failed
    """
    if not ensure_directory_exists(directory):
        return None
    
    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove last 3 digits of microseconds
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(directory, filename)
    
    try:
        success = cv2.imwrite(filepath, frame)
        if success:
            print(f"Snapshot saved: {filepath}")
            return filepath
        else:
            print("Failed to save snapshot")
            return None
    except Exception as e:
        print(f"Error saving snapshot: {e}")
        return None


def load_config(config_path: str = "config.json") -> dict:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    default_config = {
        "camera_index": 0,
        "frame_width": 640,
        "frame_height": 480,
        "default_effect": "original",
        "default_intensity": 1.0,
        "save_directory": "saved_images"
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return {**default_config, **config}
        else:
            return default_config
    except Exception as e:
        print(f"Error loading config: {e}")
        return default_config


def save_config(config: dict, config_path: str = "config.json") -> bool:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary to save
        config_path: Path to save configuration file
        
    Returns:
        bool: True if saved successfully
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def resize_frame_maintain_aspect(frame: np.ndarray, target_width: int, 
                                target_height: int) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio
    
    Args:
        frame: Input frame
        target_width: Target width
        target_height: Target height
        
    Returns:
        np.ndarray: Resized frame
    """
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    target_aspect_ratio = target_width / target_height
    
    if aspect_ratio > target_aspect_ratio:
        # Frame is wider than target
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Frame is taller than target
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    resized = cv2.resize(frame, (new_width, new_height))
    
    # Create target size frame with black padding
    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Center the resized frame
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    result[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    
    return result


def get_frame_fps_info(frame_times: list, max_samples: int = 30) -> Tuple[float, str]:
    """
    Calculate FPS from frame time samples
    
    Args:
        frame_times: List of frame timestamps
        max_samples: Maximum number of samples to keep
        
    Returns:
        tuple: (fps, fps_string)
    """
    if len(frame_times) < 2:
        return 0.0, "FPS: --"
    
    # Keep only recent samples
    if len(frame_times) > max_samples:
        frame_times = frame_times[-max_samples:]
    
    # Calculate average time between frames
    time_diffs = [frame_times[i] - frame_times[i-1] for i in range(1, len(frame_times))]
    avg_time_diff = sum(time_diffs) / len(time_diffs)
    
    if avg_time_diff > 0:
        fps = 1.0 / avg_time_diff
        return fps, f"FPS: {fps:.1f}"
    else:
        return 0.0, "FPS: --"


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between min and max
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        float: Clamped value
    """
    return max(min_val, min(max_val, value))


def interpolate_color(color1: Tuple[int, int, int], color2: Tuple[int, int, int], 
                     factor: float) -> Tuple[int, int, int]:
    """
    Interpolate between two colors
    
    Args:
        color1: First color (B, G, R)
        color2: Second color (B, G, R)
        factor: Interpolation factor (0.0 to 1.0)
        
    Returns:
        tuple: Interpolated color
    """
    factor = clamp(factor, 0.0, 1.0)
    
    b = int(color1[0] * (1 - factor) + color2[0] * factor)
    g = int(color1[1] * (1 - factor) + color2[1] * factor)
    r = int(color1[2] * (1 - factor) + color2[2] * factor)
    
    return (b, g, r)


def check_webcam_availability() -> list:
    """
    Check which webcam indices are available
    
    Returns:
        list: List of available camera indices
    """
    available_cameras = []
    
    # Test up to 5 camera indices
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
        cap.release()
    
    return available_cameras


def print_system_info():
    """Print system and library information"""
    print("PyArt System Information:")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"NumPy Version: {np.__version__}")
    
    # Check available cameras
    cameras = check_webcam_availability()
    print(f"Available Cameras: {cameras if cameras else 'None detected'}")
    print("-" * 40)
