"""
Camera module for PyArt
Handles webcam capture and video stream management
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class Camera:
    """Handles webcam capture and video stream management"""
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Initialize camera capture
        
        Args:
            camera_index: Index of the camera to use (default: 0)
            width: Frame width (default: 640)
            height: Frame height (default: 480)
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize the camera capture
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_initialized = True
            print(f"Camera initialized successfully: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a frame from the camera
        
        Returns:
            numpy.ndarray: Frame data if successful, None otherwise
        """
        if not self.is_initialized or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        return frame
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        Get the current frame size
        
        Returns:
            tuple: (width, height) of frames
        """
        return (self.width, self.height)
    
    def release(self):
        """Release the camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.is_initialized = False
            print("Camera resources released")
    
    def is_available(self) -> bool:
        """Check if camera is available and working"""
        return self.is_initialized and self.cap is not None and self.cap.isOpened()
    
    def __del__(self):
        """Destructor to ensure camera is released"""
        self.release()
