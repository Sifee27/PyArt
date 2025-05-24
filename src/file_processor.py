"""
File processing module for PyArt
Handles loading and processing of image and video files
"""

import os
import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Optional, Generator


class FileProcessor:
    """Processes image and video files for PyArt"""
    
    def __init__(self, effect_processor=None):
        """Initialize the file processor"""
        self.effect_processor = effect_processor
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    def get_supported_formats(self) -> dict:
        """Get dictionary of supported file formats"""
        return {
            'image': self.supported_image_formats,
            'video': self.supported_video_formats
        }
    
    def is_supported_image(self, file_path: str) -> bool:
        """Check if file is a supported image format"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_image_formats
    
    def is_supported_video(self, file_path: str) -> bool:
        """Check if file is a supported video format"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_video_formats
    
    def load_image(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load image from file
        
        Args:
            file_path: Path to image file
            
        Returns:
            Loaded image as numpy array or None if failed
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return None
            
        if not self.is_supported_image(file_path):
            print(f"Error: Unsupported image format - {file_path}")
            return None
            
        try:
            img = cv2.imread(file_path)
            if img is None:
                print(f"Error: Failed to load image - {file_path}")
                return None
                
            # Convert to RGB if needed (OpenCV loads as BGR)
            return img
            
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None
    
    def get_video_info(self, file_path: str) -> Optional[dict]:
        """
        Get video file information
        
        Args:
            file_path: Path to video file
            
        Returns:
            Dictionary with video properties or None if failed
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return None
            
        if not self.is_supported_video(file_path):
            print(f"Error: Unsupported video format - {file_path}")
            return None
            
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"Error: Could not open video - {file_path}")
                return None
                
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'file_path': file_path,
                'frame_count': frame_count,
                'fps': fps,
                'width': width,
                'height': height,
                'duration': duration  # in seconds
            }
            
        except Exception as e:
            print(f"Error getting video info for {file_path}: {e}")
            return None
    
    def process_image(self, image: np.ndarray, effect_name: str = 'ascii_blocks') -> np.ndarray:
        """
        Process image with specified effect
        
        Args:
            image: Input image as numpy array
            effect_name: Name of effect to apply
            
        Returns:
            Processed image
        """
        if self.effect_processor is None:
            print("Error: No effect processor available")
            return image
            
        return self.effect_processor.apply_effect(image, effect_name)
    
    def process_image_file(self, file_path: str, effect_name: str = 'ascii_blocks', 
                          save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Load and process an image file
        
        Args:
            file_path: Path to image file
            effect_name: Name of effect to apply
            save_path: Optional path to save processed image
            
        Returns:
            Processed image or None if failed
        """
        # Load the image
        image = self.load_image(file_path)
        if image is None:
            return None
            
        # Process the image
        processed = self.process_image(image, effect_name)
        
        # Save if requested
        if save_path and processed is not None:
            try:
                directory = os.path.dirname(save_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)
                cv2.imwrite(save_path, processed)
                print(f"Saved processed image to {save_path}")
            except Exception as e:
                print(f"Error saving processed image: {e}")
        
        return processed
    
    def video_frame_generator(self, file_path: str, max_frames: int = 0) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames from a video file
        
        Args:
            file_path: Path to video file
            max_frames: Maximum number of frames to yield (0 = all)
            
        Yields:
            Video frames as numpy arrays
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return
            
        if not self.is_supported_video(file_path):
            print(f"Error: Unsupported video format - {file_path}")
            return
            
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"Error: Could not open video - {file_path}")
                return
                
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                yield frame
                
                frame_count += 1
                if max_frames > 0 and frame_count >= max_frames:
                    break
                    
            cap.release()
            
        except Exception as e:
            print(f"Error reading video {file_path}: {e}")
            return
    
    def process_video(self, file_path: str, effect_name: str = 'ascii_blocks', 
                     output_path: Optional[str] = None, preview_only: bool = False) -> bool:
        """
        Process a video file with the specified effect
        
        Args:
            file_path: Path to video file
            effect_name: Name of effect to apply
            output_path: Path to save processed video (or auto-generated if None)
            preview_only: If True, only returns the first processed frame instead of processing the whole video
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return False
            
        if not self.is_supported_video(file_path):
            print(f"Error: Unsupported video format - {file_path}")
            return False
            
        if self.effect_processor is None:
            print("Error: No effect processor available")
            return False
            
        try:
            # Open the video
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"Error: Could not open video - {file_path}")
                return False
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Preview mode - just process the first frame
            if preview_only:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from video")
                    cap.release()
                    return False
                
                processed_frame = self.effect_processor.apply_effect(frame, effect_name)
                cap.release()
                
                # Save preview if output path is specified
                if output_path:
                    preview_path = f"{os.path.splitext(output_path)[0]}_preview.png"
                    cv2.imwrite(preview_path, processed_frame)
                    print(f"Saved video preview to {preview_path}")
                
                return True
            
            # Full processing mode
            if output_path is None:
                # Auto-generate output path
                base_path = os.path.splitext(file_path)[0]
                output_path = f"{base_path}_{effect_name}.mp4"
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process each frame
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process the frame
                processed = self.effect_processor.apply_effect(frame, effect_name)
                out.write(processed)
                
                frame_count += 1
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_processing = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Processed {frame_count} frames ({fps_processing:.2f} fps)")
            
            # Clean up
            cap.release()
            out.release()
            
            print(f"Video processing complete: {output_path}")
            print(f"Processed {frame_count} frames in {time.time() - start_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"Error processing video {file_path}: {e}")
            return False
