"""
Face Filters module for PyArt
Implements face detection and AR filters
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Optional

class FaceFilterProcessor:
    """Handles face detection and applies AR filters to faces"""
    
    def __init__(self):
        """Initialize face detection and AR filters"""
        # Initialize the face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Load filter images
        self.filters = {}
        self._load_filters()
        
        # Tracking variables
        self.current_filter = None
        self.last_faces = []  # For smoothing
        self.smooth_factor = 0.3  # Lower = smoother tracking
        
    def _load_filters(self):
        """Load filter images from assets directory"""
        # Create assets/filters directory if it doesn't exist
        filter_dir = os.path.join("assets", "filters")
        os.makedirs(filter_dir, exist_ok=True)
        
        # Sample filters - predefined as numpy arrays
        # Each filter is a dictionary with the image, anchor point, and scaling
        
        # Create a simple sunglasses filter (procedurally)
        sunglasses = np.zeros((100, 200, 4), dtype=np.uint8)
        cv2.rectangle(sunglasses, (10, 30), (190, 70), (0, 0, 0, 255), -1)  # Left lens
        cv2.rectangle(sunglasses, (0, 40), (200, 60), (50, 50, 50, 255), -1)  # Bridge
        self.filters["sunglasses"] = {
            "image": sunglasses,
            "anchor": "eyes",  # Where to place the filter
            "scale_factor": 1.2,  # Size relative to the face
            "y_offset": -0.1,  # Vertical offset (as ratio of face height)
            "x_offset": 0.0   # Horizontal offset (as ratio of face width)
        }
        
        # Create a simple hat filter (procedurally)
        hat = np.zeros((150, 250, 4), dtype=np.uint8)
        cv2.rectangle(hat, (25, 100), (225, 140), (51, 51, 153, 255), -1)  # Brim
        cv2.rectangle(hat, (50, 20), (200, 100), (102, 102, 204, 255), -1)  # Top
        self.filters["hat"] = {
            "image": hat,
            "anchor": "forehead",
            "scale_factor": 1.5,
            "y_offset": -1.0,
            "x_offset": 0.0
        }
        
        # Create a simple mustache filter (procedurally)
        mustache = np.zeros((50, 150, 4), dtype=np.uint8)
        points = np.array([
            [30, 20], [75, 45], [120, 20],  # Top curve
            [120, 30], [75, 55], [30, 30]   # Bottom curve
        ], np.int32)
        points = points.reshape((-1, 1, 2))
        cv2.fillPoly(mustache, [points], (30, 30, 30, 255))
        self.filters["mustache"] = {
            "image": mustache,
            "anchor": "nose",
            "scale_factor": 0.8,
            "y_offset": 0.4,
            "x_offset": 0.0
        }
        
        # Create a simple cat ears filter (procedurally)
        cat_ears = np.zeros((150, 250, 4), dtype=np.uint8)
        # Left ear
        points_left = np.array([
            [40, 80], [60, 20], [80, 80]
        ], np.int32)
        points_left = points_left.reshape((-1, 1, 2))
        cv2.fillPoly(cat_ears, [points_left], (30, 30, 30, 255))
        # Right ear
        points_right = np.array([
            [170, 80], [190, 20], [210, 80]
        ], np.int32)
        points_right = points_right.reshape((-1, 1, 2))
        cv2.fillPoly(cat_ears, [points_right], (30, 30, 30, 255))
        self.filters["cat_ears"] = {
            "image": cat_ears,
            "anchor": "forehead",
            "scale_factor": 1.5,
            "y_offset": -0.9,
            "x_offset": 0.0
        }
        
        # Create a clown nose
        clown_nose = np.zeros((60, 60, 4), dtype=np.uint8)
        cv2.circle(clown_nose, (30, 30), 25, (0, 0, 255, 255), -1)
        self.filters["clown_nose"] = {
            "image": clown_nose,
            "anchor": "nose",
            "scale_factor": 0.5,
            "y_offset": 0.0,
            "x_offset": 0.0
        }
        
    def get_filter_names(self) -> List[str]:
        """Get names of available filters"""
        return list(self.filters.keys())
    
    def set_filter(self, filter_name: Optional[str]):
        """Set the current filter"""
        if filter_name is None or filter_name not in self.filters:
            self.current_filter = None
        else:
            self.current_filter = filter_name
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in the frame
        
        Args:
            frame: Input video frame
            
        Returns:
            List of dictionaries containing face information
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        result = []
        for (x, y, w, h) in faces:
            face_info = {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "eyes": [],
                "nose": (x + w//2, y + int(h * 0.6)),  # Approximate nose position
                "forehead": (x + w//2, y + int(h * 0.2))  # Approximate forehead position
            }
            
            # Detect eyes in the face region
            roi_gray = gray[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                eye_center = (x + ex + ew//2, y + ey + eh//2)
                face_info["eyes"].append(eye_center)
            
            # If we have both eyes, calculate their midpoint
            if len(face_info["eyes"]) >= 2:
                eye_points = np.array(face_info["eyes"])
                eye_midpoint = np.mean(eye_points, axis=0).astype(int)
                face_info["eyes_midpoint"] = tuple(eye_midpoint)
            
            result.append(face_info)
        
        # Apply temporal smoothing for stability if we have previous faces
        if self.last_faces and result:
            # Try to match current faces with previous faces
            for i, curr_face in enumerate(result):
                # Find the closest previous face
                min_dist = float('inf')
                closest_idx = -1
                
                curr_center = (curr_face["x"] + curr_face["width"]//2, 
                               curr_face["y"] + curr_face["height"]//2)
                
                for j, prev_face in enumerate(self.last_faces):
                    prev_center = (prev_face["x"] + prev_face["width"]//2, 
                                   prev_face["y"] + prev_face["height"]//2)
                    
                    dist = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                   (curr_center[1] - prev_center[1])**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = j
                
                # If we found a matching face within a reasonable distance, smooth it
                if closest_idx >= 0 and min_dist < 100:  # Arbitrary threshold
                    prev_face = self.last_faces[closest_idx]
                    
                    # Smooth the positions
                    result[i]["x"] = int((1 - self.smooth_factor) * prev_face["x"] + 
                                         self.smooth_factor * curr_face["x"])
                    result[i]["y"] = int((1 - self.smooth_factor) * prev_face["y"] + 
                                         self.smooth_factor * curr_face["y"])
                    result[i]["width"] = int((1 - self.smooth_factor) * prev_face["width"] + 
                                            self.smooth_factor * curr_face["width"])
                    result[i]["height"] = int((1 - self.smooth_factor) * prev_face["height"] + 
                                             self.smooth_factor * curr_face["height"])
                    
                    # Recalculate derived positions
                    result[i]["nose"] = (result[i]["x"] + result[i]["width"]//2, 
                                        result[i]["y"] + int(result[i]["height"] * 0.6))
                    result[i]["forehead"] = (result[i]["x"] + result[i]["width"]//2, 
                                            result[i]["y"] + int(result[i]["height"] * 0.2))
        
        # Save current faces for next frame
        self.last_faces = result.copy()
        
        return result
    
    def _overlay_image(self, 
                      background: np.ndarray, 
                      foreground: np.ndarray, 
                      position: Tuple[int, int]):
        """
        Overlay foreground image on background with alpha blending
        
        Args:
            background: Background image
            foreground: Foreground image with alpha channel
            position: Top-left position to place foreground (x, y)
        
        Returns:
            Background with foreground overlaid
        """
        fg_h, fg_w = foreground.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        # Ensure the overlay is within the frame bounds
        x, y = position
        if x < 0: x = 0
        if y < 0: y = 0
        
        # Calculate the overlay region
        w = min(fg_w, bg_w - x)
        h = min(fg_h, bg_h - y)
        
        if w <= 0 or h <= 0:
            return background
        
        # Extract the alpha channel
        alpha = foreground[:h, :w, 3] / 255.0
        alpha = np.dstack((alpha, alpha, alpha))
        
        # Get the foreground and background regions
        fg = foreground[:h, :w, :3]
        bg_roi = background[y:y+h, x:x+w]
        
        # Composite the foreground onto the background
        composite = bg_roi * (1 - alpha) + fg * alpha
        
        # Update the background region
        background[y:y+h, x:x+w] = composite
        
        return background
    
    def apply_filter(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply the current filter to detected faces
        
        Args:
            frame: Input video frame
            
        Returns:
            Frame with filter applied
        """
        if self.current_filter is None:
            return frame
        
        # Get filter details
        filter_data = self.filters[self.current_filter]
        filter_img = filter_data["image"]
        anchor_type = filter_data["anchor"]
        scale_factor = filter_data["scale_factor"]
        y_offset = filter_data["y_offset"]
        x_offset = filter_data["x_offset"]
        
        # Create a copy of the frame
        result = frame.copy()
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Apply filter to each detected face
        for face in faces:
            face_width = face["width"]
            face_height = face["height"]
            
            # Calculate filter size relative to face
            filter_width = int(face_width * scale_factor)
            filter_height = int(filter_img.shape[0] * (filter_width / filter_img.shape[1]))
            
            # Resize filter image
            resized_filter = cv2.resize(filter_img, (filter_width, filter_height))
            
            # Determine anchor position
            if anchor_type == "eyes" and "eyes_midpoint" in face:
                anchor_x, anchor_y = face["eyes_midpoint"]
            elif anchor_type == "nose":
                anchor_x, anchor_y = face["nose"]
            elif anchor_type == "forehead":
                anchor_x, anchor_y = face["forehead"]
            else:
                # Default to center of face
                anchor_x = face["x"] + face_width // 2
                anchor_y = face["y"] + face_height // 2
            
            # Apply offsets
            offset_x = int(x_offset * face_width)
            offset_y = int(y_offset * face_height)
            
            # Calculate top-left position
            pos_x = anchor_x - filter_width // 2 + offset_x
            pos_y = anchor_y - filter_height // 2 + offset_y
            
            # Overlay the filter on the frame        result = self._overlay_image(result, resized_filter, (pos_x, pos_y))
        
        return result
