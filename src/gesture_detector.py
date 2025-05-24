"""
Hand gesture detection module for PyArt
Implements hand gesture recognition using OpenCV without MediaPipe
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class HandGestureDetector:
    """Detects hand gestures using OpenCV contour analysis"""
    
    def __init__(self):
        """Initialize the hand gesture detector"""
        self.background = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.gesture_history = []
        self.gesture_threshold = 5  # Number of consecutive frames for gesture confirmation
        self.last_gesture = None
        self.gesture_cooldown = 0
        self.cooldown_frames = 30  # Prevent rapid gesture triggering
        
        # Skin color range in HSV
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
    def detect_skin(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect skin-colored regions in the frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary mask of skin regions
        """
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply Gaussian blur to smooth the mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        return mask
    
    def analyze_contour(self, contour: np.ndarray) -> str:
        """
        Analyze a contour to determine gesture type
        
        Args:
            contour: OpenCV contour
            
        Returns:
            Gesture type: 'thumbs_up', 'thumbs_down', 'fist', or 'unknown'
        """
        if cv2.contourArea(contour) < 3000:  # Too small to be a hand
            return 'unknown'
        
        # Calculate convex hull and defects
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) < 4:
            return 'unknown'
            
        try:
            defects = cv2.convexityDefects(contour, hull)
            if defects is None:
                return 'fist'  # No defects usually means closed hand
        except:
            return 'unknown'
        
        # Count significant defects (fingers)
        defect_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Calculate the distance from the far point to the hull
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            
            # Calculate angle using cosine rule
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            
            # If angle is less than 90 degrees, count as a defect (finger gap)
            if angle <= np.pi / 2:
                defect_count += 1
        
        # Determine gesture based on defect count and contour properties
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        
        # Calculate solidity (how "solid" the shape is)
        if hull_area > 0:
            solidity = contour_area / hull_area
        else:
            solidity = 0
        
        # Gesture classification
        if defect_count == 0 and solidity > 0.8:
            return 'fist'
        elif defect_count == 1:
            # Could be thumbs up or thumbs down, check orientation
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Calculate the angle of the bounding rectangle
            angle = rect[2]
            if angle < -45:
                angle += 90
            
            # Simple heuristic: if the hand is more vertical, it might be a thumb gesture
            if abs(angle) < 30:  # Roughly vertical
                # Check if the centroid is in the upper or lower part of the bounding box
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Get the bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # If centroid is in upper half, might be thumbs up
                    if cy < y + h * 0.6:
                        return 'thumbs_up'
                    else:
                        return 'thumbs_down'
        
        return 'unknown'
    
    def detect_gesture(self, frame: np.ndarray) -> Optional[str]:
        """
        Detect hand gesture in the frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Detected gesture or None
        """
        # Reduce cooldown
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return None
        
        # Detect skin regions
        skin_mask = self.detect_skin(frame)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.gesture_history.clear()
            return None
        
        # Find the largest contour (assume it's the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Analyze the contour
        gesture = self.analyze_contour(largest_contour)
        
        # Add to history for stability
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.gesture_threshold:
            self.gesture_history.pop(0)
        
        # Check if we have consistent gesture detection
        if len(self.gesture_history) == self.gesture_threshold:
            if all(g == gesture for g in self.gesture_history):
                if gesture != self.last_gesture and gesture != 'unknown':
                    self.last_gesture = gesture
                    self.gesture_cooldown = self.cooldown_frames
                    self.gesture_history.clear()
                    return gesture
        
        return None
    
    def draw_debug_info(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw debug information on the frame
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with debug overlay
        """
        debug_frame = frame.copy()
        
        # Detect skin and contours for visualization
        skin_mask = self.detect_skin(frame)
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        cv2.drawContours(debug_frame, contours, -1, (0, 255, 0), 2)
        
        # Draw largest contour in different color
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(debug_frame, [largest_contour], -1, (255, 0, 0), 3)
            
            # Draw convex hull
            hull = cv2.convexHull(largest_contour)
            cv2.drawContours(debug_frame, [hull], -1, (0, 0, 255), 2)
        
        # Draw status
        status_text = f"Last: {self.last_gesture or 'None'}"
        cv2.putText(debug_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_frame
