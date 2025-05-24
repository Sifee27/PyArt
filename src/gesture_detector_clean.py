"""
Hand gesture detection module for PyArt
Implements advanced hand gesture recognition using OpenCV with ML enhancement
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from scipy.signal import savgol_filter
import imutils


class HandGestureDetector:
    """Detects hand gestures using OpenCV contour analysis"""
    
    def __init__(self):
        """Initialize the hand gesture detector with ML enhancement"""
        self.background = None
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.gesture_history = []
        self.gesture_threshold = 5  # Number of consecutive frames for gesture confirmation
        self.last_gesture = None
        self.gesture_cooldown = 0
        self.cooldown_frames = 30  # Prevent rapid gesture triggering
        
        # Initialize ML components
        self.gesture_classifier = SVC(kernel='rbf', probability=True)
        self.feature_scaler = StandardScaler()
        self.is_classifier_trained = False
        self.training_data = []
        self.training_labels = []
        
        # Feature history for smoothing
        self.feature_history = []
        self.history_length = 10
        
        # Skin color ranges in HSV (multiple ranges for better detection)
        # Base range for typical skin tones
        self.skin_ranges = [
            # General skin tone range (works for many skin tones)
            {
                'lower': np.array([0, 20, 70], dtype=np.uint8),
                'upper': np.array([20, 255, 255], dtype=np.uint8)
            },
            # Additional range for darker skin tones
            {
                'lower': np.array([0, 10, 60], dtype=np.uint8),
                'upper': np.array([25, 150, 255], dtype=np.uint8)
            },
            # Additional range for lighter skin tones in bright lighting
            {
                'lower': np.array([0, 10, 160], dtype=np.uint8),
                'upper': np.array([25, 60, 255], dtype=np.uint8)
            }
        ]
    
    def detect_skin(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect skin-colored regions in the frame using multiple color ranges with ML enhancement
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Binary mask of skin regions with noise reduction
        """
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur to reduce noise before skin detection
        hsv_smooth = cv2.GaussianBlur(hsv, (7, 7), 0)
        
        # Create a combined mask from all skin color ranges
        combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        for skin_range in self.skin_ranges:
            # Create mask for this skin color range
            mask = cv2.inRange(hsv_smooth, skin_range['lower'], skin_range['upper'])
            # Combine with previous masks
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)  # Larger kernel for better noise removal
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply Gaussian blur to smooth the mask
        combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
        
        # Use scipy for additional noise reduction and hole filling
        combined_mask = ndimage.binary_fill_holes(combined_mask).astype(np.uint8) * 255
        
        # Apply adaptive thresholding to improve detection in varying lighting
        combined_mask = cv2.adaptiveThreshold(
            combined_mask, 
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        
        # Filter small noise blobs
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(combined_mask)
        min_contour_area = 1000  # Minimum size of hand
        for contour in contours:
            if cv2.contourArea(contour) > min_contour_area:
                cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
        
        return filtered_mask
    
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
        finger_points = []
        
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
            if b * c == 0:
                continue
                
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            
            # If angle is less than 90 degrees, count as a defect (finger gap)
            if angle <= np.pi / 2:
                defect_count += 1
                finger_points.append(far)
        
        # Determine gesture based on defect count and contour properties
        contour_area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        
        # Calculate solidity (how "solid" the shape is)
        if hull_area > 0:
            solidity = contour_area / hull_area
        else:
            solidity = 0
            
        # Get bounding rectangle and minimum area rectangle
        x, y, w, h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)
        
        # Calculate aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Calculate moments for centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0
            
        # Improved fist detection
        if defect_count <= 1 and solidity > 0.85:
            # A fist is very "solid" with few convexity defects
            return 'fist'
            
        # Enhanced thumbs up/down detection with multiple criteria
        elif defect_count >= 1 and defect_count <= 3:
            # Calculate the angle of the bounding rectangle
            angle = rect[2]
            if angle < -45:
                angle += 90
                
            # Multiple criteria for thumbs detection
            criteria_met = 0
            
            # Criterion 1: Aspect ratio check (thumbs are usually taller than wide)
            if aspect_ratio < 0.8:  # Height > Width * 0.8
                criteria_met += 1
            
            # Criterion 2: Solidity check (thumbs are less solid than fists)
            if 0.6 < solidity < 0.9:
                criteria_met += 1
            
            # Criterion 3: Orientation check (hand should be roughly vertical)
            if abs(angle) < 35:  # Allow more flexibility in orientation
                criteria_met += 1
            
            # Criterion 4: Size check (thumbs have moderate area)
            if 3000 < contour_area < 15000:
                criteria_met += 1
            
            # Need at least 3 out of 4 criteria for thumbs detection
            if criteria_met >= 3:
                # Now determine up vs down using multiple methods
                
                # Method 1: Centroid position relative to bounding box
                relative_cy = (cy - y) / h if h > 0 else 0.5
                centroid_vote = 0
                if relative_cy < 0.45:  # Centroid in upper half
                    centroid_vote = 1  # Vote for thumbs up
                elif relative_cy > 0.55:  # Centroid in lower half
                    centroid_vote = -1  # Vote for thumbs down
                
                # Method 2: Find the topmost and bottommost points
                topmost = tuple(contour[contour[:, :, 1].argmin()][0])
                bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
                
                # Calculate distances from centroid to top and bottom
                dist_to_top = np.sqrt((cx - topmost[0])**2 + (cy - topmost[1])**2)
                dist_to_bottom = np.sqrt((cx - bottommost[0])**2 + (cy - bottommost[1])**2)
                
                extremity_vote = 0
                if dist_to_bottom > dist_to_top * 1.2:  # More mass towards bottom
                    extremity_vote = 1  # Vote for thumbs up
                elif dist_to_top > dist_to_bottom * 1.2:  # More mass towards top
                    extremity_vote = -1  # Vote for thumbs down
                
                # Method 3: Check convexity defect positions
                defect_vote = 0
                if defects is not None and len(defects) > 0:
                    avg_defect_y = 0
                    valid_defects = 0
                    
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        far = tuple(contour[f][0])
                        
                        # Only count significant defects
                        if d > 1000:  # Distance threshold
                            avg_defect_y += far[1]
                            valid_defects += 1
                    
                    if valid_defects > 0:
                        avg_defect_y /= valid_defects
                        relative_defect_y = (avg_defect_y - y) / h if h > 0 else 0.5
                        
                        if relative_defect_y < 0.4:  # Defects in upper part
                            defect_vote = -1  # Vote for thumbs down
                        elif relative_defect_y > 0.6:  # Defects in lower part
                            defect_vote = 1  # Vote for thumbs up
                
                # Combine votes with weights
                total_vote = centroid_vote * 2 + extremity_vote * 1.5 + defect_vote * 1
                
                # Make final decision
                if total_vote > 0.5:
                    return 'thumbs_up'
                elif total_vote < -0.5:
                    return 'thumbs_down'
        
        return 'unknown'
    
    def detect_gesture(self, frame: np.ndarray) -> Optional[str]:
        """
        Detect hand gesture in the frame with improved stability
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Detected gesture or None
        """
        # Reduce cooldown
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return None
        
        # Get frame dimensions for ROI
        height, width = frame.shape[:2]
        
        # Define region of interest (center portion of the frame)
        # This helps focus on the hand and ignore background
        roi_margin = width // 6  # Margin from edges
        roi = frame[roi_margin:height-roi_margin, roi_margin:width-roi_margin]
        
        if roi.size == 0:  # If ROI is empty, use whole frame
            roi = frame
        
        # Detect skin regions
        skin_mask = self.detect_skin(roi)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self.gesture_history.clear()
            return None
        
        # Find valid hand contours (filter by area)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 3000]
        if not valid_contours:
            self.gesture_history.clear()
            return None
        
        # Find the largest contour (assume it's the hand)
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Analyze the contour using ML-enhanced classification
        gesture = self.classify_gesture_ml(largest_contour)
        
        # Add to history for stability with weighted voting
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.gesture_threshold:
            self.gesture_history.pop(0)
        
        # Check if we have enough history for gesture confirmation
        if len(self.gesture_history) == self.gesture_threshold:
            # Use weighted voting (recent gestures have more weight)
            gesture_votes = {}
            for i, g in enumerate(self.gesture_history):
                weight = (i + 1) / len(self.gesture_history)  # Higher weight for more recent gestures
                if g not in gesture_votes:
                    gesture_votes[g] = 0
                gesture_votes[g] += weight
            
            # Find the gesture with the highest weighted vote
            best_gesture = max(gesture_votes.items(), key=lambda x: x[1])[0]
            
            # Only confirm if the best gesture has a strong majority
            total_votes = sum(gesture_votes.values())
            if gesture_votes[best_gesture] / total_votes > 0.6 and best_gesture != 'unknown':
                if best_gesture != self.last_gesture:
                    self.last_gesture = best_gesture
                    self.gesture_cooldown = self.cooldown_frames
                    self.gesture_history.clear()
                    return best_gesture
        
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
        
        # Draw skin mask as semi-transparent overlay
        skin_overlay = np.zeros_like(debug_frame)
        skin_overlay[skin_mask > 0] = [0, 255, 0]  # Green overlay
        debug_frame = cv2.addWeighted(debug_frame, 1.0, skin_overlay, 0.3, 0)
        
        if contours:
            # Find the largest contour (assume it's the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Draw contour
            cv2.drawContours(debug_frame, [largest_contour], -1, (255, 0, 0), 2)
            
            # Draw convex hull
            hull = cv2.convexHull(largest_contour)
            cv2.drawContours(debug_frame, [hull], -1, (0, 0, 255), 2)
            
            # Draw bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            
            # Draw minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)
            cv2.drawContours(debug_frame, [box], 0, (0, 255, 255), 2)
            
            # Draw convexity defects
            try:
                hull_indices = cv2.convexHull(largest_contour, returnPoints=False)
                defects = cv2.convexityDefects(largest_contour, hull_indices)
                
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(largest_contour[s][0])
                        end = tuple(largest_contour[e][0])
                        far = tuple(largest_contour[f][0])
                        
                        # Draw defect points and lines
                        cv2.circle(debug_frame, far, 5, [0, 0, 255], -1)  # Red circle at defect point
                        cv2.line(debug_frame, start, end, [255, 0, 0], 2)  # Blue line for defect edge
            except:
                pass
            
            # Draw centroid
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(debug_frame, (cx, cy), 10, (255, 0, 255), -1)  # Magenta circle at centroid
                
                # Draw vertical dividers for thumbs up/down detection
                cv2.line(debug_frame, (x, y+int(h*0.4)), (x+w, y+int(h*0.4)), (0, 255, 255), 1)  # Upper threshold
                cv2.line(debug_frame, (x, y+int(h*0.6)), (x+w, y+int(h*0.6)), (0, 255, 255), 1)  # Lower threshold
        
        # Draw detailed status information
        gesture = "None" if self.last_gesture is None else self.last_gesture
        
        # Create status information
        status_text = f"Last Gesture: {gesture}"
        cv2.putText(debug_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw help text
        help_text = "Gestures: Thumbs Up = +Detail, Thumbs Down = -Detail, Fist = Capture"
        cv2.putText(debug_frame, help_text, (10, debug_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return debug_frame
    
    def extract_hand_features(self, contour: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive features from hand contour for ML classification
        
        Args:
            contour: OpenCV contour of the hand
            
        Returns:
            Feature vector for machine learning
        """
        features = []
        
        # Basic geometric features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Hu moments (7 invariant moments)
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments)
        
        # Bounding rectangle features
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0
        
        # Convex hull features
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Convexity defects
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_indices)
        defect_count = 0
        max_defect_depth = 0
        avg_defect_depth = 0
        
        if defects is not None:
            valid_defects = []
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                depth = d / 256.0  # Convert to actual distance
                if depth > 10:  # Filter small defects
                    valid_defects.append(depth)
                    if depth > max_defect_depth:
                        max_defect_depth = depth
            
            defect_count = len(valid_defects)
            avg_defect_depth = np.mean(valid_defects) if valid_defects else 0
        
        # Minimum enclosing circle
        (_, _), radius = cv2.minEnclosingCircle(contour)
        circularity = area / (np.pi * radius * radius) if radius > 0 else 0
        
        # Ellipse fitting
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipse_area = np.pi * ellipse[1][0] * ellipse[1][1] / 4
            ellipse_ratio = float(ellipse[1][0]) / ellipse[1][1] if ellipse[1][1] > 0 else 0
        else:
            ellipse_area = 0
            ellipse_ratio = 0
        
        # Compile features
        features.extend([
            area,
            perimeter,
            aspect_ratio,
            extent,
            solidity,
            defect_count,
            max_defect_depth,
            avg_defect_depth,
            circularity,
            ellipse_ratio,
            w,  # width
            h,  # height
        ])
        
        # Add Hu moments (log transformed for better stability)
        for hu in hu_moments.flatten():
            if hu != 0:
                features.append(-np.sign(hu) * np.log10(abs(hu)))
            else:
                features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def smooth_features(self, features: np.ndarray) -> np.ndarray:
        """
        Smooth features using signal processing techniques
        
        Args:
            features: Raw feature vector
            
        Returns:
            Smoothed feature vector
        """
        # Add to history
        self.feature_history.append(features)
        if len(self.feature_history) > self.history_length:
            self.feature_history.pop(0)
        
        # Apply smoothing if we have enough history
        if len(self.feature_history) >= 5:
            feature_array = np.array(self.feature_history)
            
            # Apply Savitzky-Golay filter for smoothing
            try:
                smoothed = savgol_filter(feature_array, window_length=5, polyorder=2, axis=0)
                return smoothed[-1]  # Return the latest smoothed features
            except:
                return features  # Fallback to original if smoothing fails
        
        return features
    
    def classify_gesture_ml(self, contour: np.ndarray) -> str:
        """
        Classify gesture using machine learning if trained, otherwise fall back to rule-based
        
        Args:
            contour: Hand contour
            
        Returns:
            Classified gesture
        """
        # Extract features
        features = self.extract_hand_features(contour)
        
        # Smooth features
        features = self.smooth_features(features)
        
        # Use ML classifier if available and trained
        if self.is_classifier_trained and len(self.training_data) > 10:
            try:
                features_scaled = self.feature_scaler.transform([features])
                probabilities = self.gesture_classifier.predict_proba(features_scaled)[0]
                prediction = self.gesture_classifier.predict(features_scaled)[0]
                
                # Only accept prediction if confidence is high enough
                max_prob = np.max(probabilities)
                if max_prob > 0.7:  # High confidence threshold
                    return prediction
            except:
                pass  # Fall back to rule-based if ML fails
        
        # Fall back to rule-based classification
        return self.analyze_contour(contour)
    
    def train_classifier(self, gesture_label: str, contour: np.ndarray):
        """
        Train the ML classifier with new gesture data
        
        Args:
            gesture_label: The label for this gesture
            contour: The hand contour for this gesture
        """
        features = self.extract_hand_features(contour)
        self.training_data.append(features)
        self.training_labels.append(gesture_label)
        
        # Train classifier if we have enough data
        if len(self.training_data) >= 20 and len(set(self.training_labels)) >= 2:
            try:
                # Prepare training data
                X = np.array(self.training_data)
                y = np.array(self.training_labels)
                
                # Scale features
                self.feature_scaler.fit(X)
                X_scaled = self.feature_scaler.transform(X)
                
                # Train classifier
                self.gesture_classifier.fit(X_scaled, y)
                self.is_classifier_trained = True
                
                print(f"Classifier trained with {len(self.training_data)} samples")
            except Exception as e:
                print(f"Training failed: {e}")
