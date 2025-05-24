"""
PyArt - Interactive Webcam Art Application
Main application entry point

This is the main file that orchestrates all components of PyArt:
- Camera capture
- Effect processing
- User interface
- Input handling
"""

import sys
import time
import cv2
import numpy as np
from src.camera import Camera
from src.effects import EffectProcessor
from src.ui import UserInterface
from src.gesture_detector import HandGestureDetector
from src.utils import (
    save_image, load_config, save_config, print_system_info,
    get_frame_fps_info, clamp
)


class PyArtApp:
    """Main PyArt application class"""
    
    def __init__(self):
        """Initialize the PyArt application"""
        print("Initializing PyArt...")
        print_system_info()
        
        # Load configuration
        self.config = load_config()
          # Initialize components
        self.camera = Camera(
            camera_index=self.config['camera_index'],
            width=self.config['frame_width'],
            height=self.config['frame_height']
        )
        self.effect_processor = EffectProcessor()
        self.ui = UserInterface()
        self.gesture_detector = HandGestureDetector()
        
        # Application state
        self.current_effect_index = 0
        self.effect_names = self.effect_processor.get_effect_names()
        self.current_effect = self.config.get('default_effect', 'original')
        self.intensity = self.config.get('default_intensity', 1.0)
        self.running = False
        self.gesture_mode = True  # Enable gesture control
        self.show_debug = False   # Show gesture debug overlay
        
        # ASCII detail control
        self.ascii_detail_level = 1.0  # 0.0 = very pixelated, 2.0 = very detailed
          # Performance tracking
        self.frame_times = []
        self.last_frame_time = time.time()
        
        # Set initial effect index
        if self.current_effect in self.effect_names:
            self.current_effect_index = self.effect_names.index(self.current_effect)
        
        self.effect_processor.set_intensity(self.intensity)
    
    def initialize(self) -> bool:
        """
        Initialize all components
        
        Returns:
            bool: True if initialization successful
        """
        # Initialize camera
        if not self.camera.initialize():
            print("Failed to initialize camera. Please check your webcam connection.")
            return False
        
        # Create UI window
        self.ui.create_window()
        
        print("PyArt initialized successfully!")
        print("Press 'H' for help, 'Q' to quit")
        return True
    
    def run(self):
        """Main application loop"""
        if not self.initialize():
            return
        
        self.running = True
        print("Starting PyArt... Press 'Q' to quit")
        while self.running:
            # Capture frame
            frame = self.camera.read_frame()
            if frame is None:
                print("Failed to capture frame")
                break
            
            # Handle gesture detection if enabled
            if self.gesture_mode:
                gesture = self.gesture_detector.detect_gesture(frame)
                if gesture:
                    self.handle_gesture(gesture, frame)
            
            # Set ASCII detail level for effects processor
            self.effect_processor.set_ascii_detail(self.ascii_detail_level)
            
            # Apply current effect
            processed_frame = self.effect_processor.apply_effect(frame, self.current_effect)
            
            # Add debug overlay if enabled
            if self.show_debug and self.gesture_mode:
                processed_frame = self.gesture_detector.draw_debug_info(processed_frame)
            
            # Update performance tracking
            current_time = time.time()
            self.frame_times.append(current_time)
            
            # Display frame with UI
            self.ui.display_frame(
                processed_frame, 
                self.current_effect, 
                self.intensity,
                self.effect_names,
                gesture_mode=self.gesture_mode,
                ascii_detail=self.ascii_detail_level
            )
              # Handle user input
            action = self.ui.handle_input()
            if action:
                self.handle_action(action, processed_frame)
            
            # Update frame time
            self.last_frame_time = current_time
        
        self.cleanup()
    
    def handle_action(self, action: str, current_frame):
        """
        Handle user input actions
        
        Args:
            action: Action command string
            current_frame: Current processed frame
        """
        if action == 'quit':
            self.running = False
        
        elif action == 'cycle_effect':
            self.cycle_to_next_effect()
        
        elif action == 'save_snapshot':
            self.save_snapshot(current_frame)
        
        elif action == 'increase_intensity':
            self.adjust_intensity(0.1)
        
        elif action == 'decrease_intensity':
            self.adjust_intensity(-0.1)
        
        elif action == 'decrease_ascii_detail':
            # Decrease ASCII detail (more pixelated)
            self.ascii_detail_level = clamp(self.ascii_detail_level - 0.3, 0.0, 2.0)
            print(f"ASCII detail decreased: {self.ascii_detail_level:.1f}")
            self.ui.show_info_temporarily(90)
        
        elif action == 'increase_ascii_detail':
            # Increase ASCII detail (less pixelated)
            self.ascii_detail_level = clamp(self.ascii_detail_level + 0.3, 0.0, 2.0)
            print(f"ASCII detail increased: {self.ascii_detail_level:.1f}")
            self.ui.show_info_temporarily(90)
        
        elif action == 'reset':
            self.reset_to_original()
        
        elif action == 'toggle_help':
            self.ui.show_help = not self.ui.show_help
            
        elif action == 'toggle_gesture':
            self.gesture_mode = not self.gesture_mode
            print(f"Gesture control: {'ON' if self.gesture_mode else 'OFF'}")
            
        elif action == 'toggle_debug':
            self.show_debug = not self.show_debug
            print(f"Debug overlay: {'ON' if self.show_debug else 'OFF'}")
            
        elif action.startswith('select_effect_'):
            effect_index = int(action.split('_')[-1])
            self.select_effect_by_index(effect_index)
    
    def handle_gesture(self, gesture: str, current_frame: np.ndarray):
        """
        Handle detected hand gestures
        
        Args:
            gesture: Detected gesture ('thumbs_up', 'thumbs_down', 'fist')
            current_frame: Current processed frame
        """
        print(f"Gesture detected: {gesture}")
        
        if gesture == 'thumbs_up':
            # Increase ASCII detail (less pixelated)
            self.ascii_detail_level = clamp(self.ascii_detail_level + 0.3, 0.0, 2.0)
            print(f"ASCII detail increased: {self.ascii_detail_level:.1f}")
            self.ui.show_info_temporarily(90)
            
        elif gesture == 'thumbs_down':
            # Decrease ASCII detail (more pixelated)
            self.ascii_detail_level = clamp(self.ascii_detail_level - 0.3, 0.0, 2.0)
            print(f"ASCII detail decreased: {self.ascii_detail_level:.1f}")
            self.ui.show_info_temporarily(90)
            
        elif gesture == 'fist':
            # Take a photo
            self.save_snapshot(current_frame)
            print("Photo captured with fist gesture!")
    
    def cycle_to_next_effect(self):
        """Cycle to the next effect"""
        self.current_effect_index = (self.current_effect_index + 1) % len(self.effect_names)
        self.current_effect = self.effect_names[self.current_effect_index]
        self.ui.show_info_temporarily(90)  # Show info for 1.5 seconds
        print(f"Switched to effect: {self.current_effect}")
    
    def select_effect_by_index(self, index: int):
        """Select effect by index (0-based)"""
        if 0 <= index < len(self.effect_names):
            self.current_effect_index = index
            self.current_effect = self.effect_names[index]
            self.ui.show_info_temporarily(90)
            print(f"Selected effect: {self.current_effect}")
    
    def adjust_intensity(self, delta: float):
        """Adjust effect intensity"""
        self.intensity = clamp(self.intensity + delta, 0.0, 2.0)
        self.effect_processor.set_intensity(self.intensity)
        self.ui.show_info_temporarily(60)
        print(f"Intensity: {self.intensity:.1f}")
    
    def reset_to_original(self):
        """Reset to original effect with default intensity"""
        self.current_effect = 'original'
        self.current_effect_index = self.effect_names.index('original')
        self.intensity = 1.0
        self.effect_processor.set_intensity(self.intensity)
        self.ui.show_info_temporarily(90)
        print("Reset to original")
    
    def save_snapshot(self, frame):
        """Save current frame as snapshot"""
        saved_path = save_image(frame, self.config['save_directory'])
        if saved_path:
            self.ui.show_info_temporarily(120)  # Show confirmation
            print(f"Snapshot saved!")
        else:
            print("Failed to save snapshot")
    
    def cleanup(self):
        """Clean up all resources"""
        print("Shutting down PyArt...")
        
        # Save current settings
        self.config['default_effect'] = self.current_effect
        self.config['default_intensity'] = self.intensity
        save_config(self.config)
        
        # Clean up components
        self.camera.release()
        self.ui.cleanup()
        
        print("PyArt shut down successfully")


def main():
    """Main entry point"""
    try:
        app = PyArtApp()
        app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure OpenCV windows are closed
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
