"""
PyArt - Interactive Webcam Art Application
Main entry point for all the visual chaos

This started as a simple ASCII art converter and somehow became this whole thing.
Now it handles:
- Live camera feed processing
- 20+ visual effects (some are pretty wild)
- Hand gesture controls (surprisingly reliable)
- Voice commands (when it feels like working)
- Face tracking with emoji overlays (don't ask why)
- Recording and screenshots

If something breaks, it's probably my fault. -Jade
"""

import sys
import time
import cv2
import numpy as np
import os
import argparse
from datetime import datetime
from src.camera import Camera
from src.effects import EffectProcessor
from src.ui import UserInterface
from src.utils import save_image, save_config, load_config, clamp
from src.emoji_generator import get_emoji_generator

# Import gesture detection if available
try:
    from src.gesture_detector import HandGestureDetector
    GESTURE_AVAILABLE = True
except ImportError:
    print("Warning: Gesture detection not available")
    GESTURE_AVAILABLE = False

# Import voice command processing if available
try:
    from src.voice_commands import VoiceCommandProcessor
    VOICE_AVAILABLE = True
except ImportError:
    print("Warning: Voice command functionality not available")
    VOICE_AVAILABLE = False

from src.ui import UserInterface
from src.utils import (
    save_image, load_config, save_config, print_system_info,
    get_frame_fps_info, clamp
)


class PyArtApp:
    """Main PyArt application class with advanced features"""
    
    def __init__(self):
        """Initialize the PyArt application"""
        print("Getting PyArt ready...")
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
        
        # Initialize voice command processor if available
        self.voice_processor = None
        self.voice_mode = False
        if VOICE_AVAILABLE:
            self.voice_processor = VoiceCommandProcessor()
            self.voice_mode = self.config.get('voice_mode', False)
        
        # Application state
        self.current_effect_index = 0
        self.effect_names = self.effect_processor.get_effect_names()
        self.current_effect = self.config.get('default_effect', 'original')
        self.intensity = self.config.get('default_intensity', 1.0)
        self.running = False
        self.gesture_mode = True  # Enable gesture control
        self.show_debug = False   # Show gesture debug overlay
        
        # Voice command flag for snapshot
        self.take_snapshot_next_frame = False
        
        # ASCII detail control
        self.ascii_detail_level = 1.0  # 0.0 = very pixelated, 2.0 = very detailed
        
        # NEW ADVANCED FEATURES
        # Video recording
        self.video_writer = None
        self.is_recording = False
        self.recording_start_time = None
        
        # Frame rate control
        self.target_fps = 30
        self.frame_skip = 1  # Process every Nth frame for performance
        self.frame_counter = 0
        
        # Color themes
        self.color_themes = ['normal', 'sepia', 'cool', 'warm', 'vintage', 'cyberpunk']
        self.current_theme_index = 0
        
        # Mirror/flip effects
        self.mirror_horizontal = False
        self.mirror_vertical = False
        
        # Zoom and pan
        self.zoom_level = 1.0  # 1.0 = no zoom, 2.0 = 2x zoom
        self.pan_x = 0.0  # -1.0 to 1.0
        self.pan_y = 0.0  # -1.0 to 1.0
        
        # Burst photo mode
        self.burst_mode = False
        self.burst_count = 0
        self.burst_max = 5
        self.burst_photos = 5  # Number of photos in burst
        self.burst_duration = 2.0  # Duration in seconds for burst
        
        # Effect blending
        self.blend_mode = False
        self.secondary_effect = 'original'
        self.blend_ratio = 0.5  # 0.0 = primary only, 1.0 = secondary only
        # Performance tracking
        self.frame_times = []
        self.last_frame_time = time.time()
        
        # COOL FEATURES STATE VARIABLES
        # Time-lapse recording
        self.timelapse_mode = False
        self.timelapse_frames = []
        self.timelapse_interval = 0.5  # seconds between captures
        self.last_timelapse_capture = 0
        self.timelapse_fps = 10  # playback fps for timelapse video
        
        # Drawing mode
        self.drawing_mode = False
        self.drawing_overlay = None
        self.drawing_color = (0, 255, 255)  # Yellow by default
        self.drawing_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
        self.current_color_index = 0
        self.is_drawing = False
        self.last_drawing_point = None
        # Night vision mode
        self.night_vision_mode = False
        
        # Split-screen mode
        self.split_screen_mode = False
        self.original_frame_buffer = None
        
        # Motion detection trails
        self.motion_trails_mode = False
        self.trail_frames = []
        self.max_trail_frames = 5
        self.trail_decay = 0.7
          # Face tracking effects
        self.face_tracking_mode = False
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            print("Warning: Failed to load face cascade classifier. Face tracking will not work.")
        self.face_effect_type = 0  # 0: highlight, 1: pixelate, 2: emoji
        
        # Initialize emoji generator
        self.emoji_generator = get_emoji_generator(emoji_size=64)
        self.current_emoji_index = 0# Set initial effect index
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
        
        # Set up mouse callback for drawing mode
        cv2.setMouseCallback(self.ui.window_name, self.handle_mouse_drawing)
          # Initialize voice commands if enabled
        if self.voice_mode and self.voice_processor:
            if self.voice_processor.initialize():
                # Set up voice command callback
                self.voice_processor.set_callback(self.handle_voice_command)
                # Add effect-specific commands
                self.voice_processor.add_effect_commands(self.effect_names)
                # Start listening in background
                self.voice_processor.start_listening()
                print("Voice command system activated. Try saying 'take photo' or 'next effect'.")
            else:
                print("Warning: Voice command system failed to initialize.")
                self.voice_mode = False
        
        print("PyArt is ready to go!")
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
            
            # Apply image transformations (zoom, pan, mirror, flip)
            transformed_frame = self.apply_image_transformations(frame)
            
            # Set ASCII detail level for effects processor
            self.effect_processor.set_ascii_detail(self.ascii_detail_level)
            
            # Apply color theme before effects
            themed_frame = self.apply_color_theme(transformed_frame)
              # Apply current effect
            processed_frame = self.effect_processor.apply_effect(themed_frame, self.current_effect)
            
            # Apply cool features
            processed_frame = self.apply_cool_features(processed_frame, frame)
            
            # Apply effect blending if enabled
            if self.blend_mode and hasattr(self, 'previous_frame'):
                processed_frame = self.apply_effect_blending(processed_frame, self.previous_frame)
            self.previous_frame = processed_frame.copy()
            
            # Handle burst photo mode
            if self.burst_mode:
                self.handle_burst_mode(processed_frame)
            
            # Handle video recording
            if self.is_recording and self.video_writer is not None:
                # Resize frame to match camera dimensions for recording
                recording_frame = cv2.resize(processed_frame, (self.camera.width, self.camera.height))
                self.video_writer.write(recording_frame)
            
            # Update performance tracking
            current_time = time.time()
            
            # Frame rate control
            if self.target_fps > 0:
                elapsed = current_time - self.last_frame_time
                target_time = 1.0 / self.target_fps
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
            
            # Add debug overlay if enabled
            if self.show_debug and self.gesture_mode:
                processed_frame = self.gesture_detector.draw_debug_info(processed_frame)
              # Update performance tracking
            self.frame_times.append(current_time)
            
            # Display frame with UI
            self.ui.display_frame(
                processed_frame, 
                self.current_effect,
                self.intensity,
                self.effect_names,
                gesture_mode=self.gesture_mode,
                voice_mode=self.voice_mode,
                ascii_detail=self.ascii_detail_level
            )
            
            # Take snapshot if requested by voice command
            if self.take_snapshot_next_frame:
                self.save_snapshot(processed_frame)
                self.take_snapshot_next_frame = False
            
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
        
        elif action == 'toggle_voice':
            if VOICE_AVAILABLE:
                self.voice_mode = not self.voice_mode
                if self.voice_mode:
                    if self.voice_processor.initialize():
                        # Set up voice command callback
                        self.voice_processor.set_callback(self.handle_voice_command)
                        # Add effect-specific commands
                        self.voice_processor.add_effect_commands(self.effect_names)
                        # Start listening in background
                        self.voice_processor.start_listening()
                        print("Voice command system activated. Try saying 'take photo' or 'next effect'.")
                    else:
                        print("Voice command system failed to initialize.")
                        self.voice_mode = False
                else:
                    # Stop voice command processing
                    if self.voice_processor:
                        self.voice_processor.stop_listening()
                    print("Voice command system deactivated.")
            else:
                print("Voice command system not available. Install required dependencies.")
            
        elif action == 'toggle_debug':
            self.show_debug = not self.show_debug
            print(f"Debug overlay: {'ON' if self.show_debug else 'OFF'}")
            
        elif action.startswith('select_effect_'):
            effect_index = int(action.split('_')[-1])
            self.select_effect_by_index(effect_index)
        
        elif action == 'toggle_recording':
            self.toggle_recording()
        
        elif action == 'increase_zoom':
            self.zoom_level = clamp(self.zoom_level + 0.1, 1.0, 4.0)
            print(f"Zoom level: {self.zoom_level:.1f}x")
        
        elif action == 'decrease_zoom':
            self.zoom_level = clamp(self.zoom_level - 0.1, 1.0, 4.0)
            print(f"Zoom level: {self.zoom_level:.1f}x")
        
        elif action == 'pan_left':
            self.pan_x = clamp(self.pan_x - 0.1, -1.0, 1.0)
            print(f"Pan X: {self.pan_x:.1f}")
        
        elif action == 'pan_right':
            self.pan_x = clamp(self.pan_x + 0.1, -1.0, 1.0)
            print(f"Pan X: {self.pan_x:.1f}")
        
        elif action == 'pan_up':
            self.pan_y = clamp(self.pan_y - 0.1, -1.0, 1.0)
            print(f"Pan Y: {self.pan_y:.1f}")
        
        elif action == 'pan_down':
            self.pan_y = clamp(self.pan_y + 0.1, -1.0, 1.0)
            print(f"Pan Y: {self.pan_y:.1f}")
        
        elif action == 'toggle_blend':
            self.blend_mode = not self.blend_mode
            print(f"Blend mode: {'ON' if self.blend_mode else 'OFF'}")
        
        elif action == 'next_theme':
            self.current_theme_index = (self.current_theme_index + 1) % len(self.color_themes)
            print(f"Color theme: {self.color_themes[self.current_theme_index]}")
        
        elif action == 'previous_theme':
            self.current_theme_index = (self.current_theme_index - 1) % len(self.color_themes)
            print(f"Color theme: {self.color_themes[self.current_theme_index]}")
        
        elif action == 'toggle_mirror_h':
            self.mirror_horizontal = not self.mirror_horizontal
            print(f"Horizontal mirror: {'ON' if self.mirror_horizontal else 'OFF'}")
        
        elif action == 'toggle_mirror_v':
            self.mirror_vertical = not self.mirror_vertical
            print(f"Vertical mirror: {'ON' if self.mirror_vertical else 'OFF'}")
        
        elif action == 'start_burst':
            if not self.burst_mode:
                self.burst_mode = True
                print(f"Burst mode started - capturing {self.burst_photos} photos")
        
        elif action == 'increase_blend':
            self.blend_ratio = clamp(self.blend_ratio + 0.1, 0.0, 1.0)
            print(f"Blend ratio: {self.blend_ratio:.1f}")
        
        elif action == 'decrease_blend':
            self.blend_ratio = clamp(self.blend_ratio - 0.1, 0.0, 1.0)
            print(f"Blend ratio: {self.blend_ratio:.1f}")
        
        elif action == 'increase_fps':
            self.target_fps = clamp(self.target_fps + 5, 5, 60)
            print(f"Target FPS: {self.target_fps}")
        
        elif action == 'decrease_fps':
            self.target_fps = clamp(self.target_fps - 5, 5, 60)
            print(f"Target FPS: {self.target_fps}")
        elif action == 'reset_view':
            self.zoom_level = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            print("View reset to default")
            
        # COOL FEATURES ACTION HANDLERS
        elif action == 'toggle_timelapse':
            self.timelapse_mode = not self.timelapse_mode
            if self.timelapse_mode:
                self.timelapse_frames = []
                self.last_timelapse_capture = time.time()
                print("Time-lapse recording started")
            else:
                if self.timelapse_frames:
                    self.save_timelapse_video()
                print("Time-lapse recording stopped")
        
        elif action == 'toggle_drawing':
            self.drawing_mode = not self.drawing_mode
            if self.drawing_mode:
                h, w = current_frame.shape[:2]
                if self.drawing_overlay is None:
                    self.drawing_overlay = np.zeros((h, w, 3), dtype=np.uint8)
                print("Drawing mode ON - Use mouse to draw!")
            else:
                print("Drawing mode OFF")
        
        elif action == 'toggle_night_vision':
            self.night_vision_mode = not self.night_vision_mode
            print(f"Night vision mode: {'ON' if self.night_vision_mode else 'OFF'}")
        
        elif action == 'toggle_split_screen':
            self.split_screen_mode = not self.split_screen_mode
            if not self.split_screen_mode:
                self.original_frame_buffer = None
            print(f"Split-screen mode: {'ON' if self.split_screen_mode else 'OFF'}")
        
        elif action == 'clear_drawing':
            if self.drawing_overlay is not None:
                self.drawing_overlay.fill(0)
                print("Drawing overlay cleared")
        
        elif action == 'change_drawing_color':
            self.current_color_index = (self.current_color_index + 1) % len(self.drawing_colors)
            self.drawing_color = self.drawing_colors[self.current_color_index]
            color_names = ['Yellow', 'Magenta', 'Cyan', 'Green', 'Red', 'White']
            print(f"Drawing color changed to: {color_names[self.current_color_index]}")
        
        elif action == 'adjust_timelapse_speed':
            # Cycle through different time-lapse speeds
            speeds = [0.1, 0.25, 0.5, 1.0, 2.0]
            current_index = speeds.index(self.timelapse_interval) if self.timelapse_interval in speeds else 2
            self.timelapse_interval = speeds[(current_index + 1) % len(speeds)]
            print(f"Time-lapse interval: {self.timelapse_interval}s")

        elif action == 'toggle_motion_trails':
            self.motion_trails_mode = not self.motion_trails_mode
            if self.motion_trails_mode:
                self.trail_frames = [] # Reset trail frames when enabling
            print(f"Motion trails mode: {'ON' if self.motion_trails_mode else 'OFF'}")

        elif action == 'toggle_face_tracking':
            self.face_tracking_mode = not self.face_tracking_mode
            print(f"Face tracking mode: {'ON' if self.face_tracking_mode else 'OFF'}")
        
        elif action == 'cycle_face_effect':
            if self.face_tracking_mode:
                self.face_effect_type = (self.face_effect_type + 1) % 3 # 0: highlight, 1: pixelate, 2: emoji
                effects = ["Highlight", "Pixelate", "Emoji"]
                print(f"Face effect changed to: {effects[self.face_effect_type]}")
            else:
                print("Enable face tracking first (default key 'F') to cycle effects.")
        
        elif action == 'next_face_emoji':
            if self.face_tracking_mode and self.face_effect_type == 2: # Emoji effect is type 2
                emoji_count = self.emoji_generator.get_emoji_count()
                if emoji_count > 0:
                    self.current_emoji_index = (self.current_emoji_index + 1) % emoji_count
                    print(f"Changed to emoji index: {self.current_emoji_index}")
                else:
                    print("No emojis available to cycle through.")
            elif not self.face_tracking_mode:
                print("Enable face tracking first (default key 'F') to change emoji.")
            else: # Face tracking is on, but not emoji effect
                print("Switch to Emoji face effect (default key 'E') to change emoji.")
    
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
    
    def handle_voice_command(self, command: str):
        """
        Handle voice commands
        
        Args:
            command: Recognized voice command
        """
        print(f"Voice command recognized: {command}")
        
        # Standard commands
        if command == 'next_effect':
            self.cycle_to_next_effect()
        
        elif command == 'previous_effect':
            # Go to previous effect
            self.current_effect_index = (self.current_effect_index - 1) % len(self.effect_names)
            self.current_effect = self.effect_names[self.current_effect_index]
            self.ui.show_info_temporarily(90)
            print(f"Switched to previous effect: {self.current_effect}")
        
        elif command == 'take_snapshot':
            # Take a photo - will use the current frame in the next render cycle
            self.ui.show_info_temporarily(60)  # Show visual feedback
            print("Taking snapshot via voice command...")
            # Set a flag to take snapshot on next frame
            self.take_snapshot_next_frame = True
        
        elif command == 'start_recording':
            if not self.is_recording:
                self.toggle_recording()
        
        elif command == 'stop_recording':
            if self.is_recording:
                self.toggle_recording()
        
        elif command == 'increase_intensity':
            self.adjust_intensity(0.1)
        
        elif command == 'decrease_intensity':
            self.adjust_intensity(-0.1)
        
        elif command == 'toggle_mirror':
            self.mirror_horizontal = not self.mirror_horizontal
            print(f"Horizontal mirror: {'ON' if self.mirror_horizontal else 'OFF'}")
        
        elif command == 'reset_view':
            self.zoom_level = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            print("View reset to default")
        
        elif command == 'zoom_in':
            self.zoom_level = clamp(self.zoom_level + 0.2, 1.0, 4.0)
            print(f"Zoom level: {self.zoom_level:.1f}x")
        
        elif command == 'zoom_out':
            self.zoom_level = clamp(self.zoom_level - 0.2, 1.0, 4.0)
            print(f"Zoom level: {self.zoom_level:.1f}x")
        
        elif command == 'toggle_help':
            self.ui.show_help = not self.ui.show_help
        
        elif command == 'exit_app':
            print("Exiting application via voice command...")
            self.running = False
            
        # Advanced voice commands for cool features
        elif command == 'toggle_night_vision':
            self.night_vision_mode = not self.night_vision_mode
            print(f"Night vision mode: {'ON' if self.night_vision_mode else 'OFF'}")
            
        elif command == 'toggle_motion_trails':
            self.motion_trails_mode = not self.motion_trails_mode
            if self.motion_trails_mode:
                self.trail_frames = [] # Reset trail frames when enabling
            print(f"Motion trails mode: {'ON' if self.motion_trails_mode else 'OFF'}")
            
        elif command == 'toggle_face_tracking':
            self.face_tracking_mode = not self.face_tracking_mode
            print(f"Face tracking mode: {'ON' if self.face_tracking_mode else 'OFF'}")
            
        elif command == 'toggle_drawing':
            self.drawing_mode = not self.drawing_mode
            if self.drawing_mode:
                h, w = self.camera.read_frame().shape[:2]
                if self.drawing_overlay is None:
                    self.drawing_overlay = np.zeros((h, w, 3), dtype=np.uint8)
                print("Drawing mode ON - Use mouse to draw!")
            else:
                print("Drawing mode OFF")
                
        elif command == 'clear_drawing':
            if self.drawing_overlay is not None:
                self.drawing_overlay.fill(0)
                print("Drawing overlay cleared")
                
        elif command == 'toggle_split_screen':
            self.split_screen_mode = not self.split_screen_mode
            if not self.split_screen_mode:
                self.original_frame_buffer = None
            print(f"Split-screen mode: {'ON' if self.split_screen_mode else 'OFF'}")
            
        elif command == 'toggle_timelapse':
            self.timelapse_mode = not self.timelapse_mode
            if self.timelapse_mode:
                self.timelapse_frames = []
                self.last_timelapse_capture = time.time()
                print("Time-lapse recording started")
            else:
                if self.timelapse_frames:
                    self.save_timelapse_video()
                print("Time-lapse recording stopped")
                
        elif command == 'cycle_face_effect':
            if self.face_tracking_mode:
                self.face_effect_type = (self.face_effect_type + 1) % 3 # 0: highlight, 1: pixelate, 2: emoji
                effects = ["Highlight", "Pixelate", "Emoji"]
                print(f"Face effect changed to: {effects[self.face_effect_type]}")
            else:
                print("Enable face tracking first to cycle effects.")
        
        elif command == 'next_face_emoji':
            if self.face_tracking_mode and self.face_effect_type == 2: # Emoji effect is type 2
                emoji_count = self.emoji_generator.get_emoji_count()
                if emoji_count > 0:
                    self.current_emoji_index = (self.current_emoji_index + 1) % emoji_count
                    print(f"Changed to emoji index: {self.current_emoji_index}")
                else:
                    print("No emojis available to cycle through.")
            elif not self.face_tracking_mode:
                print("Enable face tracking first to change emoji.")
            else:
                print("Switch to Emoji face effect to change emoji.")
    
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
    
    def apply_image_transformations(self, frame):
        """Apply zoom, pan, mirror, and flip transformations to frame"""
        h, w = frame.shape[:2]
        transformed_frame = frame.copy()
        
        # Apply mirror/flip effects
        if self.mirror_horizontal:
            transformed_frame = cv2.flip(transformed_frame, 1)
        if self.mirror_vertical:
            transformed_frame = cv2.flip(transformed_frame, 0)
        
        # Apply zoom and pan
        if self.zoom_level != 1.0 or self.pan_x != 0.0 or self.pan_y != 0.0:
            # Calculate crop region for zoom
            crop_h = int(h / self.zoom_level)
            crop_w = int(w / self.zoom_level)
            
            # Calculate center position with pan offset
            center_y = int(h / 2 + self.pan_y * h / 4)
            center_x = int(w / 2 + self.pan_x * w / 4)
            
            # Calculate crop boundaries
            y1 = max(0, center_y - crop_h // 2)
            y2 = min(h, y1 + crop_h)
            x1 = max(0, center_x - crop_w // 2)
            x2 = min(w, x1 + crop_w)
            
            # Crop and resize
            cropped = transformed_frame[y1:y2, x1:x2]
            transformed_frame = cv2.resize(cropped, (w, h))
        
        return transformed_frame
    
    def apply_color_theme(self, frame):
        """Apply the current color theme to the frame"""
        if self.current_theme_index == 0:  # Default theme
            return frame
        
        theme_name = self.color_themes[self.current_theme_index]
        h, w = frame.shape[:2]
        themed_frame = frame.copy()
        
        if theme_name == "warm":
            # Warm theme - enhance reds and yellows
            themed_frame = cv2.addWeighted(themed_frame, 1.0, 
                                         np.full((h, w, 3), [0, 30, 60], dtype=np.uint8), 0.3, 0)
        elif theme_name == "cool":
            # Cool theme - enhance blues and cyans
            themed_frame = cv2.addWeighted(themed_frame, 1.0,
                                         np.full((h, w, 3), [60, 30, 0], dtype=np.uint8), 0.3, 0)
        elif theme_name == "vintage":
            # Vintage theme - sepia-like effect
            kernel = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
            themed_frame = cv2.transform(themed_frame, kernel)
        elif theme_name == "neon":
            # Neon theme - high contrast and saturation
            hsv = cv2.cvtColor(themed_frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.5)  # Increase saturation
            hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 30)  # Increase brightness
            themed_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        elif theme_name == "monochrome":
            # Monochrome theme - grayscale with tint
            gray = cv2.cvtColor(themed_frame, cv2.COLOR_BGR2GRAY)
            themed_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            themed_frame = cv2.addWeighted(themed_frame, 0.8,
                                         np.full((h, w, 3), [20, 20, 40], dtype=np.uint8), 0.2, 0)
        
        return themed_frame
    
    def apply_effect_blending(self, current_frame, previous_frame):
        """Blend current frame with previous frame for motion blur effect"""
        if previous_frame.shape != current_frame.shape:
            previous_frame = cv2.resize(previous_frame, (current_frame.shape[1], current_frame.shape[0]))
        
        # Blend frames based on blend ratio
        blended_frame = cv2.addWeighted(current_frame, self.blend_ratio, 
                                      previous_frame, 1.0 - self.blend_ratio, 0)
        return blended_frame
    
    def handle_burst_mode(self, frame):
        """Handle burst photo mode - capture multiple photos rapidly"""
        current_time = time.time()
        
        if self.burst_mode and not hasattr(self, 'burst_start_time'):
            self.burst_start_time = current_time
            self.burst_count = 0
        
        if hasattr(self, 'burst_start_time'):
            elapsed = current_time - self.burst_start_time
            
            # Capture photos at intervals during burst
            if elapsed < self.burst_duration and self.burst_count < self.burst_photos:
                if not hasattr(self, 'last_burst_capture') or \
                   current_time - self.last_burst_capture >= (self.burst_duration / self.burst_photos):
                    self.save_snapshot(frame)
                    self.burst_count += 1
                    self.last_burst_capture = current_time
                    print(f"Burst photo {self.burst_count}/{self.burst_photos}")
            
            # End burst mode
            elif elapsed >= self.burst_duration or self.burst_count >= self.burst_photos:
                self.burst_mode = False
                print(f"Burst mode completed - captured {self.burst_count} photos")
                delattr(self, 'burst_start_time')
                if hasattr(self, 'last_burst_capture'):
                    delattr(self, 'last_burst_capture')

    def toggle_recording(self):
        """Toggle video recording on/off"""
        if self.is_recording:
            # Stop recording
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            print("Video recording stopped")
        else:
            # Start recording
            self.is_recording = True
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.config['save_directory'], f"recording_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
            self.video_writer = cv2.VideoWriter(video_path, fourcc, self.target_fps, (self.camera.width, self.camera.height))
            self.recording_start_time = time.time()
            print(f"Video recording started: {video_path}")
    
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
    
    def apply_cool_features(self, processed_frame, original_frame):
        """Apply all the cool features in order - this orchestrates all the cool features"""
        # Make a copy of the frame for processing
        frame = processed_frame.copy()
        
        # Store the original frame for split-screen if needed
        if self.split_screen_mode and self.original_frame_buffer is None:
            self.original_frame_buffer = original_frame.copy()
        
        # 1. Handle time-lapse recording
        if self.timelapse_mode:
            self.handle_timelapse_capture(frame)
        
        # 2. Apply night vision effect if enabled
        if self.night_vision_mode:
            frame = self.apply_night_vision(frame)
        
        # 3. Apply motion detection trails if enabled
        if self.motion_trails_mode:
            frame = self.apply_motion_trails(frame, original_frame)
            
        # 4. Apply face tracking effects if enabled
        if self.face_tracking_mode:
            frame = self.apply_face_tracking(frame)
        
        # 5. Apply split-screen mode if enabled
        if self.split_screen_mode:
            frame = self.apply_split_screen(frame, self.original_frame_buffer)
        
        # 6. Apply drawing overlay if enabled
        if self.drawing_mode and self.drawing_overlay is not None:
            frame = self.apply_drawing_overlay(frame)
        
        return frame
    
    def handle_timelapse_capture(self, frame):
        """Handle time-lapse recording by capturing frames at intervals"""
        current_time = time.time()
        
        # Check if it's time to capture a frame
        if not hasattr(self, 'last_timelapse_capture') or \
           current_time - self.last_timelapse_capture >= self.timelapse_interval:
            # Resize the frame to save memory
            h, w = frame.shape[:2]
            resized_frame = cv2.resize(frame, (w // 2, h // 2))
            self.timelapse_frames.append(resized_frame)
            self.last_timelapse_capture = current_time
            print(f"Time-lapse: Captured frame {len(self.timelapse_frames)}")
    
    def save_timelapse_video(self):
        """Save time-lapse frames as a video"""
        if not self.timelapse_frames:
            print("No time-lapse frames to save")
            return
        
        # Create video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.config['save_directory'], f"timelapse_{timestamp}.mp4")
        
        # Get dimensions from the first frame
        h, w = self.timelapse_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        
        # Create video writer
        video_writer = cv2.VideoWriter(video_path, fourcc, self.timelapse_fps, (w, h))
        
        # Write all frames
        for frame in self.timelapse_frames:
            video_writer.write(frame)
        
        # Release the writer
        video_writer.release()
        
        print(f"Time-lapse video saved with {len(self.timelapse_frames)} frames: {video_path}")
        # Clear the frames
        self.timelapse_frames = []
    
    def apply_night_vision(self, frame):
        """Apply night vision effect (enhanced low-light visibility with green tint)"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to enhance contrast
        equalized = cv2.equalizeHist(gray)
        
        # Create a green-tinted frame (night vision style)
        h, w = frame.shape[:2]
        night_vision = np.zeros_like(frame)
        
        # Set the green channel to the equalized image, with higher intensity
        night_vision[:, :, 1] = cv2.add(equalized, 50)  # Green channel enhanced
        night_vision[:, :, 0] = equalized // 3          # Reduced blue channel
        night_vision[:, :, 2] = equalized // 3          # Reduced red channel
        
        # Add a slight green glow/bloom effect
        blurred = cv2.GaussianBlur(night_vision, (0, 0), 5)
        night_vision = cv2.addWeighted(night_vision, 0.8, blurred, 0.2, 0)
        
        # Add noise to simulate night vision device
        noise = np.zeros((h, w), dtype=np.uint8)
        cv2.randu(noise, 0, 25)
        noise_rgb = cv2.merge([noise, noise, noise])
        night_vision = cv2.add(night_vision, noise_rgb)
        
        # Add vignette effect (darker edges)
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = min(w, h) // 2
        cv2.circle(mask, center, radius, 255, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), radius // 3)
        mask_rgb = cv2.merge([mask, mask, mask])
        mask_rgb = mask_rgb.astype(float) / 255.0
        night_vision = (night_vision.astype(float) * mask_rgb).astype(np.uint8)
        
        return night_vision
    
    def apply_split_screen(self, processed_frame, original_frame):
        """Apply split-screen effect to compare before/after"""
        if original_frame is None or processed_frame.shape != original_frame.shape:
            return processed_frame
        
        h, w = processed_frame.shape[:2]
        
        # Create split view - left half original, right half processed
        split_frame = processed_frame.copy()
        split_width = w // 2
        
        # Copy the left half from the original frame
        split_frame[0:h, 0:split_width] = original_frame[0:h, 0:split_width]
        
        # Draw a dividing line
        cv2.line(split_frame, (split_width, 0), (split_width, h), (0, 255, 255), 2)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(split_frame, "Original", (20, 30), font, 0.7, (0, 255, 255), 2)
        cv2.putText(split_frame, "Processed", (split_width + 20, 30), font, 0.7, (0, 255, 255), 2)
        
        return split_frame
    
    def apply_drawing_overlay(self, frame):
        """Apply the drawing overlay on top of the frame"""
        if self.drawing_overlay is None:
            return frame
        
        # Ensure the overlay has the same dimensions as the frame
        if self.drawing_overlay.shape != frame.shape:
            h, w = frame.shape[:2]
            self.drawing_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Blend the drawing overlay with the frame
        result = cv2.addWeighted(frame, 1.0, self.drawing_overlay, 1.0, 0)
        
        return result
    
    def handle_mouse_drawing(self, event, x, y, flags, param):
        """Handle mouse events for drawing"""
        if not self.drawing_mode or self.drawing_overlay is None:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.last_drawing_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.last_drawing_point is not None:
                # Draw a line from last point to current point
                cv2.line(self.drawing_overlay, self.last_drawing_point, (x, y), 
                         self.drawing_color, 3)
                self.last_drawing_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_drawing and self.last_drawing_point is not None:
                # Complete the line
                cv2.line(self.drawing_overlay, self.last_drawing_point, (x, y), 
                         self.drawing_color, 3)
            self.is_drawing = False
            self.last_drawing_point = None

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
    
    def apply_motion_trails(self, frame, original_frame):
        """Apply motion detection trails effect"""
        # Convert frames to grayscale for motion detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize trail frames list if empty
        if not self.trail_frames:
            self.trail_frames = [gray_frame] * self.max_trail_frames
        
        # Calculate absolute difference between current frame and previous frame
        prev_frame = self.trail_frames[-1]
        frame_diff = cv2.absdiff(gray_frame, prev_frame)
        
        # Apply threshold to get areas of significant movement
        _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Apply some blur to smooth the mask
        motion_mask = cv2.GaussianBlur(motion_mask, (21, 21), 0)
        
        # Create colored motion trail
        colored_trail = np.zeros_like(frame)
        
        # Add all trail frames with decreasing opacity/intensity
        for i, trail_frame in enumerate(self.trail_frames):
            # Calculate opacity based on age of the frame
            opacity = self.trail_decay ** (self.max_trail_frames - i)
            
            # Get difference between this trail frame and the previous one
            if i > 0:
                trail_diff = cv2.absdiff(trail_frame, self.trail_frames[i-1])
                _, trail_mask = cv2.threshold(trail_diff, 30, 255, cv2.THRESH_BINARY)
                trail_mask = cv2.GaussianBlur(trail_mask, (21, 21), 0)
                
                # Create color based on position in trail (different colors for older vs newer motion)
                hue = int(180 * i / self.max_trail_frames)
                color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
                
                # Apply the color to the mask
                color_layer = np.zeros_like(frame)
                color_layer[:] = color
                mask_3channel = cv2.merge([trail_mask, trail_mask, trail_mask])
                trail_colored = cv2.bitwise_and(color_layer, mask_3channel)
                
                # Add to the trail with the calculated opacity
                colored_trail = cv2.addWeighted(colored_trail, 1.0, trail_colored, opacity, 0)
        
        # Update the trail frames list
        self.trail_frames.append(gray_frame)
        if len(self.trail_frames) > self.max_trail_frames:
            self.trail_frames.pop(0)
        
        # Blend the colored trail with the original frame
        result = cv2.addWeighted(frame, 1.0, colored_trail, 0.7, 0)
        
        return result
    
    def apply_face_tracking(self, frame):
        """Apply effects to detected faces"""
        # Make a copy of the frame
        result = frame.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Apply the selected effect to each face
        for (x, y, w, h) in faces:
            if self.face_effect_type == 0:
                # Highlight effect - draw rectangle and glow
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 255), 2)
                
                # Add glow effect around face
                mask = np.zeros_like(frame)
                cv2.rectangle(mask, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 255), 15)
                mask = cv2.GaussianBlur(mask, (21, 21), 0)
                result = cv2.addWeighted(result, 1.0, mask, 0.5, 0)
                
            elif self.face_effect_type == 1:
                # Pixelate effect
                face_region = frame[y:y+h, x:x+w]
                
                # Reduce the size to create pixelation
                pixelate_factor = 10
                small = cv2.resize(face_region, (w // pixelate_factor, h // pixelate_factor), 
                                  interpolation=cv2.INTER_LINEAR)
                
                # Scale back up using nearest neighbor interpolation for pixelated look
                pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                  # Put the pixelated face back into the frame
                result[y:y+h, x:x+w] = pixelated
            
            elif self.face_effect_type == 2:
                # Emoji overlay effect
                current_emoji_img = self.emoji_generator.get_emoji(self.current_emoji_index)
                
                if current_emoji_img is None:
                    print("No emoji available for overlay effect")
                    return result
                
                # Resize emoji to fit the face, maintaining aspect ratio
                eh, ew = current_emoji_img.shape[:2]
                scale = min(w / ew, h / eh)
                new_ew, new_eh = int(ew * scale), int(eh * scale)
                resized_emoji = cv2.resize(current_emoji_img, (new_ew, new_eh), interpolation=cv2.INTER_AREA)
                
                # Calculate position to center emoji on face
                pos_x = x + (w - new_ew) // 2
                pos_y = y + (h - new_eh) // 2

                # Ensure emoji is within frame boundaries (simple check, can be more robust)
                pos_x = max(pos_x, 0)
                pos_y = max(pos_y, 0)
                
                # Extract alpha channel if it exists (for PNGs)
                if resized_emoji.shape[2] == 4:
                    alpha_s = resized_emoji[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    emoji_bgr = resized_emoji[:, :, :3]
                    
                    # Iterate over each pixel of the emoji
                    for c in range(0, 3):
                        # Calculate the region of interest (ROI) on the frame
                        # Ensure ROI dimensions match emoji dimensions and are within frame bounds
                        roi_y_start = pos_y
                        roi_y_end = pos_y + new_eh
                        roi_x_start = pos_x
                        roi_x_end = pos_x + new_ew

                        frame_roi_h = result[roi_y_start:roi_y_end, roi_x_start:roi_x_end, c].shape[0]
                        frame_roi_w = result[roi_y_start:roi_y_end, roi_x_start:roi_x_end, c].shape[1]
                        
                        # Adjust emoji dimensions if they exceed frame ROI (e.g., face near edge)
                        emoji_h_to_blend = min(new_eh, frame_roi_h)
                        emoji_w_to_blend = min(new_ew, frame_roi_w)

                        if emoji_h_to_blend <= 0 or emoji_w_to_blend <= 0:
                            continue # Skip if ROI is invalid
                            
                        result[roi_y_start:roi_y_start+emoji_h_to_blend, roi_x_start:roi_x_start+emoji_w_to_blend, c] = \
                            (alpha_s[:emoji_h_to_blend, :emoji_w_to_blend] * emoji_bgr[:emoji_h_to_blend, :emoji_w_to_blend, c] + \
                             alpha_l[:emoji_h_to_blend, :emoji_w_to_blend] * result[roi_y_start:roi_y_start+emoji_h_to_blend, roi_x_start:roi_x_start+emoji_w_to_blend, c])
                else:
                    # If no alpha, just overlay (less ideal)
                    # This part might need adjustment if emojis without alpha are common
                    result[pos_y:pos_y+new_eh, pos_x:pos_x+new_ew] = resized_emoji[:,:,:3] # Use only BGR
        
        return result

# End of PyArtApp class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PyArt application")
    parser.add_argument("--voice", action="store_true", help="Enable voice command mode")
    args = parser.parse_args()
    app = PyArtApp()
    if args.voice and app.voice_processor:
        app.voice_mode = True
    app.run()
