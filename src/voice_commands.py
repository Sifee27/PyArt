"""
Voice commands module for PyArt
Handles speech recognition and voice command processing
"""

import threading
import time
import speech_recognition as sr
from typing import Dict, Callable, List, Optional


class VoiceCommandProcessor:
    """Processes voice commands for PyArt application"""
    
    def __init__(self):
        """Initialize voice command processor"""
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.is_listening = False
        self.listen_thread = None
        self.commands: Dict[str, List[str]] = {
            'next_effect': ['next effect', 'next filter', 'change effect', 'switch effect'],
            'previous_effect': ['previous effect', 'previous filter', 'go back'],
            'take_snapshot': ['take picture', 'take photo', 'take snapshot', 'snapshot', 'capture'],
            'start_recording': ['start recording', 'begin recording', 'record video'],
            'stop_recording': ['stop recording', 'end recording', 'finish recording'],
            'increase_intensity': ['increase intensity', 'more intensity', 'stronger effect'],
            'decrease_intensity': ['decrease intensity', 'less intensity', 'weaker effect'],
            'toggle_mirror': ['mirror', 'flip horizontal', 'mirror effect'],
            'reset_view': ['reset view', 'reset', 'normal view'],
            'zoom_in': ['zoom in', 'closer', 'magnify'],
            'zoom_out': ['zoom out', 'further', 'reduce size'],
            'toggle_help': ['show help', 'help menu', 'display help'],
            'exit_app': ['exit', 'quit', 'close application'],
            
            # Added advanced voice commands for cool features
            'toggle_night_vision': ['night vision', 'enable night vision', 'dark mode'],
            'toggle_motion_trails': ['motion trails', 'enable trails', 'show trails'],
            'toggle_face_tracking': ['face tracking', 'track faces', 'detect faces'],
            'toggle_drawing': ['drawing mode', 'enable drawing', 'start drawing'],
            'clear_drawing': ['clear drawing', 'erase drawing', 'clean canvas'],
            'toggle_split_screen': ['split screen', 'compare mode', 'before after'],
            'toggle_timelapse': ['time lapse', 'timelapse', 'start timelapse'],
            'cycle_face_effect': ['change face effect', 'next face effect', 'cycle face mode'],
            'next_face_emoji': ['next emoji', 'change emoji', 'different emoji'],
        }
        
        # Error handling counters
        self.recognition_errors = 0
        self.max_consecutive_errors = 5
        self.error_cooldown_time = 0
        self.error_cooldown_duration = 10  # Seconds to pause after too many errors
        
        # Add specific effect commands (will be populated from effect processor)
        self.effect_commands = {}
        
        # Callback function to execute commands
        self.callback = None
        
        # Last recognized command
        self.last_command = None
        self.last_command_time = 0
        self.command_cooldown = 1.0  # Seconds between commands
        
        # Microphone setup parameters
        self.energy_threshold = 300  # Microphone sensitivity
        self.dynamic_energy_threshold = True
        self.pause_threshold = 0.8   # Seconds of non-speaking audio before a phrase is considered complete
        
    def initialize(self) -> bool:
        """
        Initialize microphone
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.microphone = sr.Microphone()
            
            # Adjust microphone for ambient noise
            with self.microphone as source:
                print("Calibrating microphone for ambient noise... Please wait")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                self.recognizer.dynamic_energy_threshold = self.dynamic_energy_threshold
                self.recognizer.energy_threshold = self.energy_threshold
                self.recognizer.pause_threshold = self.pause_threshold
                
            print("Voice command system initialized")
            return True
        except Exception as e:
            print(f"Error initializing microphone: {e}")
            return False
            
    def set_callback(self, callback: Callable[[str], None]):
        """
        Set callback function to execute when command is recognized
        
        Args:
            callback: Function to call with command name
        """
        self.callback = callback
        
    def add_effect_commands(self, effect_names: List[str]):
        """
        Add effect-specific commands
        
        Args:
            effect_names: List of effect names
        """
        for effect in effect_names:
            # Convert effect name to readable format (e.g., ascii_simple → "ascii simple")
            readable_name = effect.replace('_', ' ')
            
            # Add to effect commands dictionary
            self.effect_commands[effect] = [
                f"apply {readable_name}",
                f"switch to {readable_name}",
                f"{readable_name} effect",
                readable_name
            ]
            
    def start_listening(self):
        """Start background listening thread"""
        if not self.microphone:
            if not self.initialize():
                print("Could not initialize microphone")
                return
                
        if not self.is_listening:
            self.is_listening = True
            self.listen_thread = threading.Thread(target=self._listen_loop)
            self.listen_thread.daemon = True          
            self.listen_thread.start()
            print("Voice command system is now listening")
    
    def stop_listening(self):
        """Stop listening for commands"""
        self.is_listening = False
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=1.0)
        print("Voice command system stopped")
            
    def _listen_loop(self):
        """Background thread for continuous listening"""
        while self.is_listening:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                try:
                    # Use Google's speech recognition service
                    # Could be replaced with offline models for better privacy
                    text = self.recognizer.recognize_google(audio).lower()
                    
                    if text:
                        command = self._match_command(text)
                        if command:
                            current_time = time.time()
                            if current_time - self.last_command_time >= self.command_cooldown:
                                self.last_command = command
                                self.last_command_time = current_time
                                print(f"Voice command recognized: {command}")
                                self.recognition_errors = 0  # Reset error count on success
                                
                                if self.callback:
                                    self.callback(command)
                except sr.UnknownValueError:
                    # Speech was unintelligible
                    self.recognition_errors += 1
                    print(f"Could not understand audio (error {self.recognition_errors}/{self.max_consecutive_errors})")
                except sr.RequestError as e:
                    print(f"Could not request results: {e}")
                    self.recognition_errors += 1
                    
                # Handle too many consecutive errors
                if self.recognition_errors >= self.max_consecutive_errors:
                    self.error_cooldown_time = time.time()
                    print(f"Too many errors, pausing voice recognition for {self.error_cooldown_duration} seconds")
                    time.sleep(self.error_cooldown_duration)
                    self.recognition_errors = 0  # Reset error count
                    
            except Exception as e:
                print(f"Error in voice recognition: {e}")
                time.sleep(1)  # Prevent busy loop in case of persistent errors
                
    def _match_command(self, text: str) -> Optional[str]:
        """
        Match spoken text to a command
        
        Args:
            text: Recognized speech text
        
        Returns:
            str: Command name if matched, None otherwise
        """
        # First check standard commands
        for command, phrases in self.commands.items():
            for phrase in phrases:
                if phrase in text:
                    return command
                    
        # Then check effect-specific commands
        for effect, phrases in self.effect_commands.items():
            for phrase in phrases:
                if phrase in text:
                    return f"effect:{effect}"
                    
        return None
