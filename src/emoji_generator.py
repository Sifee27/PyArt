"""
Emoji generator module for PyArt
Generates emoji images programmatically using text emojis
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import emoji
import os
from typing import List, Optional, Tuple


class EmojiGenerator:
    """Generate emoji images for face tracking"""
    
    def __init__(self, emoji_size: int = 64):
        """
        Initialize emoji generator
        
        Args:
            emoji_size: Size of generated emoji images in pixels
        """
        self.emoji_size = emoji_size
        self.emoji_list = [
            '😎',  # sunglasses
            '😂',  # joy
            '😍',  # heart eyes
            '🤔',  # thinking
            '😊',  # smiling
            '😴',  # sleepy
            '🤯',  # mind blown
            '🥳',  # party
            '😋',  # yummy
            '🤖',  # robot
        ]
        self.generated_emojis = []
        self.font = None
        self._load_font()
        self._generate_all_emojis()
    
    def _load_font(self):
        """Load system font for emoji rendering"""
        try:
            # Try to load system emoji fonts
            font_paths = [
                "C:/Windows/Fonts/seguiemj.ttf",  # Windows Segoe UI Emoji
                "C:/Windows/Fonts/arial.ttf",     # Fallback
                "/System/Library/Fonts/Apple Color Emoji.ttc",  # macOS
                "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",  # Linux
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    self.font = ImageFont.truetype(font_path, size=self.emoji_size - 10)
                    print(f"Loaded emoji font: {font_path}")
                    break
            
            if self.font is None:
                # Use default font as fallback
                self.font = ImageFont.load_default()
                print("Using default font for emojis")
                
        except Exception as e:
            print(f"Font loading error: {e}")
            self.font = ImageFont.load_default()
    
    def _generate_emoji_image(self, emoji_char: str) -> Optional[np.ndarray]:
        """
        Generate an emoji image from unicode character
        
        Args:
            emoji_char: Unicode emoji character
            
        Returns:
            numpy array representing the emoji image in BGRA format
        """
        try:
            # Create PIL image with transparent background
            img = Image.new('RGBA', (self.emoji_size, self.emoji_size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Get text bounding box for centering
            bbox = draw.textbbox((0, 0), emoji_char, font=self.font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Calculate position to center the emoji
            x = (self.emoji_size - text_width) // 2 - bbox[0]
            y = (self.emoji_size - text_height) // 2 - bbox[1]
            
            # Draw the emoji
            draw.text((x, y), emoji_char, font=self.font, fill=(255, 255, 255, 255))
            
            # Convert PIL image to OpenCV format (BGRA)
            img_array = np.array(img)
            # Convert RGBA to BGRA for OpenCV
            if img_array.shape[2] == 4:
                bgra_img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
            else:
                # If no alpha channel, add one
                bgra_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                alpha = np.ones((bgra_img.shape[0], bgra_img.shape[1], 1), dtype=np.uint8) * 255
                bgra_img = np.concatenate([bgra_img, alpha], axis=2)
            
            return bgra_img
            
        except Exception as e:
            print(f"Error generating emoji {emoji_char}: {e}")
            return self._generate_fallback_emoji()
    
    def _generate_fallback_emoji(self) -> np.ndarray:
        """
        Generate a simple fallback emoji (smiley face) using OpenCV
        
        Returns:
            numpy array representing a simple smiley face
        """
        # Create transparent image
        img = np.zeros((self.emoji_size, self.emoji_size, 4), dtype=np.uint8)
        center = self.emoji_size // 2
        radius = center - 5
        
        # Draw yellow circle for face
        cv2.circle(img, (center, center), radius, (0, 255, 255, 255), -1)
        
        # Draw eyes
        eye_radius = radius // 8
        eye_y = center - radius // 3
        cv2.circle(img, (center - radius // 3, eye_y), eye_radius, (0, 0, 0, 255), -1)
        cv2.circle(img, (center + radius // 3, eye_y), eye_radius, (0, 0, 0, 255), -1)
        
        # Draw smile
        mouth_center = (center, center + radius // 4)
        mouth_radius = radius // 2
        cv2.ellipse(img, mouth_center, (mouth_radius, mouth_radius // 2), 0, 0, 180, (0, 0, 0, 255), 3)
        
        return img
    
    def _generate_all_emojis(self):
        """Generate all emoji images"""
        print("Generating emoji images...")
        
        for emoji_char in self.emoji_list:
            emoji_img = self._generate_emoji_image(emoji_char)
            if emoji_img is not None:
                self.generated_emojis.append(emoji_img)
                print(f"Generated emoji: {emoji_char}")
            else:
                # Add fallback emoji if generation failed
                fallback = self._generate_fallback_emoji()
                self.generated_emojis.append(fallback)
                print(f"Used fallback for emoji: {emoji_char}")
        
        print(f"Generated {len(self.generated_emojis)} emoji images")
    
    def get_emoji_count(self) -> int:
        """Get number of available emojis"""
        return len(self.generated_emojis)
    
    def get_emoji(self, index: int) -> Optional[np.ndarray]:
        """
        Get emoji image by index
        
        Args:
            index: Index of emoji to retrieve
            
        Returns:
            numpy array representing the emoji image, or None if invalid index
        """
        if 0 <= index < len(self.generated_emojis):
            return self.generated_emojis[index]
        return None
    
    def get_random_emoji(self) -> np.ndarray:
        """Get a random emoji"""
        import random
        return self.generated_emojis[random.randint(0, len(self.generated_emojis) - 1)]
    
    def save_emojis_to_disk(self, output_dir: str = "assets/emojis"):
        """
        Save generated emojis to disk as PNG files
        
        Args:
            output_dir: Directory to save emoji files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        emoji_names = [
            "sunglasses.png",
            "joy.png", 
            "heart_eyes.png",
            "thinking.png",
            "smiling.png",
            "sleepy.png",
            "mind_blown.png",
            "party.png",
            "yummy.png",
            "robot.png"
        ]
        
        for i, emoji_img in enumerate(self.generated_emojis):
            if i < len(emoji_names):
                filename = emoji_names[i]
            else:
                filename = f"emoji_{i}.png"
            
            filepath = os.path.join(output_dir, filename)
            
            # Convert BGRA to RGBA for saving
            rgba_img = cv2.cvtColor(emoji_img, cv2.COLOR_BGRA2RGBA)
            pil_img = Image.fromarray(rgba_img)
            pil_img.save(filepath)
            print(f"Saved emoji: {filepath}")


# Global emoji generator instance
_emoji_generator = None

def get_emoji_generator(emoji_size: int = 64) -> EmojiGenerator:
    """
    Get global emoji generator instance
    
    Args:
        emoji_size: Size of emoji images
        
    Returns:
        EmojiGenerator instance
    """
    global _emoji_generator
    if _emoji_generator is None:
        _emoji_generator = EmojiGenerator(emoji_size)
    return _emoji_generator
