"""
PyArt Demo - Test effects with sample images
This script demonstrates PyArt effects using a generated test image instead of webcam
"""

import cv2
import numpy as np
import time
from src.effects import EffectProcessor
from src.utils import save_image


def create_test_image(width=640, height=480):
    """Create a colorful test image for demonstration"""
    # Create a gradient background
    test_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create colorful gradient
    for y in range(height):
        for x in range(width):
            test_image[y, x] = [
                int(255 * (x / width)),  # Blue component
                int(255 * (y / height)),  # Green component
                int(255 * ((x + y) / (width + height)))  # Red component
            ]
    
    # Add some geometric shapes
    cv2.circle(test_image, (width//4, height//4), 50, (255, 255, 255), -1)
    cv2.rectangle(test_image, (width//2, height//4), (width//2 + 100, height//4 + 80), (0, 255, 255), -1)
    cv2.ellipse(test_image, (3*width//4, 3*height//4), (60, 40), 45, 0, 360, (255, 0, 255), -1)
    
    # Add some text
    cv2.putText(test_image, "PyArt Demo", (width//2 - 80, height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return test_image


def demo_all_effects():
    """Demonstrate all available effects"""
    print("PyArt Effects Demo")
    print("=" * 40)
    
    # Create test image
    test_image = create_test_image()
    
    # Initialize effect processor
    effect_processor = EffectProcessor()
    effect_names = effect_processor.get_effect_names()
    
    print(f"Demonstrating {len(effect_names)} effects...")
    print("Press any key to cycle through effects, 'q' to quit, 's' to save current effect")
    
    # Create window
    cv2.namedWindow("PyArt Effects Demo", cv2.WINDOW_AUTOSIZE)
    
    current_effect_index = 0
    intensity = 1.0
    
    while True:
        current_effect = effect_names[current_effect_index]
        
        # Apply effect
        effect_processor.set_intensity(intensity)
        processed_image = effect_processor.apply_effect(test_image.copy(), current_effect)
        
        # Add info overlay
        display_image = processed_image.copy()
        cv2.putText(display_image, f"Effect: {current_effect.replace('_', ' ').title()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_image, f"Intensity: {intensity:.1f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_image, f"[{current_effect_index + 1}/{len(effect_names)}]", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_image, "SPACE: Next | +/-: Intensity | S: Save | Q: Quit", 
                   (10, display_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display
        cv2.imshow("PyArt Effects Demo", display_image)
        
        # Handle input
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
            break
        elif key == ord(' '):  # Space - next effect
            current_effect_index = (current_effect_index + 1) % len(effect_names)
            print(f"Switched to: {effect_names[current_effect_index]}")
        elif key == ord('+') or key == ord('='):  # Increase intensity
            intensity = min(2.0, intensity + 0.1)
            print(f"Intensity: {intensity:.1f}")
        elif key == ord('-') or key == ord('_'):  # Decrease intensity
            intensity = max(0.0, intensity - 0.1)
            print(f"Intensity: {intensity:.1f}")
        elif key == ord('s') or key == ord('S'):  # Save
            filename = f"demo_{current_effect}_{int(time.time())}.png"
            save_image(processed_image, "saved_images", f"demo_{current_effect}")
            print(f"Saved effect: {current_effect}")
    
    cv2.destroyAllWindows()
    print("Demo completed!")


def batch_process_effects():
    """Create sample images for all effects and save them"""
    print("Creating sample images for all effects...")
    
    test_image = create_test_image()
    effect_processor = EffectProcessor()
    effect_names = effect_processor.get_effect_names()
    
    for i, effect_name in enumerate(effect_names):
        print(f"Processing effect {i+1}/{len(effect_names)}: {effect_name}")
        
        # Apply effect with default intensity
        effect_processor.set_intensity(1.0)
        processed_image = effect_processor.apply_effect(test_image.copy(), effect_name)
        
        # Save the processed image
        save_image(processed_image, "saved_images", f"sample_{effect_name}")
    
    print(f"Created {len(effect_names)} sample images in 'saved_images' folder")


def main():
    """Main demo function"""
    print("PyArt Demo Options:")
    print("1. Interactive demo (press keys to change effects)")
    print("2. Batch create sample images")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                demo_all_effects()
                break
            elif choice == '2':
                batch_process_effects()
                break
            elif choice == '3':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nDemo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    main()
