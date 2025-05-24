"""
PyArt Demo - Test effects with sample images
This script demonstrates PyArt effects using a generated test image instead of webcam
"""

import cv2
import numpy as np
import time
import random
import math
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


def create_animated_test_image(frame_count, width=640, height=480):
    """Create an animated test image that changes over time"""
    test_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create animated gradient
    time_offset = frame_count * 0.1
    for y in range(height):
        for x in range(width):
            # Animated color waves
            r = int(127 + 127 * math.sin((x + time_offset * 50) / 50))
            g = int(127 + 127 * math.sin((y + time_offset * 30) / 30))
            b = int(127 + 127 * math.sin((x + y + time_offset * 40) / 40))
            test_image[y, x] = [b, g, r]
    
    # Add animated shapes
    center_x = int(width//2 + 100 * math.sin(time_offset))
    center_y = int(height//2 + 50 * math.cos(time_offset))
    radius = int(30 + 20 * math.sin(time_offset * 2))
    
    cv2.circle(test_image, (center_x, center_y), radius, (255, 255, 255), -1)
    
    # Rotating rectangle
    angle = time_offset * 50
    rect_center = (3*width//4, height//4)
    box_points = cv2.boxPoints(((rect_center[0], rect_center[1]), (80, 40), angle))
    box_points = np.int0(box_points)
    cv2.fillPoly(test_image, [box_points], (0, 255, 255))
    
    # Pulsing text
    text_size = 1 + 0.5 * math.sin(time_offset * 3)
    cv2.putText(test_image, "ANIMATED!", (width//2 - 100, height//2), 
                cv2.FONT_HERSHEY_SIMPLEX, text_size, (255, 255, 255), 2)
    
    return test_image


def create_kaleidoscope_effect(image):
    """Create a kaleidoscope effect from an image"""
    h, w = image.shape[:2]
    center = (w//2, h//2)
    
    # Take a triangular slice from the image
    mask = np.zeros((h, w), dtype=np.uint8)
    triangle = np.array([[center[0], center[1]], 
                        [w, center[1]], 
                        [center[0], 0]], np.int32)
    cv2.fillPoly(mask, [triangle], 255)
    
    # Extract the slice
    slice_img = cv2.bitwise_and(image, image, mask=mask)
    
    # Create kaleidoscope by rotating and mirroring
    kaleidoscope = np.zeros_like(image)
    
    for i in range(6):  # 6 segments for hexagonal kaleidoscope
        angle = i * 60
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(slice_img, M, (w, h))
        
        # Mirror every other segment
        if i % 2 == 1:
            rotated = cv2.flip(rotated, 1)
        
        kaleidoscope = cv2.add(kaleidoscope, rotated)
    
    return kaleidoscope


def create_particle_system(width=640, height=480, num_particles=100):
    """Create a particle system animation"""
    particles = []
    for _ in range(num_particles):
        particles.append({
            'x': random.randint(0, width),
            'y': random.randint(0, height),
            'vx': random.uniform(-2, 2),
            'vy': random.uniform(-2, 2),
            'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            'size': random.randint(2, 6)
        })
    return particles


def update_particles(particles, width, height):
    """Update particle positions and handle bouncing"""
    for particle in particles:
        particle['x'] += particle['vx']
        particle['y'] += particle['vy']
        
        # Bounce off walls
        if particle['x'] <= 0 or particle['x'] >= width:
            particle['vx'] *= -1
        if particle['y'] <= 0 or particle['y'] >= height:
            particle['vy'] *= -1
        
        # Keep particles in bounds
        particle['x'] = max(0, min(width, particle['x']))
        particle['y'] = max(0, min(height, particle['y']))


def draw_particles(image, particles):
    """Draw particles on the image"""
    for particle in particles:
        cv2.circle(image, (int(particle['x']), int(particle['y'])), 
                  particle['size'], particle['color'], -1)


def demo_interactive_effects():
    """Interactive demo with real-time effects and animations"""
    print("PyArt Interactive Effects Demo")
    print("=" * 40)
    print("Controls:")
    print("SPACE: Next effect | A: Animation mode | K: Kaleidoscope")
    print("P: Particle system | +/-: Intensity | S: Save | Q: Quit")
    print("Mouse: Interactive effects")
    
    # Create window
    cv2.namedWindow("PyArt Interactive Demo", cv2.WINDOW_AUTOSIZE)
    
    # Initialize
    effect_processor = EffectProcessor()
    effect_names = effect_processor.get_effect_names()
    current_effect_index = 0
    intensity = 1.0
    animation_mode = False
    kaleidoscope_mode = False
    particle_mode = False
    frame_count = 0
    
    # Mouse interaction variables
    mouse_x, mouse_y = 320, 240
    mouse_pressed = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y, mouse_pressed
        mouse_x, mouse_y = x, y
        mouse_pressed = (flags & cv2.EVENT_FLAG_LBUTTON) != 0
    
    cv2.setMouseCallback("PyArt Interactive Demo", mouse_callback)
    
    # Initialize particles
    particles = create_particle_system()
    
    while True:
        frame_count += 1
        
        # Create base image
        if animation_mode:
            base_image = create_animated_test_image(frame_count)
        else:
            base_image = create_test_image()
        
        # Add mouse interaction
        if mouse_pressed:
            cv2.circle(base_image, (mouse_x, mouse_y), 50, (255, 255, 0), 3)
            cv2.circle(base_image, (mouse_x, mouse_y), 20, (0, 255, 255), -1)
        
        # Apply special modes
        if kaleidoscope_mode:
            base_image = create_kaleidoscope_effect(base_image)
        
        # Apply current effect
        current_effect = effect_names[current_effect_index]
        effect_processor.set_intensity(intensity)
        processed_image = effect_processor.apply_effect(base_image.copy(), current_effect)
        
        # Add particle system
        if particle_mode:
            update_particles(particles, processed_image.shape[1], processed_image.shape[0])
            draw_particles(processed_image, particles)
        
        # Add info overlay with modern style
        display_image = processed_image.copy()
        
        # Semi-transparent overlay
        overlay = display_image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_image, 0.3, 0, display_image)
        
        # Status text
        y_offset = 35
        cv2.putText(display_image, f"Effect: {current_effect.replace('_', ' ').title()}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        cv2.putText(display_image, f"Intensity: {intensity:.1f} | Frame: {frame_count}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        # Mode indicators
        modes = []
        if animation_mode: modes.append("ANIMATED")
        if kaleidoscope_mode: modes.append("KALEIDOSCOPE")
        if particle_mode: modes.append("PARTICLES")
        if mouse_pressed: modes.append("MOUSE")
        
        if modes:
            cv2.putText(display_image, f"Modes: {' | '.join(modes)}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 20
        
        cv2.putText(display_image, f"[{current_effect_index + 1}/{len(effect_names)}]", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display
        cv2.imshow("PyArt Interactive Demo", display_image)
        
        # Handle input
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
            break
        elif key == ord(' '):  # Space - next effect
            current_effect_index = (current_effect_index + 1) % len(effect_names)
            print(f"Switched to: {effect_names[current_effect_index]}")
        elif key == ord('a') or key == ord('A'):  # Animation mode
            animation_mode = not animation_mode
            print(f"Animation mode: {'ON' if animation_mode else 'OFF'}")
        elif key == ord('k') or key == ord('K'):  # Kaleidoscope mode
            kaleidoscope_mode = not kaleidoscope_mode
            print(f"Kaleidoscope mode: {'ON' if kaleidoscope_mode else 'OFF'}")
        elif key == ord('p') or key == ord('P'):  # Particle mode
            particle_mode = not particle_mode
            if particle_mode:
                particles = create_particle_system()  # Reset particles
            print(f"Particle mode: {'ON' if particle_mode else 'OFF'}")
        elif key == ord('+') or key == ord('='):  # Increase intensity
            intensity = min(2.0, intensity + 0.1)
            print(f"Intensity: {intensity:.1f}")
        elif key == ord('-') or key == ord('_'):  # Decrease intensity
            intensity = max(0.0, intensity - 0.1)
            print(f"Intensity: {intensity:.1f}")
        elif key == ord('s') or key == ord('S'):  # Save
            timestamp = int(time.time())
            modes_str = "_".join(modes).lower() if modes else "normal"
            filename = f"interactive_{current_effect}_{modes_str}_{timestamp}"
            save_image(processed_image, "saved_images", filename)
            print(f"Saved: {filename}")
    
    cv2.destroyAllWindows()
    print("Interactive demo completed!")


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


def create_effect_showcase():
    """Create a showcase video of all effects"""
    print("Creating effects showcase...")
    
    # Video settings
    width, height = 640, 480
    fps = 30
    duration_per_effect = 3  # seconds
    frames_per_effect = fps * duration_per_effect
    
    # Initialize
    effect_processor = EffectProcessor()
    effect_names = effect_processor.get_effect_names()
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    timestamp = int(time.time())
    video_path = f"saved_images/effects_showcase_{timestamp}.mp4"
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    total_frames = len(effect_names) * frames_per_effect
    current_frame = 0
    
    for effect_idx, effect_name in enumerate(effect_names):
        print(f"Processing effect {effect_idx + 1}/{len(effect_names)}: {effect_name}")
        
        for frame in range(frames_per_effect):
            # Create animated base image
            base_image = create_animated_test_image(current_frame, width, height)
            
            # Apply effect with varying intensity
            intensity = 0.5 + 0.5 * math.sin(frame * 0.1)
            effect_processor.set_intensity(intensity)
            processed_image = effect_processor.apply_effect(base_image, effect_name)
            
            # Add title overlay
            overlay = processed_image.copy()
            cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, processed_image, 0.3, 0, processed_image)
            
            cv2.putText(processed_image, f"{effect_name.replace('_', ' ').title()}", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(processed_image, f"Effect {effect_idx + 1} of {len(effect_names)}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Write frame
            video_writer.write(processed_image)
            current_frame += 1
    
    video_writer.release()
    print(f"Showcase video saved: {video_path}")
    print(f"Total frames: {total_frames}, Duration: {total_frames/fps:.1f} seconds")


def main():
    """Main demo function with new options"""
    print("🎨 PyArt Demo - Enhanced Edition")
    print("=" * 40)
    print("1. Interactive demo (animated + mouse interaction)")
    print("2. Original effects demo")
    print("3. Batch create sample images")
    print("4. Create effects showcase video")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                demo_interactive_effects()
                break
            elif choice == '2':
                demo_all_effects()
                break
            elif choice == '3':
                batch_process_effects()
                break
            elif choice == '4':
                create_effect_showcase()
                break
            elif choice == '5':
                print("Goodbye! 🎨")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
        except KeyboardInterrupt:
            print("\nDemo interrupted. Goodbye! 🎨")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    main()
