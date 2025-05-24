#!/usr/bin/env python3
"""
Test script to verify ASCII character spacing improvements
"""

import cv2
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.effects import EffectProcessor

def create_test_pattern():
    """Create a test pattern with lines and shapes"""
    # Create a test image with clear patterns
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create vertical and horizontal lines to test overlap
    for i in range(0, 640, 40):
        cv2.line(img, (i, 0), (i, 480), (255, 255, 255), 2)
    
    for i in range(0, 480, 30):
        cv2.line(img, (0, i), (640, i), (255, 255, 255), 2)
    
    # Add some text
    cv2.putText(img, "SPACING TEST", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Add circles and rectangles
    cv2.circle(img, (160, 120), 60, (255, 255, 255), -1)
    cv2.rectangle(img, (400, 60), (580, 180), (255, 255, 255), -1)
    
    return img

def test_ascii_effects():
    """Test all ASCII effects with spacing improvements"""
    print("🎨 Testing ASCII Character Spacing Improvements...")
    
    # Create effect processor
    processor = EffectProcessor()
    processor.ascii_detail = 0.5  # Medium detail
    
    # Create test image
    test_img = create_test_pattern()
    
    # Test all ASCII effects
    ascii_effects = [
        ('ascii_simple', 'ASCII Simple'),
        ('ascii_detailed', 'ASCII Detailed'), 
        ('ascii_blocks', 'ASCII Blocks'),
        ('ascii_color', 'ASCII Color'),
        ('ascii_inverted', 'ASCII Inverted'),
        ('ascii_psychedelic', 'ASCII Psychedelic'),
        ('ascii_rainbow', 'ASCII Rainbow'),
        ('ascii_grid', 'ASCII Grid')
    ]
    
    print(f"✅ Created test pattern: {test_img.shape}")
    
    # Save original test image
    cv2.imwrite('test_original.png', test_img)
    print("📷 Saved original test pattern as 'test_original.png'")
    
    # Test each ASCII effect
    for effect_name, display_name in ascii_effects:
        try:
            print(f"\n🔧 Testing {display_name}...")
            
            # Get the effect method
            effect_method = getattr(processor, effect_name)
            
            # Apply effect
            result = effect_method(test_img)
            
            # Save result
            filename = f'test_{effect_name}.png'
            cv2.imwrite(filename, result)
            
            print(f"✅ {display_name}: Generated {result.shape} image -> {filename}")
            
        except Exception as e:
            print(f"❌ {display_name}: Error - {str(e)}")
    
    print("\n🎉 ASCII spacing test completed!")
    print("📁 Check the generated images to verify character spacing improvements:")
    print("   - No character overlap")
    print("   - Proper monospace alignment")  
    print("   - Consistent line spacing")
    print("   - Enhanced readability")

def main():
    """Run the ASCII spacing test"""
    print("=" * 60)
    print("🚀 PyArt ASCII Character Spacing Test")
    print("=" * 60)
    
    test_ascii_effects()
    
    print("\n" + "=" * 60)
    print("✨ Test completed! Check generated files for spacing quality.")
    print("=" * 60)

if __name__ == "__main__":
    main()
