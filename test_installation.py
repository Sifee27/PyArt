"""
Test script to verify PyArt installation and camera availability
"""

import sys
import cv2
import numpy as np

def test_opencv():
    """Test OpenCV installation"""
    print(f"OpenCV version: {cv2.__version__}")
    return True

def test_numpy():
    """Test NumPy installation"""
    print(f"NumPy version: {np.__version__}")
    return True

def test_camera():
    """Test camera availability"""
    print("Testing camera availability...")
    
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i}: Available ({frame.shape})")
                cap.release()
                return True
            else:
                print(f"Camera {i}: Can open but no frame")
        else:
            print(f"Camera {i}: Not available")
        cap.release()
    
    return False

def main():
    """Run all tests"""
    print("PyArt Installation Test")
    print("=" * 30)
    
    tests = [
        ("OpenCV", test_opencv),
        ("NumPy", test_numpy),
        ("Camera", test_camera)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            status = "PASS" if result else "FAIL"
            print(f"{test_name}: {status}")
            if not result:
                all_passed = False
        except Exception as e:
            print(f"{test_name}: FAIL ({e})")
            all_passed = False
        print()
    
    if all_passed:
        print("✓ All tests passed! PyArt is ready to run.")
        print("Run 'python main.py' to start PyArt")
    else:
        print("✗ Some tests failed. Please check your installation.")

if __name__ == "__main__":
    main()
