#!/usr/bin/env python3
"""
YOLOv8n Camera Demo with error handling
"""

import cv2
import numpy as np
import xir
import vart
import time
import sys
import os

# Set display
os.environ['DISPLAY'] = ':0'

def find_camera():
    """Find camera"""
    print("Searching for camera...")
    
    for i in range(4):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    print(f"Found camera at /dev/video{i} ({int(width)}x{int(height)})")
                    cap.release()
                    return i
                cap.release()
        except Exception as e:
            print(f"Error checking video{i}: {e}")
    return None

def main():
    # Test display first
    print("Testing display...")
    try:
        # Create a test window
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow("Test", test_img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        print("Display test passed!")
    except Exception as e:
        print(f"Display error: {e}")
        print("Falling back to image save mode...")
        
        # Run in headless mode
        headless_mode()
        return
    
    # Find camera
    camera_id = find_camera()
    if camera_id is None:
        print("ERROR: No camera found!")
        return
    
    # Continue with normal mode...
    print("Opening camera for live view...")
    cap = cv2.VideoCapture(camera_id)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.imshow("Camera", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def headless_mode():
    """Run without display"""
    print("\nRunning in headless mode - saving images...")
    
    camera_id = find_camera()
    if camera_id is None:
        return
        
    cap = cv2.VideoCapture(camera_id)
    
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            filename = f"capture_{i}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        time.sleep(1)
    
    cap.release()
    print("Done!")

if __name__ == "__main__":
    main()
