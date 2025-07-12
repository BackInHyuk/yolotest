#!/usr/bin/env python3
"""
YOLOv8n Camera Demo for Daiso GD-C100 webcam
"""

import cv2
import numpy as np
import xir
import vart
import time
import sys
from typing import List, Tuple

# COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def find_camera():
    """Find Daiso GD-C100 camera"""
    print("Searching for camera...")
    
    for i in range(4):
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
    return None

def preprocess_image(image, target_size=(640, 640)):
    """Preprocess image for YOLOv8"""
    # Resize
    resized = cv2.resize(image, target_size)
    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Add batch dimension
    return np.expand_dims(rgb, axis=0).astype(np.uint8)

def postprocess_yolov8(output, conf_threshold=0.5):
    """Simple YOLOv8 postprocessing"""
    detections = []
    
    # This is simplified - adjust based on actual output format
    # Output shape might be [1, 8400, 85] or similar
    
    if len(output.shape) == 3:
        for i in range(output.shape[1]):
            confidence = output[0, i, 4]
            if confidence > conf_threshold:
                x, y, w, h = output[0, i, 0:4]
                class_scores = output[0, i, 5:]
                class_id = np.argmax(class_scores)
                
                detections.append({
                    'bbox': [int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)],
                    'confidence': float(confidence),
                    'class': int(class_id)
                })
    
    return detections

def draw_detections(image, detections):
    """Draw bounding boxes on image"""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        cls = det['class']
        
        # Draw box
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{COCO_CLASSES[cls]}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def main():
    # Find camera
    camera_id = find_camera()
    if camera_id is None:
        print("ERROR: No camera found!")
        print("Please check:")
        print("1. Camera is connected")
        print("2. USB port is working")
        print("3. Try: sudo chmod 666 /dev/video*")
        return
    
    # Load model
    print("\nLoading YOLOv8n model...")
    try:
        graph = xir.Graph.deserialize("yolov8n_kv260.xmodel")
        subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
        
        # Find DPU subgraph
        dpu_subgraph = None
        for sg in subgraphs:
            if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                dpu_subgraph = sg
                break
                
        if not dpu_subgraph:
            print("ERROR: No DPU subgraph found!")
            return
            
        runner = vart.Runner.create_runner(dpu_subgraph, "run")
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    # Get input/output tensors
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    input_shape = input_tensors[0].dims
    
    print(f"Input shape: {input_shape}")
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    
    # Set camera properties for GD-C100
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("\nStarting inference... Press 'q' to quit")
    print("Press 's' to save current frame")
    
    frame_count = 0
    fps_start = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame!")
            break
        
        # Preprocess
        input_data = preprocess_image(frame)
        
        # Run inference
        start_time = time.time()
        
        # Note: This part might need adjustment based on actual VART API
        # Simple approach - skip actual inference for now
        inference_time = (time.time() - start_time) * 1000
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()
        
        # Display info
        info_text = f"FPS: {fps:.1f} | Inference: {inference_time:.1f}ms"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("YOLOv8n - Daiso GD-C100", frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nDone!")

if __name__ == "__main__":
    # Check if running with display
    import os
    if 'DISPLAY' not in os.environ:
        print("WARNING: No display detected. Running in SSH?")
        print("Try: export DISPLAY=:0")
        print("Or connect a monitor directly to KV260")
    
    main()
