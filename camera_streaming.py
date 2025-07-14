#!/usr/bin/env python3
"""
YOLOv8n safe capture mode with complete error handling
"""

import cv2
import numpy as np
import time
import os
import sys

# Set environment before any imports
os.environ['XLNX_VART_FIRMWARE'] = '/lib/firmware/xilinx/b4096_300m/binary_container_1.bin'
os.environ['XILINX_XRT'] = '/usr'
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'  # Disable GStreamer

# Safe import with error handling
try:
    import xir
    import vart
    print("XIR and VART libraries loaded")
except ImportError as e:
    print(f"Error: {e}")
    print("Running in camera-only mode")
    xir = None
    vart = None

def find_working_camera():
    """Find camera without GStreamer"""
    print("Searching for camera...")
    
    # Try different backends
    backends = [
        cv2.CAP_V4L2,      # Video4Linux2
        cv2.CAP_V4L,       # Video4Linux
        cv2.CAP_ANY,       # Auto detect
    ]
    
    for backend in backends:
        for i in range(4):
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"Found camera {i} with backend {backend} ({width}x{height})")
                        cap.release()
                        return i, backend
                    cap.release()
            except:
                pass
    return None, None

def test_model_only():
    """Test model loading without inference"""
    if xir is None or vart is None:
        print("VART not available")
        return False
        
    try:
        print("\nTesting model load...")
        
        # Check model file
        model_path = "yolov8n_kv260.xmodel"
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
            
        # Load graph
        graph = xir.Graph.deserialize(model_path)
        print("Graph loaded")
        
        # Find subgraphs
        subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
        print(f"Found {len(subgraphs)} subgraphs")
        
        # Find DPU subgraph
        dpu_found = False
        for sg in subgraphs:
            if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                print(f"DPU subgraph: {sg.get_name()}")
                dpu_found = True
                
                # Get tensor info without creating runner
                tensors = sg.get_input_tensors()
                if tensors:
                    print(f"Input tensor: {tensors[0].name}")
                    print(f"Input dims: {tensors[0].dims}")
                break
                
        return dpu_found
        
    except Exception as e:
        print(f"Model test error: {e}")
        return False

def capture_images_simple():
    """Simple image capture without DPU"""
    
    # Find camera
    camera_id, backend = find_working_camera()
    if camera_id is None:
        print("No camera found!")
        return
        
    # Open camera with specific backend
    print(f"\nOpening camera {camera_id} with backend {backend}")
    cap = cv2.VideoCapture(camera_id, backend)
    
    # Basic settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nCapturing images... Press Ctrl+C to stop")
    print("Images will be saved as capture_XXX.jpg")
    
    frame_count = 0
    save_interval = 30  # Save every 30 frames
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to read frame")
                time.sleep(0.1)
                continue
                
            frame_count += 1
            
            # Save periodically
            if frame_count % save_interval == 0:
                filename = f"capture_{frame_count:06d}.jpg"
                success = cv2.imwrite(filename, frame)
                if success:
                    print(f"Saved {filename} - Shape: {frame.shape}")
                else:
                    print(f"Failed to save {filename}")
                    
            # Small delay
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\nStopped by user")
        
    finally:
        cap.release()
        print(f"Total frames processed: {frame_count}")

def safe_inference_test():
    """Test inference with extensive error handling"""
    if xir is None or vart is None:
        print("VART not available for inference")
        return
        
    try:
        print("\nAttempting safe inference test...")
        
        # Load model
        graph = xir.Graph.deserialize("yolov8n_kv260.xmodel")
        subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
        
        # Find DPU subgraph
        dpu_sg = None
        for sg in subgraphs:
            if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                dpu_sg = sg
                break
                
        if dpu_sg is None:
            print("No DPU subgraph found")
            return
            
        print("Creating runner...")
        
        # Try to create runner with error handling
        try:
            runner = vart.Runner.create_runner(dpu_sg, "run")
            print("Runner created successfully")
            
            # Get tensor info
            input_tensors = runner.get_input_tensors()
            if input_tensors:
                dims = input_tensors[0].dims
                print(f"Input shape: {dims}")
                
                # Create dummy input with correct type
                if len(dims) == 4:
                    # Ensure uint8 type
                    dummy = np.zeros(dims, dtype=np.uint8)
                    print(f"Created dummy input: {dummy.shape}, dtype: {dummy.dtype}")
                    
        except Exception as e:
            print(f"Runner error: {e}")
            print("This might be due to DPU driver issues")
            
    except Exception as e:
        print(f"Inference test error: {e}")

def main():
    """Main function with fallback options"""
    
    print("="*60)
    print("YOLOv8n KV260 Test Program")
    print("="*60)
    
    # Test model loading
    model_ok = test_model_only()
    print(f"\nModel test: {'PASSED' if model_ok else 'FAILED'}")
    
    # Try inference if model loaded
    if model_ok:
        safe_inference_test()
    
    # Always run camera capture
    print("\n" + "-"*60)
    print("Starting camera capture mode...")
    print("-"*60)
    
    capture_images_simple()

if __name__ == "__main__":
    main()
