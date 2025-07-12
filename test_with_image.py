#!/usr/bin/env python3
"""
YOLOv8 inference test with test image
"""

import numpy as np
import xir
import vart
import time

def create_test_image():
    """Create 640x640 test image"""
    # Create simple pattern test image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Draw rectangles (to look like objects)
    # Blue rectangle
    img[100:200, 100:200] = [255, 0, 0]
    # Green rectangle  
    img[300:400, 300:400] = [0, 255, 0]
    # Red rectangle
    img[100:300, 450:550] = [0, 0, 255]
    
    return img

def test_inference(model_path):
    print("Starting YOLOv8 inference test...")
    
    # Load model
    g = xir.Graph.deserialize(model_path)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
    dpu_sg = [sg for sg in subgraphs if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU"][0]
    
    runner = vart.Runner.create_runner(dpu_sg, "run")
    
    # Prepare input
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    
    input_shape = input_tensors[0].shape
    print(f"Input shape: {input_shape}")
    
    # Create test image or random data
    if input_shape == [1, 640, 640, 3]:
        print("Confirmed 640x640 RGB image input")
        test_data = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
    else:
        test_data = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
    
    # Inference
    start = time.time()
    job_id = runner.execute_async([test_data], output_tensors)
    runner.wait(job_id)
    end = time.time()
    
    print(f"Inference time: {(end-start)*1000:.2f}ms")
    
    # Analyze output
    outputs = [np.array(t) for t in output_tensors]
    for i, out in enumerate(outputs):
        print(f"\nOutput {i}:")
        print(f"  Shape: {out.shape}")
        print(f"  Range: [{out.min():.2f}, {out.max():.2f}]")
        
        # Guess YOLOv8 output format
        if len(out.shape) == 3 and out.shape[-1] == 85:
            print("  → Looks like YOLOv8 detection output (4 bbox + 1 conf + 80 classes)")
        elif len(out.shape) == 3 and out.shape[-1] == 84:
            print("  → Looks like YOLOv8 detection output (4 bbox + 80 classes)")

if __name__ == "__main__":
    test_inference("yolov8n_kv260.xmodel")
