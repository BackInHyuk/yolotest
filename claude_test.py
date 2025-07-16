#!/usr/bin/env python3
"""
Debug version to diagnose why detections are not working
"""

import cv2
import numpy as np
import xir
import vart

def debug_model_outputs():
    """Debug function to check model outputs"""
    
    print("="*60)
    print("YOLOv8n Model Debug")
    print("="*60)
    
    # Load model
    model_path = "yolov8n_kv260.xmodel"
    graph = xir.Graph.deserialize(model_path)
    subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
    
    # Find DPU subgraph
    dpu_subgraph = None
    for sg in subgraphs:
        if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
            dpu_subgraph = sg
            break
    
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    
    # Print tensor info
    print("\nInput Tensor Info:")
    for t in input_tensors:
        print(f"  Name: {t.name}")
        print(f"  Shape: {t.dims}")
        print(f"  Type: {t.dtype.name}")
        fix_point = t.get_attr("fix_point") if t.has_attr("fix_point") else 0
        print(f"  Fix Point: {fix_point}")
    
    print("\nOutput Tensor Info:")
    for i, t in enumerate(output_tensors):
        print(f"\nOutput {i}:")
        print(f"  Name: {t.name}")
        print(f"  Shape: {t.dims}")
        print(f"  Type: {t.dtype.name}")
        fix_point = t.get_attr("fix_point") if t.has_attr("fix_point") else 0
        print(f"  Fix Point: {fix_point}")
    
    # Create test image with obvious objects
    test_img = np.ones((640, 640, 3), dtype=np.uint8) * 255  # White background
    
    # Draw black rectangle (person-like)
    cv2.rectangle(test_img, (200, 100), (280, 400), (0, 0, 0), -1)
    
    # Draw gray rectangle (car-like)
    cv2.rectangle(test_img, (400, 300), (550, 450), (128, 128, 128), -1)
    
    cv2.imwrite("debug_input.jpg", test_img)
    print("\nSaved test image as debug_input.jpg")
    
    # Preprocess
    rgb_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    # Check input data type and scale
    input_dtype = input_tensors[0].dtype
    fix_point = input_tensors[0].get_attr("fix_point") if input_tensors[0].has_attr("fix_point") else 0
    
    if input_dtype.name == 'INT8':
        input_scale = (2**fix_point) / 255.0
        input_data = (rgb_img.astype(np.float32) * input_scale).astype(np.int8)
        print(f"\nInput preprocessing: INT8, scale={input_scale:.6f}")
        print(f"Input data range: [{input_data.min()}, {input_data.max()}]")
    else:
        input_scale = 1.0 / 255.0
        input_data = rgb_img.astype(np.float32) * input_scale
        print(f"\nInput preprocessing: FLOAT32, scale={input_scale:.6f}")
        print(f"Input data range: [{input_data.min():.3f}, {input_data.max():.3f}]")
    
    input_data = np.expand_dims(input_data, axis=0)
    
    # Allocate output
    output_dtype = output_tensors[0].dtype
    output_buffer = np.zeros(output_tensors[0].dims, dtype=output_dtype.as_numpy_dtype)
    
    # Run inference
    print("\nRunning inference...")
    job_id = runner.execute_async([input_data], [output_buffer])
    runner.wait(job_id)
    
    # Analyze output
    print("\nRaw output analysis:")
    print(f"Output shape: {output_buffer.shape}")
    print(f"Output dtype: {output_buffer.dtype}")
    print(f"Output range: [{output_buffer.min()}, {output_buffer.max()}]")
    print(f"Output mean: {output_buffer.mean():.6f}")
    print(f"Output std: {output_buffer.std():.6f}")
    
    # Check for dead output
    if np.abs(output_buffer).max() < 0.001:
        print("\n⚠️  WARNING: Output is nearly zero - possible quantization failure")
    
    # De-quantize if INT8
    if output_dtype.name == 'INT8':
        fix_point = output_tensors[0].get_attr("fix_point") if output_tensors[0].has_attr("fix_point") else 0
        output_scale = 1.0 / (2**fix_point)
        output_float = output_buffer.astype(np.float32) * output_scale
        print(f"\nDe-quantized output (scale={output_scale:.6f}):")
        print(f"Range: [{output_float.min():.3f}, {output_float.max():.3f}]")
    else:
        output_float = output_buffer
    
    # Check YOLOv8 output format
    if len(output_buffer.shape) == 3:
        batch, dim1, dim2 = output_buffer.shape
        print(f"\nOutput interpretation:")
        print(f"  Batch size: {batch}")
        
        if dim1 == 84 or dim2 == 84:
            print("  ✓ Looks like YOLOv8 format (84 = 4 bbox + 80 classes)")
            
            # Transpose if needed
            if dim1 == 84:
                predictions = output_float[0].transpose(1, 0)
            else:
                predictions = output_float[0]
            
            # Check for valid detections
            print(f"\nChecking for detections (first 10 predictions):")
            for i in range(min(10, predictions.shape[0])):
                cx, cy, w, h = predictions[i, 0:4]
                class_scores = predictions[i, 4:84]
                max_score = class_scores.max()
                class_id = class_scores.argmax()
                
                print(f"  Pred {i}: cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f}, "
                      f"max_score={max_score:.3f}, class={class_id}")
                
                if max_score > 0.1:
                    print(f"    → Potential detection!")
        else:
            print(f"  ⚠️  Unexpected dimensions: {dim1} x {dim2}")
    
    print("\n" + "="*60)
    print("Debug complete. Check the output above for issues.")
    print("="*60)

if __name__ == "__main__":
    debug_model_outputs()
