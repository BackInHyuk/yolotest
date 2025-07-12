#!/usr/bin/env python3
"""
Fixed xmodel check for yolov8n_kv260.xmodel
"""

import xir
import vart
import numpy as np
import time
import os

def check_xmodel(model_path):
    print("="*60)
    print(f"Analyzing xmodel file: {model_path}")
    print("="*60)
    
    try:
        # 1. Load XIR graph
        print("\n[1] Loading XIR graph...")
        graph = xir.Graph.deserialize(model_path)
        print("Graph loaded successfully!")
        
        # 2. Check subgraphs
        print("\n[2] Analyzing subgraphs...")
        subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
        print(f"Total subgraphs: {len(subgraphs)}")
        
        # 3. List ALL subgraphs
        dpu_subgraph = None
        for i, sg in enumerate(subgraphs):
            print(f"\n  Subgraph {i}: {sg.get_name()}")
            if sg.has_attr("device"):
                device = sg.get_attr("device")
                print(f"    - Device: {device}")
                if device.upper() == "DPU" and dpu_subgraph is None:
                    dpu_subgraph = sg
                    print("    DPU subgraph found!")
        
        if not dpu_subgraph:
            print("\nCannot find DPU subgraph!")
            return
            
        # 4. Create Runner
        print("\n[3] Creating DPU Runner...")
        runner = vart.Runner.create_runner(dpu_subgraph, "run")
        print("Runner created successfully!")
        
        # 5. Get tensor info
        print("\n[4] Tensor Information:")
        input_tensors = runner.get_input_tensors()
        output_tensors = runner.get_output_tensors()
        
        print(f"\nInput Tensors ({len(input_tensors)}):")
        for i, tensor in enumerate(input_tensors):
            print(f"  [{i}] {tensor.name}")
            dims = tensor.dims
            print(f"      - Dims: {dims}")
            # Remove get_data_type() call
            
        print(f"\nOutput Tensors ({len(output_tensors)}):")
        for i, tensor in enumerate(output_tensors):
            print(f"  [{i}] {tensor.name}")
            dims = tensor.dims
            print(f"      - Dims: {dims}")
        
        # 6. Inference test
        print("\n[5] Testing inference with dummy data...")
        
        input_dims = input_tensors[0].dims
        print(f"Input dims: {input_dims}")
        
        # Create dummy input
        dummy_input = np.random.randint(0, 255, size=input_dims, dtype=np.uint8)
        
        # Run inference
        print("Running inference...")
        start_time = time.time()
        
        job_id = runner.execute_async([dummy_input], output_tensors)
        runner.wait(job_id)
        
        inference_time = (time.time() - start_time) * 1000
        print(f"Inference completed! Time: {inference_time:.2f}ms")
        
        print("\n" + "="*60)
        print("xmodel test completed! Model is working properly.")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check DPU
    if os.path.exists("/dev/dpu"):
        print("/dev/dpu exists")
    else:
        print("/dev/dpu not found")
        
    # Test model
    check_xmodel("yolov8n_kv260.xmodel")
