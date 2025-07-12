#!/usr/bin/env python3
"""
xmodel file information check and test
"""

import xir
import vart
import numpy as np
import time

def check_xmodel(model_path):
    print(f"\n{'='*60}")
    print(f"Analyzing xmodel file: {model_path}")
    print(f"{'='*60}\n")
    
    try:
        # 1. Load XIR graph
        print("[1] Loading XIR graph...")
        graph = xir.Graph.deserialize(model_path)
        print("✓ Graph loaded successfully!")
        
        # 2. Check subgraphs
        print("\n[2] Analyzing subgraphs...")
        subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
        print(f"✓ Total subgraphs: {len(subgraphs)}")
        
        # 3. Find DPU subgraph
        dpu_subgraph = None
        for i, sg in enumerate(subgraphs):
            print(f"\n  Subgraph {i}: {sg.get_name()}")
            if sg.has_attr("device"):
                device = sg.get_attr("device")
                print(f"    - Device: {device}")
                if device.upper() == "DPU":
                    dpu_subgraph = sg
                    print("    ✓ DPU subgraph found!")
        
        if not dpu_subgraph:
            print("\n Cannot find DPU subgraph!")
            return
            
        # 4. Create Runner
        print("\n[3] Creating DPU Runner...")
        runner = vart.Runner.create_runner(dpu_subgraph, "run")
        print("✓ Runner created successfully!")
        
        # 5. Input/Output tensor information
        print("\n[4] Tensor Information:")
        input_tensors = runner.get_input_tensors()
        output_tensors = runner.get_output_tensors()
        
        print(f"\nInput Tensors ({len(input_tensors)}):")
        for i, tensor in enumerate(input_tensors):
            print(f"  [{i}] {tensor.name}")
            print(f"      - Shape: {tensor.shape}")
            print(f"      - Data Type: {tensor.get_data_type()}")
            
        print(f"\nOutput Tensors ({len(output_tensors)}):")
        for i, tensor in enumerate(output_tensors):
            print(f"  [{i}] {tensor.name}")
            print(f"      - Shape: {tensor.shape}")
            print(f"      - Data Type: {tensor.get_data_type()}")
        
        # 6. Inference test with dummy data
        print("\n[5] Testing inference with dummy data...")
        
        # Prepare input data (random image)
        input_shape = input_tensors[0].shape
        print(f"Input shape: {input_shape}")
        
        # Create dummy input
        dummy_input = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
        
        # Run inference
        print("Running inference...")
        start_time = time.time()
        
        job_id = runner.execute_async([dummy_input], output_tensors)
        runner.wait(job_id)
        
        inference_time = (time.time() - start_time) * 1000
        print(f"✓ Inference completed! Time: {inference_time:.2f}ms")
        
        # Check output
        print("\n[6] Output Check:")
        for i, tensor in enumerate(output_tensors):
            output_data = np.array(tensor)
            print(f"  Output tensor {i}: shape={output_data.shape}, dtype={output_data.dtype}")
            print(f"  - Min: {output_data.min():.4f}, Max: {output_data.max():.4f}")
            print(f"  - Mean: {output_data.mean():.4f}, Std: {output_data.std():.4f}")
        
        print(f"\n{'='*60}")
        print(" xmodel test completed! Model is working properly.")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n Error occurred: {str(e)}")
        print("\nPossible causes:")
        print("1. DPU might not be loaded")
        print("   → Run: sudo xmutil loadapp b4096_300m")
        print("2. xmodel file might be corrupted")
        print("3. Required libraries might be missing")

if __name__ == "__main__":
    # xmodel file path
    model_path = "YOLOv8Wrapper_int.xmodel"
    
    # Check DPU status
    print("\nChecking DPU status...")
    import subprocess
    try:
        result = subprocess.run(['dexplorer', '-w'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ DPU is active.")
        else:
            print(" Cannot find DPU. Please run 'sudo xmutil loadapp b4096_300m'")
    except:
        print("  Cannot execute dexplorer command.")
    
    # Test xmodel
    check_xmodel(model_path)
