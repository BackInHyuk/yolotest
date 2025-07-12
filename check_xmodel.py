#!/usr/bin/env python3
"""
xmodel 파일 정보 확인 및 테스트
"""

import xir
import vart
import numpy as np
import time

def check_xmodel(model_path):
    print(f"\n{'='*60}")
    print(f"xmodel 파일 분석: {model_path}")
    print(f"{'='*60}\n")
    
    try:
        # 1. XIR 그래프 로드
        print("[1] XIR 그래프 로드 중...")
        graph = xir.Graph.deserialize(model_path)
        print("✓ 그래프 로드 성공!")
        
        # 2. 서브그래프 확인
        print("\n[2] 서브그래프 분석...")
        subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
        print(f"✓ 전체 서브그래프 개수: {len(subgraphs)}")
        
        # 3. DPU 서브그래프 찾기
        dpu_subgraph = None
        for i, sg in enumerate(subgraphs):
            print(f"\n  서브그래프 {i}: {sg.get_name()}")
            if sg.has_attr("device"):
                device = sg.get_attr("device")
                print(f"    - Device: {device}")
                if device.upper() == "DPU":
                    dpu_subgraph = sg
                    print("    ✓ DPU 서브그래프 발견!")
        
        if not dpu_subgraph:
            print("\n❌ DPU 서브그래프를 찾을 수 없습니다!")
            return
            
        # 4. Runner 생성
        print("\n[3] DPU Runner 생성...")
        runner = vart.Runner.create_runner(dpu_subgraph, "run")
        print("✓ Runner 생성 성공!")
        
        # 5. 입출력 텐서 정보
        print("\n[4] 텐서 정보:")
        input_tensors = runner.get_input_tensors()
        output_tensors = runner.get_output_tensors()
        
        print(f"\n입력 텐서 ({len(input_tensors)}개):")
        for i, tensor in enumerate(input_tensors):
            print(f"  [{i}] {tensor.name}")
            print(f"      - Shape: {tensor.shape}")
            print(f"      - Data Type: {tensor.get_data_type()}")
            
        print(f"\n출력 텐서 ({len(output_tensors)}개):")
        for i, tensor in enumerate(output_tensors):
            print(f"  [{i}] {tensor.name}")
            print(f"      - Shape: {tensor.shape}")
            print(f"      - Data Type: {tensor.get_data_type()}")
        
        # 6. 더미 데이터로 추론 테스트
        print("\n[5] 더미 데이터로 추론 테스트...")
        
        # 입력 데이터 준비 (랜덤 이미지)
        input_shape = input_tensors[0].shape
        print(f"입력 shape: {input_shape}")
        
        # 더미 입력 생성
        dummy_input = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
        
        # 추론 실행
        print("추론 실행 중...")
        start_time = time.time()
        
        job_id = runner.execute_async([dummy_input], output_tensors)
        runner.wait(job_id)
        
        inference_time = (time.time() - start_time) * 1000
        print(f"✓ 추론 완료! 소요 시간: {inference_time:.2f}ms")
        
        # 출력 확인
        print("\n[6] 출력 확인:")
        for i, tensor in enumerate(output_tensors):
            output_data = np.array(tensor)
            print(f"  출력 텐서 {i}: shape={output_data.shape}, dtype={output_data.dtype}")
            print(f"  - Min: {output_data.min():.4f}, Max: {output_data.max():.4f}")
            print(f"  - Mean: {output_data.mean():.4f}, Std: {output_data.std():.4f}")
        
        print(f"\n{'='*60}")
        print("✅ xmodel 테스트 완료! 모델이 정상적으로 작동합니다.")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        print("\n가능한 원인:")
        print("1. DPU가 로드되지 않았을 수 있습니다")
        print("   → sudo xmutil loadapp b4096_300m")
        print("2. xmodel 파일이 손상되었을 수 있습니다")
        print("3. 필요한 라이브러리가 없을 수 있습니다")

if __name__ == "__main__":
    # xmodel 파일 경로
    model_path = "YOLOv8Wrapper_int.xmodel"
    
    # DPU 확인
    print("\nDPU 상태 확인...")
    import subprocess
    try:
        result = subprocess.run(['dexplorer', '-w'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ DPU가 활성화되어 있습니다.")
        else:
            print("❌ DPU를 찾을 수 없습니다. 'sudo xmutil loadapp b4096_300m' 실행 필요")
    except:
        print("⚠️  dexplorer 명령을 실행할 수 없습니다.")
    
    # xmodel 테스트
    check_xmodel(model_path)