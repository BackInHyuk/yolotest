#!/usr/bin/env python3
"""
테스트 이미지로 YOLOv8 추론 테스트
"""

import numpy as np
import xir
import vart
import time

def create_test_image():
    """640x640 테스트 이미지 생성"""
    # 간단한 패턴의 테스트 이미지 생성
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # 사각형 그리기 (객체처럼 보이도록)
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)  # 파란 사각형
    cv2.rectangle(img, (300, 300), (400, 400), (0, 255, 0), -1)  # 녹색 사각형
    cv2.rectangle(img, (450, 100), (550, 300), (0, 0, 255), -1)  # 빨간 사각형
    
    return img

def test_inference(model_path):
    print("YOLOv8 추론 테스트 시작...")
    
    # 모델 로드
    g = xir.Graph.deserialize(model_path)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
    dpu_sg = [sg for sg in subgraphs if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU"][0]
    
    runner = vart.Runner.create_runner(dpu_sg, "run")
    
    # 입력 준비
    input_tensors = runner.get_input_tensors()
    output_tensors = runner.get_output_tensors()
    
    input_shape = input_tensors[0].shape
    print(f"입력 shape: {input_shape}")
    
    # 테스트 이미지 생성 또는 랜덤 데이터
    if input_shape == [1, 640, 640, 3]:
        print("640x640 RGB 이미지 입력 확인")
        test_data = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
    else:
        test_data = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
    
    # 추론
    start = time.time()
    job_id = runner.execute_async([test_data], output_tensors)
    runner.wait(job_id)
    end = time.time()
    
    print(f"추론 시간: {(end-start)*1000:.2f}ms")
    
    # 출력 분석
    outputs = [np.array(t) for t in output_tensors]
    for i, out in enumerate(outputs):
        print(f"\n출력 {i}:")
        print(f"  Shape: {out.shape}")
        print(f"  범위: [{out.min():.2f}, {out.max():.2f}]")
        
        # YOLOv8 출력 형식 추측
        if len(out.shape) == 3 and out.shape[-1] == 85:
            print("  → YOLOv8 감지 출력으로 보임 (4 bbox + 1 conf + 80 classes)")
        elif len(out.shape) == 3 and out.shape[-1] == 84:
            print("  → YOLOv8 감지 출력으로 보임 (4 bbox + 80 classes)")

if __name__ == "__main__":
    test_inference("YOLOv8Wrapper_int.xmodel")