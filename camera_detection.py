#!/usr/bin/env python3
"""
KV260에서 YOLOv8n 실시간 카메라 데모
"""

import cv2
import numpy as np
import xir
import vart
import time
import sys
from typing import List, Tuple

# COCO 클래스 이름 (80개 클래스)
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

class YOLOv8nDPU:
    def __init__(self, model_path: str, need_preprocess: bool = True):
        """
        YOLOv8n DPU 추론 클래스
        
        Args:
            model_path: xmodel 파일 경로
            need_preprocess: 전처리 필요 여부
        """
        print(f"[INFO] 모델 로드 중: {model_path}")
        
        # XIR 그래프 로드
        self.graph = xir.Graph.deserialize(model_path)
        subgraphs = self.graph.get_root_subgraph().toposort_child_subgraph()
        
        # DPU 서브그래프 찾기
        self.dpu_subgraph = None
        for sg in subgraphs:
            if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                self.dpu_subgraph = sg
                break
                
        if not self.dpu_subgraph:
            raise ValueError("DPU 서브그래프를 찾을 수 없습니다!")
            
        print(f"[INFO] DPU 서브그래프 찾음: {self.dpu_subgraph.get_name()}")
        
        # Runner 생성
        self.runner = vart.Runner.create_runner(self.dpu_subgraph, "run")
        
        # 입력/출력 텐서 정보 얻기
        self.input_tensors = self.runner.get_input_tensors()
        self.output_tensors = self.runner.get_output_tensors()
        
        # 입력 크기 정보
        self.input_shape = self.input_tensors[0].shape
        print(f"[INFO] 입력 shape: {self.input_shape}")
        
        # 전처리 설정
        self.need_preprocess = need_preprocess
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        # 리사이즈
        input_size = (self.input_shape[1], self.input_shape[2])  # (H, W)
        resized = cv2.resize(image, input_size)
        
        # 정규화 (0-255 -> 0-1)
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
        
    def postprocess(self, outputs: List[np.ndarray], conf_threshold: float = 0.5) -> List[dict]:
        """
        YOLOv8 출력 후처리
        
        Returns:
            List of detections: [{'bbox': [x1,y1,x2,y2], 'score': float, 'class': int}, ...]
        """
        # YOLOv8 출력 형식에 따라 구현
        # 일반적으로 [1, num_boxes, 85] 형태 (85 = 4 bbox + 1 obj_conf + 80 classes)
        
        detections = []
        output = outputs[0]  # 첫 번째 출력 사용
        
        # 실제 YOLOv8 출력 형식에 맞춰 수정 필요
        # 예시 구현:
        for i in range(output.shape[1]):
            obj_conf = output[0, i, 4]
            if obj_conf < conf_threshold:
                continue
                
            # 클래스 확률
            class_probs = output[0, i, 5:]
            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]
            
            # 최종 신뢰도
            score = obj_conf * class_conf
            if score < conf_threshold:
                continue
                
            # 바운딩 박스 (중심점 형식을 좌표 형식으로 변환)
            cx, cy, w, h = output[0, i, :4]
            x1 = int((cx - w/2) * 640)
            y1 = int((cy - h/2) * 640)
            x2 = int((cx + w/2) * 640)
            y2 = int((cy + h/2) * 640)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'score': float(score),
                'class': int(class_id)
            })
            
        return detections
        
    def run(self, image: np.ndarray) -> List[dict]:
        """추론 실행"""
        # 전처리
        if self.need_preprocess:
            input_data = self.preprocess(image)
        else:
            input_data = image
            
        # DPU 입력 준비
        input_data = input_data[np.newaxis, ...]  # 배치 차원 추가
        
        # DPU 실행
        job_id = self.runner.execute_async([input_data], self.output_tensors)
        self.runner.wait(job_id)
        
        # 출력 가져오기
        outputs = []
        for tensor in self.output_tensors:
            output = np.array(tensor)
            outputs.append(output)
            
        # 후처리
        detections = self.postprocess(outputs)
        
        return detections

def draw_detections(image: np.ndarray, detections: List[dict]) -> np.ndarray:
    """검출 결과 그리기"""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        score = det['score']
        class_id = det['class']
        
        # 색상 (클래스별로 다른 색상)
        color = ((class_id * 50) % 255, (class_id * 100) % 255, (class_id * 150) % 255)
        
        # 바운딩 박스
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 라벨
        label = f"{COCO_CLASSES[class_id]}: {score:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return image

def main():
    # 설정
    model_path = "yolov8n_kv260.xmodel"
    camera_id = 0  # 또는 "/dev/video0"
    
    # 모델 로드
    print("[INFO] YOLOv8n 모델 초기화...")
    model = YOLOv8nDPU(model_path)
    
    # 카메라 열기
    print(f"[INFO] 카메라 열기: {camera_id}")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다!")
        return
        
    # 카메라 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # FPS 계산용
    fps_time = time.time()
    fps_counter = 0
    fps = 0
    
    print("[INFO] 추론 시작... (종료: 'q' 키)")
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 프레임을 읽을 수 없습니다!")
                break
                
            # 추론 실행
            start_time = time.time()
            detections = model.run(frame)
            inference_time = (time.time() - start_time) * 1000
            
            # 검출 결과 그리기
            result_frame = draw_detections(frame.copy(), detections)
            
            # FPS 계산
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
                
            # 정보 표시
            info_text = f"FPS: {fps} | Inference: {inference_time:.1f}ms | Objects: {len(detections)}"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 화면 표시
            cv2.imshow("YOLOv8n on KV260", result_frame)
            
            # 키 입력 확인
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] 중단됨")
        
    finally:
        # 정리
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] 종료")

if __name__ == "__main__":
    main()
