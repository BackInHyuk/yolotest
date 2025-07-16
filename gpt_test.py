#!/usr/bin/env python3
"""
yolov8_multi_cam_recursive.py

KV260 + PetaLinux 2023.1 환경에서,
/home/petalinux/yolotest/yolov8n_kv260.xmodel (멀티 DPU SG) 모델로
USB UVC 카메라 스트리밍 → 모든 COCO 클래스 탐지 + 바운딩박스

사용법:
  sudo python3 -m pip install opencv-python numpy
  chmod +x yolov8_multi_cam_recursive.py
  ./yolov8_multi_cam_recursive.py
"""

import cv2
import numpy as np
import xir
import vart
import time
import sys

# 사용자 설정
MODEL_PATH   = "/home/petalinux/yolotest/yolov8n_kv260.xmodel"
INPUT_SIZE   = (640, 640)  # 컴파일 시 지정한 입력 해상도
CONF_THRES   = 0.25
NMS_THRES    = 0.45
CAMERA_INDEX = 0         # /dev/video0

# COCO 클래스 이름 (0부터 79)
COCO_CLASSES = (
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
)

def find_dpu_subgraphs(sg, out):
    """재귀 탐색으로 DPU 서브그래프만 골라내기"""
    if sg.has_attr("device") and sg.get_attr("device") == "DPU":
        out.append(sg)
    for c in sg.get_children():
        find_dpu_subgraphs(c, out)

def initialize_dpu(model_path):
    # 1) Graph 로드
    graph = xir.Graph.deserialize(model_path)
    root  = graph.get_root_subgraph()

    # 2) 재귀 탐색으로 DPU 서브그래프 수집
    subgraphs = []
    find_dpu_subgraphs(root, subgraphs)
    if not subgraphs:
        print("Error: DPU 서브그래프를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Found {len(subgraphs)} DPU subgraph(s).")

    # 3) Runner 생성 & 입출력 버퍼 준비
    runners, io = [], []
    for sg in subgraphs:
        r = vart.Runner.create_runner(sg, "run")
        runners.append(r)
        in_t  = r.get_input_tensors()[0]
        out_t = r.get_output_tensors()[0]
        io.append({
            "in":  np.empty(in_t.dims,  dtype=np.int8,    order='C'),
            "out": np.empty(out_t.dims, dtype=np.float32, order='C')
        })
    return runners, io

def preprocess(frame):
    # BGR→RGB, 리사이즈, uint8→int8(-128~127), shape=(1,C,H,W)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    img = img.astype(np.int8) - 128
    return np.expand_dims(img, axis=0)

def run_dpu(runners, io, frame):
    # 첫 서브그래프 입력
    io[0]["in"][...] = preprocess(frame)
    # 순차 실행 및 중간 버퍼 연결
    for i, r in enumerate(runners):
        job_id = r.execute_async([io[i]["in"]], [io[i]["out"]])
        r.wait(job_id)
        if i+1 < len(runners):
            io[i+1]["in"][...] = io[i]["out"]
    return io[-1]["out"]

def postprocess(raw_out, orig_shape):
    # raw_out → (N,6): [x1,y1,x2,y2,conf,class]
    preds = raw_out.reshape(-1, 6)
    # confidence 필터
    mask = preds[:,4] > CONF_THRES
    preds = preds[mask]

    boxes     = preds[:, :4]
    scores    = preds[:, 4]
    class_ids = preds[:, 5].astype(int)

    idxs = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(),
        CONF_THRES, NMS_THRES
    )
    if len(idxs) == 0:
        return []

    h, w = orig_shape[:2]
    scale = np.array([w, h, w, h], dtype=np.float32) / INPUT_SIZE
    dets = []
    for i in idxs.flatten():
        x1, y1, x2, y2 = (boxes[i] * scale).astype(int)
        conf = float(scores[i])
        cls  = class_ids[i]
        name = COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else str(cls)
        dets.append((x1, y1, x2, y2, conf, name))
    return dets

def draw_detections(frame, dets):
    for x1, y1, x2, y2, conf, name in dets:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(
            frame, f"{name} {conf:.2f}",
            (x1, max(y1-5,0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1
        )

def main():
    runners, io = initialize_dpu(MODEL_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: 카메라({CAMERA_INDEX})를 열 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    fps, t0 = 0, time.time()
    window = "YOLOv8 Multi-Class Detection"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw  = run_dpu(runners, io, frame)
            dets = postprocess(raw, frame.shape)
            draw_detections(frame, dets)

            fps += 1
            if time.time() - t0 >= 1.0:
                cv2.putText(
                    frame, f"FPS: {fps}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0,255,255), 2
                )
                fps, t0 = 0, time.time()

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
