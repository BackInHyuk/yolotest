#!/usr/bin/env python3
"""
MJPEG streaming server for YOLOv8 on Xilinx KV260 with DPU.
This script loads a compiled .xmodel, runs inference on camera frames,
and streams the output with bounding boxes over HTTP.
"""

from flask import Flask, Response, render_template_string
import cv2
import threading
import time
import numpy as np
import vart
import xir # VART 실행기 생성을 위해 xir 라이브러리 추가
import pathlib
import os
import traceback # 상세한 오류 출력을 위해 traceback 라이브러리 추가

# --- Configuration ---
MODEL_PATH = "yolov8n_kv260.xmodel" # IMPORTANT: Set the correct path to your compiled model
CAMERA_DEVICE = 0 # Camera device index (e.g., 0, 1, or a video file path)
CONF_THRESHOLD = 0.2 # Confidence threshold for displaying detections
NMS_THRESHOLD = 0.4 # Non-Maximum Suppression threshold
INPUT_WIDTH = 640 # YOLOv8 model input width
INPUT_HEIGHT = 640 # YOLOv8 model input height

# --- COCO Class Names ---
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# --- Global variables for Flask app ---
output_frame = None
lock = threading.Lock()
app = Flask(__name__)

# --- YOLOv8 DPU Handler Class ---
class YOLOv8_DPU:
    def __init__(self, model_path):
        """
        Initializes the DPU runner for the YOLOv8 model.
        Uses a more robust method to be compatible with different VART versions.
        """
        if not pathlib.Path(model_path).exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        g = xir.Graph.deserialize(model_path)
        subgraphs = g.get_root_subgraph().get_children()
        dpu_subgraph = [s for s in subgraphs if s.get_attr("device") == "DPU"]
        if not dpu_subgraph:
            raise RuntimeError("모델에서 DPU 서브그래프를 찾을 수 없습니다.")
        
        self.runner = vart.Runner.create_runner(dpu_subgraph[0], "run")
        
        input_tensors = self.runner.get_input_tensors()
        output_tensors = self.runner.get_output_tensors()
        self.input_tensor = input_tensors[0]
        self.output_tensor = output_tensors[0]

        # --- [MODIFIED] Replaced deprecated scale functions ---
        # The vart.get_input_scale and vart.get_output_scale functions are not available
        # in all VART versions. We get the scaling factor from the tensor's 'fix_point' attribute instead.
        print("스케일 팩터를 텐서의 'fix_point' 속성에서 직접 가져옵니다.", flush=True)
        input_fix_point = self.input_tensor.get_attr("fix_point")
        self.input_scale = 2**input_fix_point

        output_fix_point = self.output_tensor.get_attr("fix_point")
        self.output_scale = 2**(-output_fix_point)
        # --- END OF MODIFICATION ---
        
        self.input_shape = tuple(self.input_tensor.dims)
        
        print(f"DPU가 초기화되었습니다:", flush=True)
        print(f"  - 모델: {model_path}", flush=True)
        print(f"  - 입력 형태: {self.input_shape}", flush=True)
        print(f"  - 출력 형태: {self.output_tensor.dims}", flush=True)
        print(f"  - 계산된 입력 스케일: {self.input_scale}", flush=True)
        print(f"  - 계산된 출력 스케일: {self.output_scale}", flush=True)

    def preprocess(self, frame):
        # --- [MODIFIED] Corrected preprocessing step ---
        # Resize the frame to the model's input size
        img = cv2.resize(frame, (self.input_shape[2], self.input_shape[1]))
        # Normalize pixel values from [0, 255] to [0, 1]
        img = img / 255.0
        # Apply the scaling factor and cast to the DPU input type (int8)
        img = (img * self.input_scale).astype(np.int8)
        # --- END OF MODIFICATION ---
        return np.expand_dims(img, 0)

    def postprocess(self, dpu_output, original_shape):
        h, w = original_shape
        predictions = dpu_output.reshape(1, -1, len(COCO_CLASSES) + 4)[0]
        boxes, confidences, class_ids = [], [], []

        for pred in predictions:
            obj_conf = pred[4]
            if obj_conf < CONF_THRESHOLD:
                continue
            
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            final_conf = obj_conf * class_scores[class_id]
            
            if final_conf > CONF_THRESHOLD:
                cx, cy, bw, bh = pred[:4]
                x = int((cx - bw / 2) * w)
                y = int((cy - bh / 2) * h)
                width = int(bw * w)
                height = int(bh * h)
                boxes.append([x, y, width, height])
                confidences.append(float(final_conf))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        final_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_detections.append({"box": boxes[i], "confidence": confidences[i], "class_id": class_ids[i]})
        return final_detections

    def draw_detections(self, frame, detections):
        for det in detections:
            box = det["box"]
            x, y, w, h = box
            label = f"{COCO_CLASSES[det['class_id']]}: {det['confidence']:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

    def run(self, frame):
        original_shape = frame.shape[:2]
        preprocessed_frame = self.preprocess(frame)
        input_data = [preprocessed_frame]
        output_data = [np.empty(self.output_tensor.dims, dtype=np.float32)]

        job_id = self.runner.execute_async(input_data, output_data)
        self.runner.wait(job_id)

        raw_output = output_data[0] * self.output_scale
        detections = self.postprocess(raw_output, original_shape)
        processed_frame = self.draw_detections(frame, detections)
        return processed_frame

# --- Frame Capture Thread ---
def capture_frames():
    global output_frame, lock
    yolo_detector = None
    try:
        print("DPU 핸들러 초기화를 시도합니다...", flush=True)
        yolo_detector = YOLOv8_DPU(MODEL_PATH)
    except Exception as e:
        print("="*50, flush=True)
        print("!!! DPU 초기화 중 심각한 오류 발생 !!!", flush=True)
        print(f"오류 종류: {type(e).__name__}", flush=True)
        print(f"오류 메시지: {e}", flush=True)
        print("--- 상세 오류 정보 (Traceback) ---", flush=True)
        traceback.print_exc()
        print("="*50, flush=True)
        print("탐지 기능 없이 원본 카메라 영상만 스트리밍합니다.", flush=True)

    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        print(f"오류: 카메라 장치({CAMERA_DEVICE})를 열 수 없습니다.", flush=True)
        return

    print("카메라 캡처를 시작합니다...", flush=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        if yolo_detector:
            try:
                processed_frame = yolo_detector.run(frame)
            except Exception as e:
                print(f"추론 중 오류 발생: {e}", flush=True)
                processed_frame = frame 
        else:
            processed_frame = frame

        with lock:
            output_frame = processed_frame.copy()
    
    cap.release()

# --- MJPEG Streaming Generator & Flask Routes ---
def generate():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None: continue
            ret, jpeg = cv2.imencode('.jpg', output_frame)
            if not ret: continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(1/30)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    html_page = """
    <html><head><title>YOLOv8 DPU Live Stream</title>
        <style>
            body { font-family: sans-serif; text-align: center; background-color: #282c34; color: white; }
            h1 { margin-top: 20px; }
            img { margin-top: 20px; border: 5px solid #61dafb; border-radius: 10px; background-color: #000; max-width: 90%; }
        </style></head>
        <body><h1>YOLOv8 DPU Live Detection on KV260</h1>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
        </body></html>"""
    return render_template_string(html_page)

# --- Main Execution ---
if __name__ == '__main__':
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()
    
    print("Flask 서버를 시작합니다... http://<your_kv260_ip>:5000 에서 접속하세요.", flush=True)
    app.run(host='0.0.0.0', port=5000, threaded=True)
