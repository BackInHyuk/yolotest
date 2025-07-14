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
import pathlib
import os

# --- Configuration ---
MODEL_PATH = "yolov8n_kv260.xmodel" # IMPORTANT: Set the correct path to your compiled model
CAMERA_DEVICE = 0 # Camera device index (e.g., 0, 1, or a video file path)
# --- DEBUGGING: Lowered threshold to see if any weak detections appear ---
CONF_THRESHOLD = 0.2 # Confidence threshold for displaying detections (기존 0.5에서 하향 조정)
NMS_THRESHOLD = 0.4 # Non-Maximum Suppression threshold
INPUT_WIDTH = 640 # YOLOv8 model input width
INPUT_HEIGHT = 640 # YOLOv8 model input height

# --- COCO Class Names ---
# This list should match the classes the model was trained on.
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
        """
        if not pathlib.Path(model_path).exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        # Create the DPU runner
        self.runner = vart.Runner.create_runner(pathlib.Path(model_path), "run")
        
        # Get input and output tensor details
        input_tensors = self.runner.get_input_tensors()
        output_tensors = self.runner.get_output_tensors()
        
        self.input_tensor = input_tensors[0]
        self.output_tensor = output_tensors[0]

        # Get input scaling factor
        self.input_scale = vart.get_input_scale(self.input_tensor)
        
        # Get output scaling factor
        self.output_scale = vart.get_output_scale(self.output_tensor)

        # Get model input shape
        self.input_shape = tuple(self.input_tensor.dims) # e.g., (1, 640, 640, 3)
        
        print(f"DPU가 초기화되었습니다:")
        print(f"  - 모델: {model_path}")
        print(f"  - 입력 형태: {self.input_shape}")
        print(f"  - 입력 스케일: {self.input_scale}")
        print(f"  - 출력 스케일: {self.output_scale}")


    def preprocess(self, frame):
        """
        Preprocesses a single frame for YOLOv8 inference.
        - Resizes the image to the model's input dimensions.
        - Normalizes pixel values.
        - Adds a batch dimension.
        """
        # Resize the frame to the model's input size
        img = cv2.resize(frame, (self.input_shape[2], self.input_shape[1]))
        
        # Normalize and scale
        img = (img * self.input_scale).astype(np.int8)

        # Add batch dimension
        return np.expand_dims(img, 0)

    def postprocess(self, dpu_output, original_shape):
        """
        Postprocesses the DPU output to get bounding boxes.
        - Decodes the output tensor.
        - Applies Non-Maximum Suppression (NMS).
        - Scales boxes to the original frame size.
        """
        h, w = original_shape
        
        # The output tensor from YOLOv8 DPU is typically (1, 8400, 85)
        # where 85 = 4 (box) + 1 (confidence) + 80 (class scores)
        predictions = dpu_output.reshape(1, -1, len(COCO_CLASSES) + 4)[0]
        
        boxes = []
        confidences = []
        class_ids = []

        for pred in predictions:
            box = pred[:4]
            obj_conf = pred[4]
            
            if obj_conf < CONF_THRESHOLD:
                continue
            
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            max_class_score = class_scores[class_id]
            
            final_conf = obj_conf * max_class_score
            if final_conf > CONF_THRESHOLD:
                cx, cy, bw, bh = box
                
                x = int((cx - bw / 2) * w)
                y = int((cy - bh / 2) * h)
                width = int(bw * w)
                height = int(bh * h)

                boxes.append([x, y, width, height])
                confidences.append(float(final_conf))
                class_ids.append(class_id)

        # --- DEBUGGING: Print number of raw detections before NMS ---
        # 주석을 해제하면 터미널에서 탐지된 박스 개수를 확인할 수 있습니다.
        if len(boxes) > 0:
            print(f"임계값({CONF_THRESHOLD})을 넘은 박스 {len(boxes)}개 발견 (NMS 적용 전)")

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        
        final_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                final_detections.append({
                    "box": boxes[i],
                    "confidence": confidences[i],
                    "class_id": class_ids[i]
                })
        return final_detections

    def draw_detections(self, frame, detections):
        """
        Draws bounding boxes and labels on the frame.
        """
        for det in detections:
            box = det["box"]
            x, y, w, h = box
            class_id = det["class_id"]
            confidence = det["confidence"]
            
            label = f"{COCO_CLASSES[class_id]}: {confidence:.2f}"
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), (0, 255, 0), cv2.FILLED)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame

    def run(self, frame):
        """
        Runs the full pipeline: preprocess, inference, postprocess, draw.
        """
        original_shape = frame.shape[:2]
        preprocessed_frame = self.preprocess(frame)
        input_data = [preprocessed_frame]
        output_data = [np.empty(self.output_tensor.dims, dtype=np.float32)]

        job_id = self.runner.execute_async(input_data, output_data)
        self.runner.wait(job_id)

        detections = self.postprocess(output_data[0] * self.output_scale, original_shape)
        processed_frame = self.draw_detections(frame, detections)
        
        return processed_frame

# --- Frame Capture Thread ---
def capture_frames():
    global output_frame, lock

    try:
        yolo_detector = YOLOv8_DPU(MODEL_PATH)
    except Exception as e:
        print(f"DPU 초기화 오류: {e}")
        print("탐지 기능 없이 원본 카메라 영상만 스트리밍합니다.")
        yolo_detector = None

    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        print(f"오류: 카메라 장치({CAMERA_DEVICE})를 열 수 없습니다.")
        return

    print("카메라 캡처를 시작합니다...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("경고: 프레임을 가져오지 못했습니다. 재시도합니다...")
            time.sleep(0.1)
            continue
        
        if yolo_detector:
            try:
                processed_frame = yolo_detector.run(frame)
            except Exception as e:
                print(f"추론 중 오류 발생: {e}")
                processed_frame = frame 
        else:
            processed_frame = frame

        with lock:
            output_frame = processed_frame.copy()
    
    cap.release()

# --- MJPEG Streaming Generator ---
def generate():
    global output_frame, lock
    
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, jpeg = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
        time.sleep(1/30)

# --- Flask Routes ---
@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Video streaming home page."""
    html_page = """
    <html>
        <head>
            <title>YOLOv8 DPU Live Stream</title>
            <style>
                body { font-family: sans-serif; text-align: center; background-color: #282c34; color: white; }
                h1 { margin-top: 20px; }
                img { 
                    margin-top: 20px; 
                    border: 5px solid #61dafb;
                    border-radius: 10px;
                    background-color: #000;
                    max-width: 90%;
                }
            </style>
        </head>
        <body>
            <h1>YOLOv8 DPU Live Detection on KV260</h1>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
        </body>
    </html>
    """
    return render_template_string(html_page)

# --- Main Execution ---
if __name__ == '__main__':
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()
    
    print("Flask 서버를 시작합니다... http://<your_kv260_ip>:5000 에서 접속하세요.")
    app.run(host='0.0.0.0', port=5000, threaded=True)
