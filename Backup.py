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
import xir # For VART runner creation
import pathlib
import os
import traceback # For detailed error logging

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
        """
        if not pathlib.Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        g = xir.Graph.deserialize(model_path)
        subgraphs = g.get_root_subgraph().get_children()
        dpu_subgraph = [s for s in subgraphs if s.get_attr("device") == "DPU"]
        if not dpu_subgraph:
            raise RuntimeError("Could not find DPU subgraph in the model.")
        
        self.runner = vart.Runner.create_runner(dpu_subgraph[0], "run")
        
        input_tensors = self.runner.get_input_tensors()
        output_tensors = self.runner.get_output_tensors()
        self.input_tensor = input_tensors[0]
        self.output_tensor = output_tensors[0]

        print("Getting scaling factors directly from tensor 'fix_point' attribute.", flush=True)
        input_fix_point = self.input_tensor.get_attr("fix_point")
        self.input_scale = 2**input_fix_point

        output_fix_point = self.output_tensor.get_attr("fix_point")
        self.output_scale = 2**(-output_fix_point)
        
        self.input_shape = tuple(self.input_tensor.dims)
        
        print(f"DPU initialized successfully:", flush=True)
        print(f"  - Model: {model_path}", flush=True)
        print(f"  - Input Shape: {self.input_shape}", flush=True)
        print(f"  - Output Shape: {self.output_tensor.dims}", flush=True)
        print(f"  - Calculated Input Scale: {self.input_scale}", flush=True)
        print(f"  - Calculated Output Scale: {self.output_scale}", flush=True)

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_shape[2], self.input_shape[1]))
        img = img / 255.0
        img = (img * self.input_scale).astype(np.int8)
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
        print("[RUN] Starting preprocessing...", flush=True)
        preprocessed_frame = self.preprocess(frame)
        
        print("[RUN] Preparing I/O buffers...", flush=True)
        input_data = [preprocessed_frame]
        output_data = [np.empty(self.output_tensor.dims, dtype=np.float32)]

        print("[RUN] Executing DPU runner...", flush=True)
        job_id = self.runner.execute_async(input_data, output_data)
        self.runner.wait(job_id)
        print("[RUN] DPU execution finished.", flush=True)

        raw_output = output_data[0] * self.output_scale
        
        print("[RUN] Postprocessing output...", flush=True)
        detections = self.postprocess(raw_output, frame.shape[:2])
        
        print("[RUN] Drawing detections...", flush=True)
        processed_frame = self.draw_detections(frame, detections)
        return processed_frame

# --- Frame Capture Thread ---
def capture_frames():
    global output_frame, lock
    yolo_detector = None
    try:
        print("Attempting to initialize DPU handler...", flush=True)
        yolo_detector = YOLOv8_DPU(MODEL_PATH)
    except Exception as e:
        print("="*50, flush=True)
        print("!!! FATAL ERROR during DPU initialization !!!", flush=True)
        print(f"Error Type: {type(e).__name__}", flush=True)
        print(f"Error Message: {e}", flush=True)
        print("--- Traceback ---", flush=True)
        traceback.print_exc()
        print("="*50, flush=True)
        print("Streaming will continue with raw camera feed (no detection).", flush=True)

    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera device {CAMERA_DEVICE}", flush=True)
        return

    print("Starting camera capture loop...", flush=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        if yolo_detector:
            try:
                # print("[LOOP] Calling yolo_detector.run()", flush=True)
                processed_frame = yolo_detector.run(frame)
            except Exception as e:
                print(f"ERROR during inference: {e}", flush=True)
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
    
    print("Starting Flask server... Access at http://<your_kv260_ip>:5000", flush=True)
    app.run(host='0.0.0.0', port=5000, threaded=True)
