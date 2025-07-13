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
CONF_THRESHOLD = 0.5 # Confidence threshold for displaying detections
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
            raise FileNotFoundError(f"Model file not found at {model_path}")

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
        
        print(f"DPU Initialized:")
        print(f"  - Model: {model_path}")
        print(f"  - Input Shape: {self.input_shape}")
        print(f"  - Input Scale: {self.input_scale}")
        print(f"  - Output Scale: {self.output_scale}")


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
            # The first 4 values are box coordinates (cx, cy, w, h)
            # The 5th value is the object confidence score
            # The rest are class scores
            
            box = pred[:4]
            obj_conf = pred[4]
            
            if obj_conf < CONF_THRESHOLD:
                continue
            
            # Find the class with the highest score
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            max_class_score = class_scores[class_id]
            
            # Final confidence is obj_conf * max_class_score
            final_conf = obj_conf * max_class_score
            if final_conf > CONF_THRESHOLD:
                cx, cy, bw, bh = box
                
                # Convert from center coordinates to x1, y1
                x = int((cx - bw / 2) * w)
                y = int((cy - bh / 2) * h)
                width = int(bw * w)
                height = int(bh * h)

                boxes.append([x, y, width, height])
                confidences.append(float(final_conf))
                class_ids.append(class_id)

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
        
        # Preprocess the frame
        preprocessed_frame = self.preprocess(frame)

        # Prepare input/output buffers
        input_data = [preprocessed_frame]
        output_data = [np.empty(self.output_tensor.dims, dtype=np.float32)]

        # Execute the model on DPU
        job_id = self.runner.execute_async(input_data, output_data)
        self.runner.wait(job_id)

        # Postprocess the output
        detections = self.postprocess(output_data[0] * self.output_scale, original_shape)

        # Draw the results on the original frame
        processed_frame = self.draw_detections(frame, detections)
        
        return processed_frame

# --- Frame Capture Thread ---
def capture_frames():
    global output_frame, lock

    # Initialize YOLOv8 DPU handler
    try:
        yolo_detector = YOLOv8_DPU(MODEL_PATH)
    except Exception as e:
        print(f"Error initializing DPU: {e}")
        print("Streaming will show raw camera feed without detections.")
        yolo_detector = None

    # Initialize video capture
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        print(f"Error: Could not open camera device {CAMERA_DEVICE}")
        return

    print("Starting camera capture...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to grab frame, retrying...")
            time.sleep(0.1)
            continue
        
        # If DPU is initialized, run inference
        if yolo_detector:
            try:
                processed_frame = yolo_detector.run(frame)
            except Exception as e:
                print(f"Error during inference: {e}")
                processed_frame = frame # Fallback to original frame on error
        else:
            processed_frame = frame

        # Update global frame
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
            
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
        
        # Return as MJPEG stream part
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
        # Control frame rate
        time.sleep(1/30) # Limit to ~30 FPS

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
    # Check if model path is set
    if MODEL_PATH == "yolov8n_kv260.xmodel":
        print("\n" + "="*50)
        print("!!! IMPORTANT: Please edit this script and set the 'MODEL_PATH' variable !!!")
        print("="*50 + "\n")
        exit(1)
        
    # Start the background thread for frame capture and inference
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()
    
    # Run Flask web server
    print("Starting Flask server... Access at http://<your_kv260_ip>:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)
