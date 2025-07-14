#!/usr/bin/env python3
"""
YOLOv8n MJPEG Streaming with complete error handling
"""

from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import threading
import time
import os
import sys

# Environment setup
os.environ['XLNX_VART_FIRMWARE'] = '/lib/firmware/xilinx/b4096_300m/binary_container_1.bin'
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'

# Try importing DPU libraries
USE_DPU = True
try:
    import xir
    import vart
    print("DPU libraries loaded successfully")
except ImportError:
    print("Warning: DPU libraries not available, running camera-only mode")
    USE_DPU = False

app = Flask(__name__)
output_frame = None
lock = threading.Lock()

# COCO classes
COCO_CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

class SimpleDPUModel:
    """Simplified DPU model with error handling"""
    def __init__(self, model_path):
        self.ready = False
        self.runner = None
        
        if not USE_DPU:
            print("DPU disabled")
            return
            
        try:
            # Load model
            self.graph = xir.Graph.deserialize(model_path)
            subgraphs = self.graph.get_root_subgraph().toposort_child_subgraph()
            
            # Find DPU subgraph
            for sg in subgraphs:
                if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                    self.dpu_sg = sg
                    break
            else:
                print("No DPU subgraph found")
                return
                
            # Create runner
            self.runner = vart.Runner.create_runner(self.dpu_sg, "run")
            self.input_tensors = self.runner.get_input_tensors()
            self.output_tensors = self.runner.get_output_tensors()
            self.input_shape = tuple(self.input_tensors[0].dims)
            
            print(f"Model loaded: input shape {self.input_shape}")
            self.ready = True
            
        except Exception as e:
            print(f"Model init error: {e}")
            self.ready = False
    
    def process(self, frame):
        """Process frame with error handling"""
        if not self.ready or self.runner is None:
            return frame, []
            
        try:
            # Preprocess
            resized = cv2.resize(frame, (640, 640))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)
            
            # Allocate outputs
            outputs = []
            for tensor in self.output_tensors:
                shape = tuple(tensor.dims)
                outputs.append(np.empty(shape, dtype=np.float32))
            
            # Run inference
            job_id = self.runner.execute_async(input_data, outputs)
            self.runner.wait(job_id)
            
            # Simple detection (placeholder)
            # In real implementation, add proper postprocessing
            detections = []
            
            return frame, detections
            
        except Exception as e:
            print(f"Process error: {e}")
            return frame, []

def find_camera():
    """Find working camera"""
    backends = [cv2.CAP_V4L2, cv2.CAP_V4L, cv2.CAP_ANY]
    
    for backend in backends:
        for i in range(4):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f"Found camera {i} with backend {backend}")
                    return i, backend
    return None, None

def camera_thread():
    """Camera capture thread"""
    global output_frame, lock
    
    # Find camera
    cam_id, backend = find_camera()
    if cam_id is None:
        print("No camera found!")
        return
        
    # Initialize model (optional)
    model = None
    if USE_DPU and os.path.exists("yolov8n_kv260.xmodel"):
        try:
            model = SimpleDPUModel("yolov8n_kv260.xmodel")
        except:
            print("Model load failed, continuing without DPU")
            model = None
    
    # Open camera
    cap = cv2.VideoCapture(cam_id, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    fps = 0
    fps_time = time.time()
    frame_count = 0
    
    print("Camera thread started")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Process with model if available
        if model and model.ready:
            try:
                start = time.time()
                processed, detections = model.process(frame)
                inference_ms = (time.time() - start) * 1000
                
                # Draw detections
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                # Add info
                info = f"FPS: {fps:.1f} | Inference: {inference_ms:.1f}ms | Objects: {len(detections)}"
                cv2.putText(processed, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                result = processed
            except:
                result = frame
        else:
            # No model, just show camera
            info = f"FPS: {fps:.1f} | Camera Only Mode"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            result = frame
            
        # Update FPS
        frame_count += 1
        if time.time() - fps_time > 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()
            
        # Update global frame
        with lock:
            output_frame = result.copy()

def generate():
    """Generate MJPEG stream"""
    global output_frame, lock
    
    while True:
        with lock:
            if output_frame is None:
                continue
                
            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
        time.sleep(0.03)  # ~33ms = 30fps

@app.route('/')
def index():
    """Home page"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv8n Stream - KV260</title>
        <style>
            body {
                font-family: Arial;
                text-align: center;
                background: #f0f0f0;
                margin: 0;
                padding: 20px;
            }
            h1 {
                color: #333;
            }
            #stream {
                border: 3px solid #333;
                border-radius: 5px;
                max-width: 100%;
            }
            .info {
                margin: 20px;
                padding: 10px;
                background: white;
                border-radius: 5px;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <h1>YOLOv8n Object Detection Stream</h1>
        <div>
            <img id="stream" src="{{ url_for('video_feed') }}" width="640" height="480">
        </div>
        <div class="info">
            <p><strong>KV260 DPU Accelerated</strong></p>
            <p>Real-time object detection using YOLOv8n</p>
            <p>Stream URL: http://[KV260_IP]:5000/video_feed</p>
        </div>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    # Start camera thread
    camera_t = threading.Thread(target=camera_thread)
    camera_t.daemon = True
    camera_t.start()
    
    # Wait for initialization
    time.sleep(2)
    
    # Start Flask server
    print("\n" + "="*60)
    print("Starting MJPEG Streaming Server")
    print("="*60)
    print(f"Open browser and navigate to:")
    print(f"  http://localhost:5000 (on KV260)")
    print(f"  http://[KV260_IP]:5000 (from other devices)")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    main()
