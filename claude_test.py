#!/usr/bin/env python3
"""
YOLOv8n Object Detection on KV260 DPU
Complete implementation with all error handling
"""

from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import threading
import time
import os
import sys
import subprocess

# Environment setup for DPU
os.environ['XLNX_VART_FIRMWARE'] = '/lib/firmware/xilinx/b4096_300m/binary_container_1.bin'
os.environ['XILINX_XRT'] = '/usr'
os.environ['XLNX_ENABLE_FINGERPRINT_CHECK'] = '0'  # Disable fingerprint check
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'

# Import with error handling
try:
    import xir
    import vart
    DPU_AVAILABLE = True
    print("DPU libraries loaded successfully")
except:
    print("ERROR: Cannot import DPU libraries")
    sys.exit(1)

app = Flask(__name__)
output_frame = None
lock = threading.Lock()
status_message = "Initializing..."

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

class YOLOv8DPU:
    def __init__(self, model_path, conf_threshold=0.5, nms_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.model_loaded = False
        
        print(f"Initializing YOLOv8 with model: {model_path}")
        
        try:
            # Load XIR graph
            self.graph = xir.Graph.deserialize(model_path)
            subgraphs = self.graph.get_root_subgraph().toposort_child_subgraph()
            
            # Find DPU subgraph
            self.dpu_subgraph = None
            for sg in subgraphs:
                if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                    self.dpu_subgraph = sg
                    print(f"Found DPU subgraph: {sg.get_name()}")
                    break
            
            if not self.dpu_subgraph:
                raise Exception("No DPU subgraph found")
            
            # Create runner - with retry mechanism
            for attempt in range(3):
                try:
                    self.runner = vart.Runner.create_runner(self.dpu_subgraph, "run")
                    print(f"Runner created successfully on attempt {attempt + 1}")
                    break
                except Exception as e:
                    print(f"Runner creation attempt {attempt + 1} failed: {e}")
                    if attempt == 2:
                        raise
                    time.sleep(1)
            
            # Get input/output tensors
            self.input_tensors = self.runner.get_input_tensors()
            self.output_tensors = self.runner.get_output_tensors()
            
            # Get input dimensions
            self.input_dims = tuple(self.input_tensors[0].dims)
            print(f"Input dimensions: {self.input_dims}")
            
            # Determine input size
            if len(self.input_dims) == 4:
                self.input_height = self.input_dims[1]
                self.input_width = self.input_dims[2]
            else:
                self.input_height = self.input_width = 640
                
            self.model_loaded = True
            print("Model initialization complete")
            
        except Exception as e:
            print(f"Model initialization failed: {e}")
            self.model_loaded = False
            raise

    def preprocess(self, image):
        """Preprocess image for YOLOv8"""
        # Resize
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to uint8 (DPU expects uint8)
        # No normalization to 0-1 for DPU!
        input_data = np.expand_dims(rgb, axis=0).astype(np.uint8)
        
        return input_data

    def postprocess(self, outputs, original_shape):
        """
        Process YOLOv8 outputs
        Handles multiple possible output formats
        """
        detections = []
        
        try:
            # Get main output
            output = outputs[0]
            
            # Handle different shapes
            if len(output.shape) == 3:
                # Shape: [1, N, 84] or [1, 84, N]
                if output.shape[1] == 84:
                    # [1, 84, N] -> transpose to [1, N, 84]
                    output = output.transpose(0, 2, 1)
                predictions = output[0]  # Remove batch dimension
            elif len(output.shape) == 2:
                # Shape: [N, 84]
                predictions = output
            else:
                print(f"Unexpected output shape: {output.shape}")
                return detections
            
            # Process each prediction
            for i in range(predictions.shape[0]):
                row = predictions[i]
                
                # YOLOv8 format: [cx, cy, w, h, obj_conf, class_scores...]
                if len(row) >= 84:
                    # Extract values
                    cx, cy, w, h = row[0:4]
                    obj_conf = row[4]
                    class_scores = row[5:85]  # 80 classes
                    
                    # Get best class
                    class_id = np.argmax(class_scores)
                    class_conf = class_scores[class_id]
                    
                    # Combined confidence
                    confidence = obj_conf * class_conf
                    
                    if confidence > self.conf_threshold:
                        # Scale to original image
                        scale_x = original_shape[1] / self.input_width
                        scale_y = original_shape[0] / self.input_height
                        
                        # Convert to corner coordinates
                        x1 = int((cx - w/2) * scale_x)
                        y1 = int((cy - h/2) * scale_y)
                        x2 = int((cx + w/2) * scale_x)
                        y2 = int((cy + h/2) * scale_y)
                        
                        # Clip to image bounds
                        x1 = max(0, min(x1, original_shape[1]))
                        y1 = max(0, min(y1, original_shape[0]))
                        x2 = max(0, min(x2, original_shape[1]))
                        y2 = max(0, min(y2, original_shape[0]))
                        
                        if x2 > x1 and y2 > y1:
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'score': float(confidence),
                                'class': int(class_id)
                            })
            
            # Apply NMS
            if len(detections) > 0:
                detections = self.apply_nms(detections)
                
        except Exception as e:
            print(f"Postprocessing error: {e}")
            
        return detections

    def apply_nms(self, detections):
        """Apply Non-Maximum Suppression"""
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        # Apply NMS per class
        final_detections = []
        
        for class_id in range(80):
            class_dets = [d for d in detections if d['class'] == class_id]
            
            while len(class_dets) > 0:
                # Take highest confidence detection
                best = class_dets[0]
                final_detections.append(best)
                class_dets = class_dets[1:]
                
                # Remove overlapping detections
                remaining = []
                for det in class_dets:
                    iou = self.calculate_iou(best['bbox'], det['bbox'])
                    if iou < self.nms_threshold:
                        remaining.append(det)
                class_dets = remaining
                
        return final_detections

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def detect(self, image):
        """Run detection on image"""
        if not self.model_loaded:
            return image, []
        
        try:
            # Preprocess
            input_data = self.preprocess(image)
            
            # Prepare output buffers
            output_buffers = []
            for tensor in self.output_tensors:
                dims = tuple(tensor.dims)
                buffer = np.empty(dims, dtype=np.float32)
                output_buffers.append(buffer)
            
            # Run inference
            job_id = self.runner.execute_async(input_data, output_buffers)
            self.runner.wait(job_id)
            
            # Postprocess
            detections = self.postprocess(output_buffers, image.shape)
            
            # Draw detections
            result_image = self.draw_detections(image.copy(), detections)
            
            return result_image, detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return image, []

    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            class_id = det['class']
            
            # Choose color based on class
            color = self.get_color(class_id)
            
            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            if class_id < len(COCO_CLASSES):
                label = f"{COCO_CLASSES[class_id]}: {score:.2f}"
            else:
                label = f"Class{class_id}: {score:.2f}"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            y_label = max(y1 - 5, 20)
            cv2.rectangle(image, (x1, y_label - 20), (x1 + label_size[0], y_label), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y_label - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return image

    def get_color(self, class_id):
        """Get color for class"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        return colors[class_id % len(colors)]

def find_camera():
    """Find working camera"""
    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.release()
                return i
            cap.release()
    return None

def processing_thread():
    """Main processing thread"""
    global output_frame, lock, status_message
    
    # Initialize model
    model = None
    try:
        model = YOLOv8DPU("yolov8n_kv260.xmodel", conf_threshold=0.5)
        status_message = "Model loaded successfully"
    except Exception as e:
        status_message = f"Model load failed: {str(e)}"
        print(status_message)
        return
    
    # Find camera
    cam_id = find_camera()
    if cam_id is None:
        status_message = "No camera found"
        return
    
    # Open camera
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    fps = 0
    fps_time = time.time()
    frame_count = 0
    
    status_message = "Running object detection"
    print("Object detection started")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Run detection
        start_time = time.time()
        result_frame, detections = model.detect(frame)
        inference_time = (time.time() - start_time) * 1000
        
        # Calculate FPS
        frame_count += 1
        if time.time() - fps_time > 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()
        
        # Add status info
        info_text = f"FPS: {fps} | Time: {inference_time:.1f}ms | Objects: {len(detections)}"
        cv2.putText(result_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update frame
        with lock:
            output_frame = result_frame.copy()
        
        # Print detections periodically
        if frame_count % 30 == 0 and len(detections) > 0:
            print(f"Detected: {', '.join([COCO_CLASSES[d['class']] for d in detections[:5]])}")

def generate():
    """Generate MJPEG stream"""
    global output_frame, lock
    
    while True:
        with lock:
            if output_frame is None:
                continue
            
            ret, buffer = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.03)

@app.route('/')
def index():
    global status_message
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv8n Object Detection - KV260 DPU</title>
        <style>
            body { 
                font-family: Arial; 
                text-align: center; 
                background: #1a1a1a; 
                color: white;
                margin: 0;
                padding: 20px;
            }
            h1 { color: #4CAF50; }
            .container { 
                max-width: 800px; 
                margin: 0 auto;
                background: #2a2a2a;
                padding: 20px;
                border-radius: 10px;
            }
            #stream { 
                border: 3px solid #4CAF50; 
                border-radius: 5px;
                max-width: 100%;
            }
            .status { 
                background: #333; 
                padding: 10px; 
                margin: 10px 0;
                border-radius: 5px;
            }
            .objects { 
                background: #3a3a3a; 
                padding: 15px; 
                margin: 20px 0;
                border-radius: 5px;
                text-align: left;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 YOLOv8n Object Detection</h1>
            <div class="status">Status: {{ status }}</div>
            <img id="stream" src="{{ url_for('video_feed') }}" width="640" height="480">
            <div class="objects">
                <h3>Detected Objects:</h3>
                <p>Objects will appear here when detected...</p>
                <p>Supported: person, car, bicycle, dog, cat, etc. (80 COCO classes)</p>
            </div>
        </div>
    </body>
    </html>
    ''', status=status_message)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Check DPU before starting
    if not os.path.exists("/dev/dpu"):
        print("WARNING: /dev/dpu not found, checking zocl module...")
        subprocess.run(['sudo', 'modprobe', 'zocl'], capture_output=True)
    
    # Start processing thread
    t = threading.Thread(target=processing_thread)
    t.daemon = True
    t.start()
    
    time.sleep(3)
    
    # Start server
    print("\n" + "="*60)
    print("YOLOv8n Object Detection Server")
    print("="*60)
    print("Open browser: http://[KV260_IP]:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
