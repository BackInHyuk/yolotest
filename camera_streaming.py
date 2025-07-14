#!/usr/bin/env python3
"""
YOLOv8n streaming with complete error handling
"""

from flask import Flask, Response
import cv2
import numpy as np
import threading
import time
import os
import sys

# Set environment variables before importing DPU libraries
os.environ['XLNX_VART_FIRMWARE'] = '/lib/firmware/xilinx/b4096_300m/binary_container_1.bin'
os.environ['XILINX_XRT'] = '/usr'

try:
    import xir
    import vart
except ImportError as e:
    print(f"Error importing libraries: {e}")
    print("Please ensure VART is installed")
    sys.exit(1)

app = Flask(__name__)
output_frame = None
lock = threading.Lock()
model_loaded = False
error_message = ""

# COCO classes
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

class YOLOv8DPU:
    def __init__(self, model_path):
        """Initialize with extensive error checking"""
        self.runner = None
        self.input_shape = None
        
        # Check model file
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        print("Loading YOLOv8n model...")
        
        try:
            # Load graph
            self.graph = xir.Graph.deserialize(model_path)
            subgraphs = self.graph.get_root_subgraph().toposort_child_subgraph()
            
            if len(subgraphs) == 0:
                raise ValueError("No subgraphs found in model")
            
            # Find DPU subgraph
            self.dpu_subgraph = None
            for sg in subgraphs:
                if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                    self.dpu_subgraph = sg
                    print(f"Found DPU subgraph: {sg.get_name()}")
                    break
                    
            if not self.dpu_subgraph:
                raise ValueError("No DPU subgraph found in model")
            
            # Create runner with retry
            max_retries = 3
            for i in range(max_retries):
                try:
                    self.runner = vart.Runner.create_runner(self.dpu_subgraph, "run")
                    print("Runner created successfully")
                    break
                except Exception as e:
                    print(f"Runner creation attempt {i+1} failed: {e}")
                    if i == max_retries - 1:
                        raise
                    time.sleep(1)
            
            # Get tensor info
            self.input_tensors = self.runner.get_input_tensors()
            self.output_tensors = self.runner.get_output_tensors()
            
            if not self.input_tensors:
                raise ValueError("No input tensors found")
                
            # Get input shape
            self.input_shape = tuple(self.input_tensors[0].dims)
            print(f"Model input shape: {self.input_shape}")
            
        except Exception as e:
            print(f"Model initialization error: {e}")
            raise
            
    def preprocess(self, image):
        """Preprocess with error checking"""
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
            
        try:
            # Get target size
            if len(self.input_shape) == 4:
                input_h, input_w = self.input_shape[1], self.input_shape[2]
            else:
                input_h = input_w = 640  # Default
                
            # Resize
            resized = cv2.resize(image, (input_w, input_h))
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Add batch dimension
            batch = np.expand_dims(rgb, axis=0).astype(np.uint8)
            
            return batch
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            raise
            
    def run_inference(self, image):
        """Run inference with error handling"""
        if self.runner is None:
            raise RuntimeError("Runner not initialized")
            
        try:
            # Preprocess
            input_data = self.preprocess(image)
            
            # Allocate output buffers
            output_buffers = []
            for tensor in self.output_tensors:
                shape = tuple(tensor.dims)
                dtype = np.float32
                buffer = np.empty(shape, dtype=dtype)
                output_buffers.append(buffer)
            
            # Execute
            job_id = self.runner.execute_async(input_data, output_buffers)
            self.runner.wait(job_id)
            
            return output_buffers
            
        except Exception as e:
            print(f"Inference error: {e}")
            return None

    def postprocess(self, outputs, img_shape, conf_threshold=0.5):
        """Safe postprocessing"""
        if outputs is None or len(outputs) == 0:
            return []
            
        detections = []
        
        try:
            output = outputs[0]
            
            # Handle different output formats
            if len(output.shape) == 3:
                if output.shape[1] == 84:  # [1, 84, N]
                    output = output.transpose(0, 2, 1)  # -> [1, N, 84]
                predictions = output[0]
            else:
                return []
            
            # Process predictions
            for i in range(min(predictions.shape[0], 1000)):  # Limit iterations
                pred = predictions[i]
                
                if len(pred) < 84:
                    continue
                    
                # Extract coordinates
                cx, cy, w, h = pred[:4]
                
                # Extract class scores
                class_scores = pred[4:]
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]
                
                if confidence > conf_threshold:
                    # Scale to image
                    img_h, img_w = img_shape[:2]
                    scale_w = img_w / self.input_shape[2]
                    scale_h = img_h / self.input_shape[1]
                    
                    # Convert to corners
                    x1 = int((cx - w/2) * scale_w)
                    y1 = int((cy - h/2) * scale_h)
                    x2 = int((cx + w/2) * scale_w)
                    y2 = int((cy + h/2) * scale_h)
                    
                    # Clip to image
                    x1 = max(0, min(x1, img_w))
                    y1 = max(0, min(y1, img_h))
                    x2 = max(0, min(x2, img_w))
                    y2 = max(0, min(y2, img_h))
                    
                    if x2 > x1 and y2 > y1:  # Valid box
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'score': float(confidence),
                            'class': int(class_id) % len(COCO_CLASSES)
                        })
                        
        except Exception as e:
            print(f"Postprocess error: {e}")
            
        return detections
        
    def draw_detections(self, image, detections):
        """Safe drawing"""
        try:
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                score = det['score']
                class_id = det['class']
                
                # Draw box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                if class_id < len(COCO_CLASSES):
                    label = f"{COCO_CLASSES[class_id]}: {score:.2f}"
                else:
                    label = f"Class{class_id}: {score:.2f}"
                    
                cv2.putText(image, label, (x1, max(y1-5, 20)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                           
        except Exception as e:
            print(f"Draw error: {e}")
            
        return image

def check_camera():
    """Find available camera"""
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"Found camera at /dev/video{i}")
                return i
    return None

def capture_and_process():
    """Main loop with comprehensive error handling"""
    global output_frame, lock, model_loaded, error_message
    
    # Check camera first
    camera_id = check_camera()
    if camera_id is None:
        error_message = "No camera found"
        print(error_message)
        return
        
    # Initialize model with error handling
    model = None
    try:
        # Check DPU
        if not os.path.exists("/dev/dpu"):
            print("Warning: /dev/dpu not found, but continuing...")
            
        # Load model
        model = YOLOv8DPU("yolov8n_kv260.xmodel")
        model_loaded = True
        
    except Exception as e:
        error_message = f"Model load failed: {str(e)}"
        print(error_message)
        model_loaded = False
        
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    fps_time = time.time()
    frame_count = 0
    fps = 0
    
    print("Starting capture loop...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        result_frame = frame.copy()
        
        # If model loaded, run inference
        if model_loaded and model is not None:
            try:
                # Run inference
                start_time = time.time()
                outputs = model.run_inference(frame)
                
                if outputs is not None:
                    detections = model.postprocess(outputs, frame.shape, conf_threshold=0.5)
                    result_frame = model.draw_detections(result_frame, detections)
                    inference_time = (time.time() - start_time) * 1000
                    
                    # Add info
                    info = f"FPS: {fps} | Time: {inference_time:.1f}ms | Objects: {len(detections)}"
                else:
                    info = f"FPS: {fps} | Inference failed"
                    
            except Exception as e:
                info = f"FPS: {fps} | Error: {str(e)[:30]}"
                
        else:
            # No model, just show camera
            info = f"FPS: {fps} | Model not loaded: {error_message[:30]}"
            
        # Draw info
        cv2.putText(result_frame, info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Calculate FPS
        frame_count += 1
        if time.time() - fps_time > 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()
            
        # Update global frame
        with lock:
            output_frame = result_frame.copy()

def generate():
    """Generate MJPEG stream"""
    global output_frame, lock
    
    # Wait for first frame
    while output_frame is None:
        time.sleep(0.1)
        
    while True:
        with lock:
            if output_frame is not None:
                ret, jpeg = cv2.imencode('.jpg', output_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + 
                           jpeg.tobytes() + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    global model_loaded, error_message
    status = "Model loaded" if model_loaded else f"Error: {error_message}"
    
    return f'''
    <html>
        <head>
            <title>YOLOv8n on KV260</title>
            <style>
                body {{ text-align: center; font-family: Arial; }}
                .status {{ color: {'green' if model_loaded else 'red'}; }}
            </style>
        </head>
        <body>
            <h1>YOLOv8n Object Detection</h1>
            <p class="status">{status}</p>
            <img src="/video_feed" width="640" height="480">
        </body>
    </html>
    '''

if __name__ == '__main__':
    # Start capture thread
    t = threading.Thread(target=capture_and_process)
    t.daemon = True
    t.start()
    
    # Start server
    print("\nStarting server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
