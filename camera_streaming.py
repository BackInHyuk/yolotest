#!/usr/bin/env python3
"""
Complete YOLOv8n object detection with MJPEG streaming
"""

from flask import Flask, Response
import cv2
import numpy as np
import xir
import vart
import threading
import time

app = Flask(__name__)
output_frame = None
lock = threading.Lock()

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
        print("Loading YOLOv8n model...")
        self.graph = xir.Graph.deserialize(model_path)
        subgraphs = self.graph.get_root_subgraph().toposort_child_subgraph()
        
        # Find DPU subgraph
        self.dpu_subgraph = None
        for sg in subgraphs:
            if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                self.dpu_subgraph = sg
                break
                
        if not self.dpu_subgraph:
            raise Exception("No DPU subgraph found!")
            
        self.runner = vart.Runner.create_runner(self.dpu_subgraph, "run")
        self.input_tensors = self.runner.get_input_tensors()
        self.output_tensors = self.runner.get_output_tensors()
        
        # Get input shape [N, H, W, C]
        self.input_shape = tuple(self.input_tensors[0].dims)
        print(f"Model input shape: {self.input_shape}")
        
    def preprocess(self, image):
        """Preprocess image for YOLOv8"""
        # Resize to model input size (usually 640x640)
        input_h, input_w = self.input_shape[1], self.input_shape[2]
        resized = cv2.resize(image, (input_w, input_h))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Add batch dimension and ensure uint8
        batch = np.expand_dims(rgb, axis=0).astype(np.uint8)
        return batch
    
    def run_inference(self, image):
        """Run DPU inference"""
        # Preprocess
        input_data = self.preprocess(image)
        
        # Create output arrays based on output tensor shapes
        outputs = []
        for tensor in self.output_tensors:
            shape = tuple(tensor.dims)
            output = np.empty(shape, dtype=np.float32)
            outputs.append(output)
        
        # Execute on DPU
        job_id = self.runner.execute_async(input_data, outputs)
        self.runner.wait(job_id)
        
        return outputs
    
    def postprocess(self, outputs, img_shape, conf_threshold=0.5):
        """Process YOLOv8 outputs"""
        detections = []
        
        # YOLOv8 output is typically [1, 84, 8400] or [1, 8400, 84]
        output = outputs[0]
        
        # Reshape if needed
        if output.shape[1] == 84:  # [1, 84, 8400]
            output = output.transpose(0, 2, 1)  # -> [1, 8400, 84]
        
        predictions = output[0]  # Remove batch dimension
        
        # Process each prediction
        for i in range(predictions.shape[0]):
            pred = predictions[i]
            
            # Extract box coordinates (normalized)
            cx, cy, w, h = pred[:4]
            
            # Extract class scores
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence > conf_threshold:
                # Convert normalized coordinates to pixel coordinates
                img_h, img_w = img_shape[:2]
                
                # Scale to image size
                cx *= img_w / self.input_shape[2]
                cy *= img_h / self.input_shape[1]
                w *= img_w / self.input_shape[2]
                h *= img_h / self.input_shape[1]
                
                # Convert center format to corner format
                x1 = int(cx - w/2)
                y1 = int(cy - h/2)
                x2 = int(cx + w/2)
                y2 = int(cy + h/2)
                
                # Clip to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_w, x2)
                y2 = min(img_h, y2)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': float(confidence),
                    'class': int(class_id)
                })
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            class_id = det['class']
            
            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{COCO_CLASSES[class_id]}: {score:.2f}"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1-25), (x1+label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x1, y1-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
        return image

# Global model instance
model = None

def capture_and_process():
    """Main processing loop"""
    global output_frame, lock, model
    
    # Initialize model
    model = YOLOv8DPU("yolov8n_kv260.xmodel")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    fps_time = time.time()
    frame_count = 0
    fps = 0
    
    print("Starting capture and inference loop...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Run YOLOv8 inference
        start_time = time.time()
        outputs = model.run_inference(frame)
        detections = model.postprocess(outputs, frame.shape, conf_threshold=0.5)
        inference_time = (time.time() - start_time) * 1000
        
        # Draw detections
        result_frame = frame.copy()
        result_frame = model.draw_detections(result_frame, detections)
        
        # Calculate FPS
        frame_count += 1
        if time.time() - fps_time > 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()
        
        # Add info overlay
        info_text = f"FPS: {fps} | Inference: {inference_time:.1f}ms | Objects: {len(detections)}"
        cv2.putText(result_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update global frame
        with lock:
            output_frame = result_frame.copy()
            
        # Debug print every 30 frames
        if frame_count % 30 == 0:
            print(f"FPS: {fps}, Detections: {len(detections)}")
            for det in detections:
                print(f"  - {COCO_CLASSES[det['class']]}: {det['score']:.2f}")

def generate():
    """Generate MJPEG stream"""
    global output_frame, lock
    
    while True:
        with lock:
            if output_frame is None:
                continue
                
            # Encode as JPEG
            ret, jpeg = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
                
        # Yield frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>YOLOv8n Live Detection on KV260</title>
            <style>
                body { font-family: Arial; text-align: center; background: #f0f0f0; }
                h1 { color: #333; }
                .container { margin: 20px auto; }
                img { border: 2px solid #333; }
            </style>
        </head>
        <body>
            <h1>YOLOv8n Object Detection - KV260 DPU</h1>
            <div class="container">
                <img src="/video_feed" width="640" height="480">
            </div>
            <p>Real-time object detection using YOLOv8n on Xilinx KV260</p>
        </body>
    </html>
    '''

if __name__ == '__main__':
    # Start capture thread
    t = threading.Thread(target=capture_and_process)
    t.daemon = True
    t.start()
    
    # Wait a bit for initialization
    time.sleep(2)
    
    # Run Flask server
    print("\nStarting web server...")
    print("Open browser and go to: http://[KV260_IP]:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
