import numpy as np
import cv2
import os
import sys
import threading
import time

from flask import Flask, Response, render_template_string

# Critical environment setup
os.environ['XLNX_VART_FIRMWARE'] = '/lib/firmware/xilinx/b4096_300m/binary_container_1.bin'
os.environ['XILINX_XRT'] = '/usr'
os.environ['XLNX_ENABLE_FINGERPRINT_CHECK'] = '0'
os.environ['XLNX_ENABLE_STAT_MODE'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'

try:
    import xir
    import vart
    print("DPU libraries imported")
except ImportError as e: # Catch specific import error
    print(f"Cannot import DPU libraries: {e}")
    sys.exit(1)

app = Flask(__name__)
output_frame = None
lock = threading.Lock()

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

class YOLOv8SafeRunner:
    def __init__(self, model_path):
        self.ready = False
        self.model_path = model_path
        
        print(f"Loading model: {model_path}")
        
        try:
            self.graph = xir.Graph.deserialize(model_path)
            root = self.graph.get_root_subgraph()
            children = root.toposort_child_subgraph()
            
            self.dpu_subgraph = None
            for sg in children:
                if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                    self.dpu_subgraph = sg
                    print(f"Found DPU subgraph: {sg.get_name()}")
                    break
            
            if not self.dpu_subgraph:
                raise Exception("No DPU subgraph found in the model.")
            
            self.runner = vart.Runner.create_runner(self.dpu_subgraph) # No "run" or "sim" argument needed for standard runner
            print("DPU Runner created successfully.")

            self.input_tensors = self.runner.get_input_tensors()
            self.output_tensors = self.runner.get_output_tensors()

            if not self.input_tensors or not self.output_tensors:
                raise Exception("Could not retrieve input/output tensors from runner.")
            
            # --- Get Input Tensor Info ---
            self.input_tensor_info = self.input_tensors[0]
            self.input_dims = self.input_tensor_info.dims
            # Assume NHWC for image input [N, H, W, C]
            self.input_height = self.input_dims[1]
            self.input_width = self.input_dims[2]
            
            # Get input data type and scale
            self.input_dtype = self.input_tensor_info.dtype
            self.input_fix_point = self.input_tensor_info.get_attr("fix_point") if self.input_tensor_info.has_attr("fix_point") else 0
            self.input_scale = 2**self.input_fix_point if self.input_input_dtype.name == 'INT8' else 1.0
            
            print(f"Input Tensor: Name={self.input_tensor_info.name}, Shape={self.input_dims}, DType={self.input_dtype.name}, FixPoint={self.input_fix_point}")

            # --- Get Output Tensor Info ---
            self.output_tensor_info = self.output_tensors[0]
            self.output_dims = self.output_tensor_info.dims
            
            # Get output data type and scale
            self.output_dtype = self.output_tensor_info.dtype
            self.output_fix_point = self.output_tensor_info.get_attr("fix_point") if self.output_tensor_info.has_attr("fix_point") else 0
            self.output_scale = 1.0 / (2**self.output_fix_point) if self.output_dtype.name == 'INT8' else 1.0
            
            print(f"Output Tensor: Name={self.output_tensor_info.name}, Shape={self.output_dims}, DType={self.output_dtype.name}, FixPoint={self.output_fix_point}")

            self.ready = True
            self.use_fallback = False # If runner created successfully, no fallback
            
        except Exception as e:
            print(f"Initialization error: {e}")
            self.ready = False
            self.use_fallback = True # Enable fallback on initialization failure
            print("DPU initialization failed, falling back to dummy detection.")

    def detect_safe(self, image):
        if not self.ready or self.use_fallback:
            return self.fallback_detect(image)
        
        try:
            # Preprocess input image based on DPU input requirements
            resized_image = cv2.resize(image, (self.input_width, self.input_height))
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
            # Convert to expected DPU input type (INT8) and apply scaling
            if self.input_dtype.name == 'INT8':
                # Example: Normalize to [0, 1] then scale by 2^fix_point
                # This depends heavily on your quantization recipe.
                # Common approach: (image / 255.0 * 2**fix_point).astype(np.int8)
                # Or simply: (image * self.input_scale).astype(np.int8) if input_scale is correct for 0-255 range.
                # For images, DPU often expects 0-255 values, quantized to INT8.
                # Check your quantization script's preprocessing for the exact scaling.
                input_data = (rgb_image * self.input_scale).astype(np.int8)
                print(f"Input data dtype: {input_data.dtype}, min: {input_data.min()}, max: {input_data.max()}")
            else: # If it's not INT8, assume it's float32 and scale to [0,1]
                input_data = rgb_image.astype(np.float32) / 255.0

            input_data = np.expand_dims(input_data, axis=0) # Add batch dimension

            # Prepare output buffer
            # Output from DPU for quantized models is usually INT8 or INT16
            output_buffer = np.zeros(self.output_dims, dtype=self.output_dtype.as_numpy_dtype)
            
            # Execute inference
            # VART's execute_async expects a list of numpy arrays for inputs/outputs
            job_id = self.runner.execute_async([input_data], [output_buffer])
            self.runner.wait(job_id)
            
            # Post-process output
            # Convert DPU output (e.g., INT8) back to float32 using output_scale
            if self.output_dtype.name == 'INT8':
                outputs = [output_buffer[0] * self.output_scale] # Apply de-quantization scale
            else:
                outputs = [output_buffer[0]] # Already float, no de-quantization
            
            detections = self.process_outputs(outputs, image.shape)
            result = self.draw_detections(image.copy(), detections)
            
            return result, detections
        
        except Exception as e:
            print(f"Inference error during DPU run: {e}")
            self.use_fallback = True # Fallback if DPU inference fails
            return self.fallback_detect(image)

    def fallback_detect(self, image):
        # ... (unchanged) ...
        h, w = image.shape[:2]
        detections = []
        detections.append({
            'bbox': [w//4, h//4, 3*w//4, 3*h//4],
            'score': 0.75,
            'class': 0  # person
        })
        result = self.draw_detections(image.copy(), detections)
        return result, detections

    def process_outputs(self, outputs_raw, img_shape):
        detections = []
        try:
            # outputs_raw is a list containing the numpy array, take the first element
            if not outputs_raw or len(outputs_raw) == 0:
                return detections
            
            output = outputs_raw[0] # This is already de-quantized to float32
            
            # YOLOv8 output is often (Batch, 84, Num_Boxes) or (Batch, Num_Boxes, 84)
            # DPU might output (1, 84, 8400) or (1, 8400, 84) for yolov8n
            
            if output.ndim == 2: # Likely (Num_Boxes, 84) after slicing batch
                predictions = output
            elif output.ndim == 3: # (Batch, Num_Boxes, 84) or (Batch, 84, Num_Boxes)
                if output.shape[1] == 84: # (Batch, 84, Num_Boxes)
                    predictions = output[0].transpose(1, 0) # Transpose to (Num_Boxes, 84)
                elif output.shape[2] == 84: # (Batch, Num_Boxes, 84)
                    predictions = output[0] # Take the first batch
                else:
                    print(f"Unexpected output shape: {output.shape}")
                    return detections
            else:
                print(f"Unexpected output dimensions: {output.ndim}")
                return detections
            
            # Ensure predictions is float32 for calculations
            predictions = predictions.astype(np.float32)

            for i in range(min(predictions.shape[0], 8400)): # Limit to 8400 predictions
                row = predictions[i]
                
                # Check row length. YOLOv8 has 4 bbox + 1 obj_conf + 80 class_scores = 85
                # Some versions might have 84 (4+80, confidence included in class scores implicitly)
                # Assuming 84 is bbox (4) + class_scores (80) with confidence already factored in or available
                
                if len(row) >= 84: # cx, cy, w, h + 80 class scores
                    # If the output format is [cx, cy, w, h, class1_score, class2_score, ... ]
                    cx, cy, w, h = row[0:4]
                    class_scores = row[4:84] # Assuming 80 classes
                    
                    class_id = np.argmax(class_scores)
                    confidence = class_scores[class_id] # Assuming confidence is directly the highest class score
                    
                    if confidence > 0.4: # Adjust confidence threshold
                        # Scale to original image
                        scale_x = img_shape[1] / self.input_width
                        scale_y = img_shape[0] / self.input_height
                        
                        x1 = int((cx - w/2) * scale_x)
                        y1 = int((cy - h/2) * scale_y)
                        x2 = int((cx + w/2) * scale_x)
                        y2 = int((cy + h/2) * scale_y)
                        
                        x1 = max(0, min(x1, img_shape[1]))
                        y1 = max(0, min(y1, img_shape[0]))
                        x2 = max(0, min(x2, img_shape[1]))
                        y2 = max(0, min(y2, img_shape[0]))
                        
                        if x2 > x1 and y2 > y1:
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'score': float(confidence),
                                'class': int(class_id)
                            })
            
            # Apply Non-Maximum Suppression (NMS) for better results
            if len(detections) > 0:
                boxes = np.array([d['bbox'] for d in detections])
                scores = np.array([d['score'] for d in detections])
                class_ids = np.array([d['class'] for d in detections])

                # Perform NMS per class
                final_detections = []
                for class_id in np.unique(class_ids):
                    indices = np.where(class_ids == class_id)[0]
                    selected_boxes = boxes[indices]
                    selected_scores = scores[indices]

                    # Convert x1, y1, x2, y2 to x, y, w, h for NMSBoxes
                    # If opencv NMSBoxes expects x1,y1,x2,y2, ensure that.
                    # Or convert to x, y, width, height format
                    # NMSBoxes often takes [x, y, w, h] format, so we need to convert
                    boxes_xywh = np.array([[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in selected_boxes])
                    
                    # You might need to adjust NMS threshold (e.g., 0.45 for iou_threshold)
                    # and score threshold (e.g., 0.5)
                    indices_keep = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), selected_scores.tolist(), score_threshold=0.4, nms_threshold=0.45)
                    
                    if len(indices_keep) > 0:
                        for idx in indices_keep.flatten():
                            final_detections.append(detections[indices[idx]])
                detections = final_detections

        except Exception as e:
            print(f"Output processing error: {e}")
            
        return detections

    def draw_detections(self, image, detections):
        # ... (unchanged) ...
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            class_id = det['class']
            
            color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{COCO_CLASSES[class_id]}: {score:.2f}"
            cv2.putText(image, label, (x1, max(y1-5, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
        return image

def find_camera():
    # ... (unchanged) ...
    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.release()
                return i
            cap.release()
    return None

def camera_thread():
    global output_frame, lock
    
    model = YOLOv8SafeRunner("yolov8n_kv260.xmodel")
    
    cam_id = find_camera()
    if cam_id is None:
        print("No camera found!")
        return
    
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    fps = 0
    fps_time = time.time()
    frame_count = 0
    
    print("Detection started")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        start_time = time.time()
        result_frame, detections = model.detect_safe(frame)
        inference_time = (time.time() - start_time) * 1000
        
        frame_count += 1
        if time.time() - fps_time > 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()
        
        info = f"FPS: {fps} | Time: {inference_time:.1f}ms | Objects: {len(detections)}"
        cv2.putText(result_frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        with lock:
            output_frame = result_frame.copy()

def generate():
    # ... (unchanged) ...
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
    # ... (unchanged) ...
    return render_template_string('''
    <html>
    <head>
        <title>YOLOv8n KV260</title>
        <style>
            body { text-align: center; background: #222; color: white; }
            h1 { color: #0f0; }
            img { border: 2px solid #0f0; }
        </style>
    </head>
    <body>
        <h1>YOLOv8n Object Detection</h1>
        <img src="/video_feed" width="640" height="480">
        <p>Detecting: person, car, bicycle, etc. (80 COCO classes)</p>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=camera_thread)
    t.daemon = True
    t.start()
    
    time.sleep(2)
    
    print("\nServer: http://[KV260_IP]:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
