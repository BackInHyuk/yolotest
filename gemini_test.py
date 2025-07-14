#!/usr/bin/env python3
"""
YOLOv8n Object Detection - Data Type Error Bypass Version
"""

from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import threading
import time
import os
import sys

# Critical environment setup for Vitis AI
os.environ['XLNX_VART_FIRMWARE'] = '/lib/firmware/xilinx/b4096_300m/binary_container_1.bin'
os.environ['XILINX_XRT'] = '/usr'
os.environ['XLNX_ENABLE_FINGERPRINT_CHECK'] = '0' # Disable fingerprint check for flexibility
os.environ['XLNX_ENABLE_STAT_MODE'] = '0' # Disable statistics mode
os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0' # Prioritize V4L2 over GStreamer

# Import DPU libraries with specific error handling
try:
    import xir
    import vart
    print("DPU libraries (xir, vart) imported successfully.")
except ImportError as e:
    print(f"ERROR: Cannot import DPU libraries. Please ensure Vitis AI Runtime (VART) is installed and environment variables are set correctly. Error: {e}")
    sys.exit(1)

app = Flask(__name__)
output_frame = None
lock = threading.Lock()

# COCO dataset classes for YOLOv8
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
    """
    A class to run YOLOv8n model on Xilinx DPU, handling common Vitis AI
    data type and runner initialization issues.
    """
    
    def __init__(self, model_path):
        self.ready = False
        self.model_path = model_path
        self.input_width = 640  # Default YOLOv8 input width
        self.input_height = 640 # Default YOLOv8 input height
        self.input_scale = 1.0  # Scale for input image quantization
        self.output_scale = 1.0 # Scale for output de-quantization
        self.input_dtype = None # DPU input data type
        self.output_dtype = None # DPU output data type

        print(f"Attempting to load model: {model_path}")
        
        try:
            # Deserialize the XIR graph from the .xmodel file
            self.graph = xir.Graph.deserialize(model_path)
            root = self.graph.get_root_subgraph()
            children = root.toposort_child_subgraph()
            
            # Find the DPU subgraph within the model
            self.dpu_subgraph = None
            for sg in children:
                if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                    self.dpu_subgraph = sg
                    print(f"Found DPU subgraph: {sg.dpu_kernel.name if sg.has_attr('dpu_kernel') else sg.get_name()}")
                    break
            
            if not self.dpu_subgraph:
                raise Exception("No DPU subgraph found in the .xmodel. Please ensure the model was compiled for DPU.")
            
            # Create the DPU runner. Reverting to "run" mode as per original code.
            # This helps in cases where the VART expects a specific mode for runner creation.
            self.runner = vart.Runner.create_runner(self.dpu_subgraph, "run")
            print("DPU Runner created successfully with 'run' mode.")

            # Get input and output tensor information directly from the runner.
            # This is crucial for handling data types and scaling.
            self.input_tensors = self.runner.get_input_tensors()
            self.output_tensors = self.runner.get_output_tensors()
            
            if not self.input_tensors or not self.output_tensors:
                raise Exception("Could not retrieve input/output tensors from the DPU runner.")
            
            # --- Input Tensor Information ---
            input_tensor_meta = self.input_tensors[0]
            self.input_dims = input_tensor_meta.dims
            self.input_height = self.input_dims[1] # Assuming NCHW or NHWC, H is 2nd dim
            self.input_width = self.input_dims[2]  # W is 3rd dim
            self.input_dtype = input_tensor_meta.dtype
            input_fix_point = input_tensor_meta.get_attr("fix_point") if input_tensor_meta.has_attr("fix_point") else 0

            # Calculate input scale for quantization (if INT8)
            # This is crucial: image (0-255) needs to be scaled to the DPU's INT8 range.
            # The exact formula depends on your quantization toolchain's preprocessing.
            # A common pattern for Vitis AI is (2**fix_point) / 255.0 if input is 0-255.
            if self.input_dtype.name == 'INT8':
                self.input_scale = (2**input_fix_point) / 255.0
            else: # If float32, usually input is normalized to 0-1
                self.input_scale = 1.0 / 255.0 # For float32, normalize 0-255 to 0-1
            
            print(f"Input Tensor: Name={input_tensor_meta.name}, Shape={self.input_dims}, DType={self.input_dtype.name}, FixPoint={input_fix_point}, Calculated Input Scale={self.input_scale:.4f}")

            # --- Output Tensor Information ---
            output_tensor_meta = self.output_tensors[0]
            self.output_dims = output_tensor_meta.dims
            self.output_dtype = output_tensor_meta.dtype
            output_fix_point = output_tensor_meta.get_attr("fix_point") if output_tensor_meta.has_attr("fix_point") else 0

            # Calculate output scale for de-quantization (if INT8)
            # This converts DPU's INT8 output back to float32.
            if self.output_dtype.name == 'INT8':
                self.output_scale = 1.0 / (2**output_fix_point)
            else: # If float32, no de-quantization needed
                self.output_scale = 1.0
            
            print(f"Output Tensor: Name={output_tensor_meta.name}, Shape={self.output_dims}, DType={self.output_dtype.name}, FixPoint={output_fix_point}, Calculated Output Scale={self.output_scale:.4f}")

            self.ready = True
            self.use_fallback = False # If runner created successfully, no fallback
            
        except Exception as e:
            print(f"ERROR: DPU initialization failed: {e}")
            self.ready = False
            self.use_fallback = True # Enable fallback on initialization failure
            print("DPU initialization failed, falling back to dummy detection for testing.")

    def detect_safe(self, image):
        """
        Performs object detection using the DPU model or a fallback.
        Handles image preprocessing, DPU inference, and post-processing.
        """
        if not self.ready:
            return self.fallback_detect(image)
        
        try:
            # 1. Preprocess the input image
            # Resize to DPU input dimensions (e.g., 640x640)
            resized_image = cv2.resize(image, (self.input_width, self.input_height))
            # Convert BGR to RGB (DPU models often expect RGB)
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
            # 2. Prepare input data for DPU
            # Apply scaling and convert to the DPU's expected input data type (e.g., INT8)
            if self.input_dtype.name == 'INT8':
                # Convert to float32, normalize to 0-1, then scale by 2^fix_point and cast to int8
                # This matches common Vitis AI quantization flow for images
                input_data = (rgb_image.astype(np.float32) * self.input_scale).astype(np.int8)
            elif self.input_dtype.name == 'FLOAT32':
                # If the DPU expects float32, just normalize to 0-1
                input_data = rgb_image.astype(np.float32) * self.input_scale
            else:
                raise ValueError(f"Unsupported input data type from DPU model: {self.input_dtype.name}")

            # Add batch dimension (e.g., from (H, W, C) to (1, H, W, C))
            input_data = np.expand_dims(input_data, axis=0)
            
            # 3. Allocate output buffer for DPU
            # Allocate with the DPU's native output data type (e.g., INT8)
            output_buffer = np.zeros(self.output_dims, dtype=self.output_dtype.as_numpy_dtype)
            
            # 4. Execute DPU inference
            # Using execute_async for non-blocking execution, then wait for completion
            job_id = self.runner.execute_async([input_data], [output_buffer])
            self.runner.wait(job_id)
            
            # 5. Post-process DPU output
            # De-quantize the output buffer back to float32 using the calculated output_scale
            if self.output_dtype.name == 'INT8':
                # Apply de-quantization scale to convert INT8 values to float32
                outputs_dequantized = [output_buffer[0] * self.output_scale]
            elif self.output_dtype.name == 'FLOAT32':
                # If output is already float32, just take the array
                outputs_dequantized = [output_buffer[0]]
            else:
                raise ValueError(f"Unsupported output data type from DPU model: {self.output_dtype.name}")
            
            # Process the de-quantized outputs (NMS, bounding box calculations)
            detections = self.process_outputs(outputs_dequantized, image.shape)
            
            # Draw bounding boxes and labels on the original image
            result_frame = self.draw_detections(image.copy(), detections)
            
            return result_frame, detections
        
        except Exception as e:
            print(f"ERROR: Inference error during DPU run: {e}. Falling back to dummy detection.")
            self.use_fallback = True # Enable fallback if DPU inference fails
            return self.fallback_detect(image)

    def fallback_detect(self, image):
        """
        Provides a dummy detection for testing or when DPU inference fails.
        Simulates detecting a 'person' in the center of the frame.
        """
        h, w = image.shape[:2]
        
        detections = []
        # Always detect a "person" in center for testing
        detections.append({
            'bbox': [w//4, h//4, 3*w//4, 3*h//4], # [x1, y1, x2, y2]
            'score': 0.75,
            'class': 0  # person
        })
        
        result = self.draw_detections(image.copy(), detections)
        return result, detections

    def process_outputs(self, outputs_raw, img_shape):
        """
        Processes the raw model outputs to extract bounding boxes, scores, and class IDs.
        Applies Non-Maximum Suppression (NMS) to filter overlapping boxes.
        """
        detections = []
        try:
            if not outputs_raw or len(outputs_raw) == 0:
                print("Warning: No raw outputs from model.")
                return detections
            
            # YOLOv8 output tensor typically has shape (1, 84, 8400) or (1, 8400, 84)
            # where 84 = 4 (bbox) + 80 (classes)
            output = outputs_raw[0] 
            
            # Adjust output shape to (Num_Boxes, 84) for easier processing
            if output.ndim == 2: # Already (Num_Boxes, 84) or (84, Num_Boxes)
                predictions = output
            elif output.ndim == 3: # (Batch, Channels, Num_Boxes) or (Batch, Num_Boxes, Channels)
                if output.shape[1] == 84 and output.shape[2] > 84: # (1, 84, 8400) -> transpose to (8400, 84)
                    predictions = output[0].transpose(1, 0) 
                elif output.shape[2] == 84 and output.shape[1] > 84: # (1, 8400, 84) -> take batch 0
                    predictions = output[0]
                else:
                    print(f"Warning: Unexpected output shape from DPU: {output.shape}. Skipping processing.")
                    return detections
            else:
                print(f"Warning: Unexpected output dimensions from DPU: {output.ndim}. Skipping processing.")
                return detections
            
            # Ensure predictions are float32 for calculations
            predictions = predictions.astype(np.float32)

            boxes = []
            scores = []
            class_ids = []

            # Iterate through each prediction row
            for i in range(predictions.shape[0]): 
                row = predictions[i]
                
                # Check if row has enough elements for bbox and class scores (4 bbox + 80 classes = 84)
                if len(row) >= 84:
                    # YOLOv8 output format: [cx, cy, w, h, class_score_0, ..., class_score_79]
                    cx, cy, w, h = row[0:4] # Center_x, Center_y, Width, Height
                    class_scores = row[4:84] # Scores for 80 COCO classes
                    
                    class_id = np.argmax(class_scores)
                    confidence = class_scores[class_id] # Confidence is the highest class score
                    
                    # Apply a confidence threshold to filter weak detections
                    if confidence > 0.4: # Adjustable threshold
                        # Scale bounding box coordinates back to original image dimensions
                        scale_x = img_shape[1] / self.input_width
                        scale_y = img_shape[0] / self.input_height
                        
                        # Convert center-width-height to x1, y1, x2, y2 (top-left, bottom-right)
                        x1 = int((cx - w/2) * scale_x)
                        y1 = int((cy - h/2) * scale_y)
                        x2 = int((cx + w/2) * scale_x)
                        y2 = int((cy + h/2) * scale_y)
                        
                        # Clip coordinates to ensure they are within image boundaries
                        x1 = max(0, min(x1, img_shape[1]))
                        y1 = max(0, min(y1, img_shape[0]))
                        x2 = max(0, min(x2, img_shape[1]))
                        y2 = max(0, min(y2, img_shape[0]))
                        
                        # Ensure valid bounding box dimensions
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            scores.append(float(confidence))
                            class_ids.append(int(class_id))
            
            # Apply Non-Maximum Suppression (NMS) to remove redundant overlapping boxes
            if len(boxes) > 0:
                boxes_np = np.array(boxes)
                scores_np = np.array(scores)
                class_ids_np = np.array(class_ids)

                final_detections = []
                # Perform NMS class-wise
                for class_id_val in np.unique(class_ids_np):
                    indices = np.where(class_ids_np == class_id_val)[0]
                    selected_boxes = boxes_np[indices]
                    selected_scores = scores_np[indices]

                    # Convert x1, y1, x2, y2 to x, y, width, height for OpenCV's NMSBoxes
                    boxes_xywh = np.array([[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in selected_boxes])
                    
                    # NMSBoxes returns indices of boxes to keep
                    # score_threshold: minimum confidence to consider a box
                    # nms_threshold: IoU threshold for overlapping boxes
                    indices_keep = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), selected_scores.tolist(), score_threshold=0.4, nms_threshold=0.45) 
                    
                    if len(indices_keep) > 0:
                        for idx in indices_keep.flatten(): # Flatten the indices array
                            original_idx = indices[idx] # Get the index in the original `detections` list
                            final_detections.append({
                                'bbox': boxes[original_idx],
                                'score': scores[original_idx],
                                'class': class_ids[original_idx]
                            })
                detections = final_detections

        except Exception as e:
            print(f"ERROR: Output processing error: {e}")
            
        return detections

    def draw_detections(self, image, detections):
        """
        Draws bounding boxes, labels, and confidence scores on the image.
        """
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            class_id = det['class']
            
            # Choose color based on class_id (e.g., green for person, red for others)
            color = (0, 255, 0) if class_id == 0 else (255, 0, 0)
            
            # Draw bounding box rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label = f"{COCO_CLASSES[class_id]}: {score:.2f}"
            
            # Put text label on the image
            cv2.putText(image, label, (x1, max(y1-5, 20)), # Position text above the box, or at y=20 if too high
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            
        return image

def find_camera():
    """
    Attempts to find an available camera device by iterating through /dev/videoX.
    """
    print("Searching for camera device...")
    for i in range(4): # Check up to /dev/video3
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2) # Use CAP_V4L2 backend for Linux
        if cap.isOpened():
            ret, _ = cap.read() # Try to read a frame to confirm it's working
            if ret:
                cap.release()
                print(f"Found camera at /dev/video{i}")
                return i
            cap.release()
    print("No camera found on /dev/video0 to /dev/video3.")
    return None

def camera_thread():
    """
    Main thread for camera capture, DPU inference, and frame update.
    """
    global output_frame, lock
    
    # Initialize the YOLOv8 DPU model runner
    model = YOLOv8SafeRunner("yolov8n_kv260.xmodel")
    
    # Find and open the camera
    cam_id = find_camera()
    if cam_id is None:
        print("ERROR: Exiting camera thread - No camera found!")
        return
    
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    # Set camera properties for consistent input
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce buffer size for lower latency
    
    fps = 0
    fps_time = time.time()
    frame_count = 0
    
    print("DPU Object Detection started. Capturing from camera...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from camera. Retrying...")
            time.sleep(0.1) # Wait a bit before retrying
            continue
        
        # Perform object detection
        start_time = time.time()
        result_frame, detections = model.detect_safe(frame)
        inference_time = (time.time() - start_time) * 1000 # Inference time in ms
        
        # Calculate FPS
        frame_count += 1
        if time.time() - fps_time > 1.0:
            fps = frame_count
            frame_count = 0
            fps_time = time.time()
        
        # Display performance info on the frame
        info = f"FPS: {fps} | Time: {inference_time:.1f}ms | Objects: {len(detections)}"
        cv2.putText(result_frame, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update the global output_frame for the web stream
        with lock:
            output_frame = result_frame.copy()

def generate():
    """
    Generator function to create an MJPEG stream for Flask web server.
    """
    global output_frame, lock
    
    while True:
        with lock:
            if output_frame is None:
                # If no frame is available yet, wait briefly
                time.sleep(0.05)
                continue
            
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', output_frame)
            if not ret:
                print("Warning: Failed to encode frame to JPEG.")
                time.sleep(0.05)
                continue
        
        # Yield the JPEG frame with MJPEG headers
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.03) # Control stream frame rate

@app.route('/')
def index():
    """
    Flask route for the main web page, displaying the video feed.
    """
    return render_template_string('''
    <html>
    <head>
        <title>YOLOv8n KV260 Object Detection</title>
        <style>
            body { 
                font-family: 'Arial', sans-serif; 
                text-align: center; 
                background: #282c34; 
                color: #e0e0e0; 
                margin: 0; 
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
            }
            h1 { 
                color: #61dafb; 
                margin-bottom: 20px; 
                font-size: 2.5em;
            }
            img { 
                border: 3px solid #61dafb; 
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                max-width: 100%; /* Responsive image */
                height: auto;
            }
            p { 
                margin-top: 20px; 
                font-size: 1.1em; 
                color: #bbbbbb; 
            }
            .info-box {
                background-color: #3a3f47;
                padding: 15px 25px;
                border-radius: 8px;
                margin-top: 30px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .info-box h2 {
                color: #a0a0a0;
                font-size: 1.2em;
                margin-bottom: 10px;
            }
            .info-box ul {
                list-style: none;
                padding: 0;
                margin: 0;
            }
            .info-box li {
                margin-bottom: 5px;
                color: #e0e0e0;
            }
        </style>
    </head>
    <body>
        <h1>YOLOv8n Object Detection on KV260</h1>
        <img src="/video_feed" width="640" height="480">
        <p>Real-time object detection using a quantized YOLOv8n model on Xilinx KV260.</p>
        <div class="info-box">
            <h2>Detected COCO Classes:</h2>
            <ul>
                <li>Person, Car, Bicycle, Motorcycle, Bus, Truck, Train, Boat, Traffic Light, Fire Hydrant, Stop Sign, Bench, Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe, Backpack, Umbrella, Handbag, Tie, Suitcase, Frisbee, Skis, Snowboard, Sports Ball, Kite, Baseball Bat, Baseball Glove, Skateboard, Surfboard, Tennis Racket, Bottle, Wine Glass, Cup, Fork, Knife, Spoon, Bowl, Banana, Apple, Sandwich, Orange, Broccoli, Carrot, Hot Dog, Pizza, Donut, Cake, Chair, Couch, Potted Plant, Bed, Dining Table, Toilet, TV, Laptop, Mouse, Remote, Keyboard, Cell Phone, Microwave, Oven, Toaster, Sink, Refrigerator, Book, Clock, Vase, Scissors, Teddy Bear, Hair Drier, Toothbrush</li>
            </ul>
        </div>
    </body>
    </html>
    ''')

@app.route('/video_feed')
def video_feed():
    """
    Flask route to stream the video feed as MJPEG.
    """
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the camera and DPU inference thread
    t = threading.Thread(target=camera_thread)
    t.daemon = True # Allow the main program to exit even if this thread is running
    t.start()
    
    # Give the camera thread a moment to initialize
    time.sleep(2) 
    
    print("\nWeb server starting. Access the video feed at: http://[KV260_IP]:5000\n")
    # Run the Flask web server
    app.run(host='0.0.0.0', port=5000, debug=
