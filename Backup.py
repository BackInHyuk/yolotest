#!/usr/bin/env python3
"""
Ultra-Safe Detection with Atomic File Operations
Uses double-buffering and atomic file operations to prevent corruption
"""

import cv2
import numpy as np
import time
import gc
import argparse
import os
import threading
import tempfile
import shutil
from flask import Flask, Response, render_template_string

try:
    import vart
    import xir
except ImportError:
    print("Vitis AI runtime not installed.")
    exit(1)

class UltraSafeDetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.dpu_runner = None
        self.input_tensors = None
        self.output_tensors = None
        
        # Load model with full compatibility
        self._load_model()
        
        # Stats
        self.inference_time = 0
        self.detection_count = 0
        
    def _load_model(self):
        try:
            graph = xir.Graph.deserialize(self.model_path)
            root_subgraph = graph.get_root_subgraph()
            
            if root_subgraph is None:
                raise ValueError("Failed to get root subgraph")
            
            # Get subgraphs with comprehensive compatibility
            subgraphs = []
            try:
                if hasattr(root_subgraph, 'children_topological_sort'):
                    children = root_subgraph.children_topological_sort()
                    if isinstance(children, (list, tuple)):
                        subgraphs = children
                    elif isinstance(children, set):
                        subgraphs = list(children)
                    else:
                        subgraphs = [children] if children else []
                elif hasattr(root_subgraph, 'get_children'):
                    children = root_subgraph.get_children()
                    if isinstance(children, (list, tuple)):
                        subgraphs = children
                    elif isinstance(children, set):
                        subgraphs = list(children)
                    else:
                        subgraphs = [children] if children else []
                else:
                    subgraphs = [root_subgraph]
            except Exception as e:
                print(f"Error getting subgraphs: {e}")
                subgraphs = [root_subgraph]
            
            if not subgraphs:
                raise ValueError("No subgraphs found")
            
            print(f"Found {len(subgraphs)} subgraphs")
            
            # Find DPU subgraph with all methods
            dpu_subgraph = None
            
            # Method 1: Device attribute
            for i, sg in enumerate(subgraphs):
                if sg is None:
                    continue
                try:
                    if sg.has_attr("device"):
                        device = sg.get_attr("device")
                        if isinstance(device, str) and device.upper() == "DPU":
                            dpu_subgraph = sg
                            print(f"Found DPU by device at index {i}")
                            break
                except:
                    continue
            
            # Method 2: Validation test
            if dpu_subgraph is None:
                for i, sg in enumerate(subgraphs):
                    if sg is None:
                        continue
                    try:
                        name = sg.get_name()
                        if name != "root":
                            test_runner = vart.Runner.create_runner(sg, "run")
                            if test_runner is not None:
                                dpu_subgraph = sg
                                print(f"Found DPU by validation at index {i}")
                                break
                    except:
                        continue
            
            # Method 3: Any available subgraph
            if dpu_subgraph is None:
                for sg in subgraphs:
                    if sg is not None:
                        try:
                            dpu_subgraph = sg
                            print("Using first available subgraph")
                            break
                        except:
                            continue
            
            if dpu_subgraph is None:
                raise ValueError("No DPU subgraph found")
            
            # Create runner
            self.dpu_runner = vart.Runner.create_runner(dpu_subgraph, "run")
            self.input_tensors = self.dpu_runner.get_input_tensors()
            self.output_tensors = self.dpu_runner.get_output_tensors()
            
            print("✅ DPU loaded successfully")
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            exit(1)
    
    def process_image_ultra_safe(self, image_path: str) -> tuple:
        """Ultra-safe image processing with atomic operations"""
        try:
            start_time = time.time()
            
            # Wait for file to be completely written
            max_wait = 5.0  # 5 seconds max wait
            wait_start = time.time()
            
            while time.time() - wait_start < max_wait:
                try:
                    # Try to read file
                    with open(image_path, 'rb') as f:
                        file_size = len(f.read())
                    
                    # Wait a bit and check if size changed (file still being written)
                    time.sleep(0.1)
                    
                    with open(image_path, 'rb') as f:
                        new_size = len(f.read())
                    
                    if file_size == new_size and file_size > 0:
                        # File is stable
                        break
                        
                except (IOError, OSError):
                    # File not ready yet
                    time.sleep(0.1)
                    continue
            
            # Read image with retry mechanism
            frame = None
            for attempt in range(3):
                try:
                    frame = cv2.imread(image_path)
                    if frame is not None and frame.size > 0:
                        break
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Read attempt {attempt + 1} failed: {e}")
                    time.sleep(0.1)
            
            if frame is None or frame.size == 0:
                return None, False, "Could not read image file"
            
            # Create completely isolated memory for DPU
            # Copy data to new memory region
            frame_copy = np.array(frame, copy=True)
            del frame  # Release original
            gc.collect()
            
            # Preprocess with proven method
            img_h, img_w = frame_copy.shape[:2]
            scale = min(416 / img_w, 416 / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            
            resized = cv2.resize(frame_copy, (new_w, new_h))
            
            # Create padded array in new memory
            padded = np.full((416, 416, 3), 114, dtype=np.uint8)
            pad_x = (416 - new_w) // 2
            pad_y = (416 - new_h) // 2
            padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
            
            # Release intermediate data
            del resized
            gc.collect()
            
            # Convert to int8 (proven method)
            input_data = (padded.astype(np.int16) - 128).astype(np.int8)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = np.ascontiguousarray(input_data)
            
            # Release padded data
            del padded
            gc.collect()
            
            # Prepare outputs with contiguous memory
            output_arrays = []
            for tensor in self.output_tensors:
                shape = tuple(tensor.dims)
                output_array = np.zeros(shape, dtype=np.float32)
                output_array = np.ascontiguousarray(output_array)
                output_arrays.append(output_array)
            
            # Final memory cleanup before DPU
            gc.collect()
            
            # DPU inference (proven method)
            job_id = self.dpu_runner.execute_async([input_data], output_arrays)
            self.dpu_runner.wait(job_id)
            
            # Simple detection count
            detection_count = 0
            for output in output_arrays:
                if len(output.shape) == 4 and output.shape[-1] >= 85:
                    objectness = output[0, :, :, 4]
                    detection_count += np.sum(objectness > 0.3)
            
            self.inference_time = time.time() - start_time
            self.detection_count = min(detection_count, 50)
            
            # Create result frame
            result_frame = frame_copy.copy()
            cv2.putText(result_frame, f"Detections: {self.detection_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Inference: {self.inference_time*1000:.1f}ms", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, "ULTRA-SAFE", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            return result_frame, True, "Success"
            
        except Exception as e:
            error_msg = f"Detection error: {str(e)[:50]}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            return None, False, error_msg

class UltraSafeWebApp:
    def __init__(self, detector, camera_id=0):
        self.detector = detector
        self.camera_id = camera_id
        
        # Flask app
        self.app = Flask(__name__)
        self.app.config['THREADED'] = False
        
        # Atomic file operations using double buffering
        self.temp_dir = tempfile.mkdtemp()
        self.frame_file_a = os.path.join(self.temp_dir, "frame_a.jpg")
        self.frame_file_b = os.path.join(self.temp_dir, "frame_b.jpg")
        self.current_file = os.path.join(self.temp_dir, "current.jpg")
        self.ready_file = os.path.join(self.temp_dir, "ready.jpg")
        
        self.file_toggle = False  # Toggle between A and B
        
        # Camera
        self.cap = None
        self.is_running = False
        self.capture_thread = None
        
        # Display frame
        self.current_display_frame = None
        self.frame_lock = threading.Lock()
        
        # Stats
        self.frame_count = 0
        self.start_time = time.time()
        
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>🛡️ Ultra-Safe Detection</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #1a1a2e;
            color: white;
            text-align: center;
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
        }
        img { 
            max-width: 100%; 
            border: 2px solid #16213e;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(22, 33, 62, 0.5);
        }
        .status {
            margin: 20px 0;
            font-size: 18px;
            color: #0f3460;
        }
        .safe-indicator {
            color: #e94560;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ Ultra-Safe DPU Detection</h1>
        <div class="status safe-indicator">
            Atomic File Operations + Memory Isolation
        </div>
        <img src="/video_feed" alt="Detection Stream" />
        <div class="status">
            Double-buffered + Premature JPEG protection
        </div>
    </div>
</body>
</html>
            ''')
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(self.generate_frames(),
                           mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def atomic_file_write(self, frame):
        """Atomic file write using double buffering"""
        try:
            # Choose buffer file
            target_file = self.frame_file_a if self.file_toggle else self.frame_file_b
            self.file_toggle = not self.file_toggle
            
            # Write to buffer file first
            success = cv2.imwrite(target_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            if success:
                # Atomic move to ready file
                shutil.move(target_file, self.ready_file)
                return True
            
            return False
            
        except Exception as e:
            print(f"Atomic write error: {e}")
            return False
    
    def camera_capture_thread(self):
        """Camera capture with atomic file operations"""
        print("Starting ultra-safe camera capture...")
        
        frame_counter = 0
        
        while self.is_running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        frame_counter += 1
                        
                        # Write frame atomically
                        if self.atomic_file_write(frame):
                            self.frame_count += 1
                        
                        # Reduce capture rate to prevent overwhelming
                        time.sleep(0.2)  # 5 FPS capture rate
                    else:
                        time.sleep(0.1)
                else:
                    time.sleep(0.5)
            except Exception as e:
                print(f"Camera capture error: {e}")
                time.sleep(1)
        
        print("Camera capture thread stopped")
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("Cannot open camera")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.is_running = True
        self.start_time = time.time()
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self.camera_capture_thread, daemon=True)
        self.capture_thread.start()
        
        print("Ultra-safe camera started")
        return True
    
    def generate_frames(self):
        """Generate frames with ultra-safe detection"""
        frame_counter = 0
        last_detection_time = 0
        
        while True:
            try:
                frame_counter += 1
                
                # Check for ready frame file
                if os.path.exists(self.ready_file):
                    try:
                        current_time = time.time()
                        
                        if frame_counter < 50:  # Extended initialization period
                            # First 50 frames: just display
                            current_frame = cv2.imread(self.ready_file)
                            if current_frame is not None:
                                display_frame = current_frame.copy()
                                cv2.putText(display_frame, f"Ultra-Safe Init: {frame_counter}/50", 
                                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                                cv2.putText(display_frame, "Atomic file operations active", 
                                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            else:
                                display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                                cv2.putText(display_frame, "File read error", 
                                           (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        elif current_time - last_detection_time > 3.0:  # Run detection every 3 seconds
                            print(f"Running ultra-safe detection on frame {frame_counter}")
                            
                            # Process using ultra-safe method
                            result_frame, success, message = self.detector.process_image_ultra_safe(self.ready_file)
                            
                            if success and result_frame is not None:
                                display_frame = result_frame
                                cv2.putText(display_frame, "ULTRA-SAFE SUCCESS", 
                                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                print("Ultra-safe detection successful!")
                            else:
                                current_frame = cv2.imread(self.ready_file)
                                if current_frame is not None:
                                    display_frame = current_frame.copy()
                                    cv2.putText(display_frame, "DETECTION FAILED", 
                                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                    cv2.putText(display_frame, message[:30], 
                                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                else:
                                    display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                                    cv2.putText(display_frame, "CRITICAL ERROR", 
                                               (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                print(f"Ultra-safe detection failed: {message}")
                            
                            last_detection_time = current_time
                        
                        else:
                            # Show current frame with countdown
                            current_frame = cv2.imread(self.ready_file)
                            if current_frame is not None:
                                display_frame = current_frame.copy()
                                next_detection = 3.0 - (current_time - last_detection_time)
                                cv2.putText(display_frame, f"Next: {next_detection:.1f}s", 
                                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                                cv2.putText(display_frame, "Ultra-safe mode", 
                                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                            else:
                                display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                                cv2.putText(display_frame, "No frame available", 
                                           (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    except Exception as e:
                        print(f"Frame processing error: {e}")
                        display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(display_frame, "PROCESSING ERROR", 
                                   (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                else:
                    # No ready file
                    display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(display_frame, "Waiting for camera...", 
                               (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                # Encode and yield
                _, buffer = cv2.imencode('.jpg', display_frame, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                time.sleep(0.2)  # 5 FPS display rate
                
            except Exception as e:
                print(f"Generate frames error: {e}")
                emergency_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(emergency_frame, "CRITICAL ERROR", 
                           (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                _, buffer = cv2.imencode('.jpg', emergency_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)
    
    def run(self, host='0.0.0.0', port=5000):
        if not self.start_camera():
            return
        
        print(f"🛡️ Ultra-safe web server: http://{host}:{port}")
        print("🔒 Atomic file operations + Memory isolation")
        
        try:
            self.app.run(host=host, port=port, debug=False, threaded=False)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.is_running = False
            if self.cap:
                self.cap.release()
            
            # Cleanup temp directory
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolox_nano_pt.xmodel')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    
    args = parser.parse_args()
    
    print("🛡️ Initializing ultra-safe detector...")
    detector = UltraSafeDetector(args.model)
    
    print("🌐 Starting ultra-safe web app...")
    app = UltraSafeWebApp(detector, args.camera)
    
    app.run(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
