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
except ImportError as e:
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
                    print(f"Found DPU subgraph: {sg.dpu_kernel.name}") # Use dpu_kernel.name for DPU subgraph
                    break
            
            if not self.dpu_subgraph:
                raise Exception("No DPU subgraph found in the model. Check .xmodel with 'xir svg'.")
            
            # --- Revert to original runner creation method if it worked before ---
            # Using "run" is common for actual DPU execution.
            self.runner = vart.Runner.create_runner(self.dpu_subgraph, "run")
            print("DPU Runner created successfully with 'run' mode.")

            # --- Safely get input/output tensor information ---
            # Use subgraph.get_input_tensors() and subgraph.get_output_tensors()
            # This is more robust as it's directly from the graph/subgraph
            # rather than assuming the runner has all tensor properties after creation.
            
            subgraph_input_tensors = self.dpu_subgraph.get_input_tensors()
            subgraph_output_tensors = self.dpu_subgraph.get_output_tensors()

            if not subgraph_input_tensors or not subgraph_output_tensors:
                raise Exception("Could not retrieve input/output tensors from DPU subgraph.")
            
            self.input_tensor_info = subgraph_input_tensors[0]
            self.output_tensor_info = subgraph_output_tensors[0]

            self.input_dims = self.input_tensor_info.dims
            self.input_height = self.input_dims[1]
            self.input_width = self.input_dims[2]
            
            self.input_dtype = self.input_tensor_info.dtype
            self.input_fix_point = self.input_tensor_info.get_attr("fix_point") if self.input_tensor_info.has_attr("fix_point") else 0
            # Scale for input: Data from opencv (0-255 uint8) to DPU INT8.
            # Usually: (2**fix_point) / 255.0 OR just 2**fix_point if DPU expects 0-255 scaled.
            # This requires knowing your quantization method. Let's assume (2**fix_point) for now.
            # If the quantization normalized to [0,1] THEN scaled, it would be 2**fix_point
            # If it directly quantized 0-255, it might be 1.0. This is crucial.
            # For typical DPU INT8, input_scale is 1.0 or 255.0 / (2**fix_point) etc.
            # For now, let's assume it scales 0-255 into the INT8 range directly
            # A common pattern for images into INT8 DPU:
            #   input_data = (image * (2**fix_point) / 255.0).astype(np.int8)
            # This means input_scale for `(rgb_image * self.input_scale)` would be `(2**self.input_fix_point) / 255.0`
            # However, sometimes, the model expects raw uint8 (0-255) to be fed directly to runner
            # and runner handles internal quantization. Let's try simple scale first.
            self.input_scale = 1.0 # This might need adjustment. Start simple.
            
            print(f"Input Tensor: Name={self.input_tensor_info.name}, Shape={self.input_dims}, DType={self.input_dtype.name}, FixPoint={self.input_fix_point}, Scale={self.input_scale}")

            self.output_dims = self.output_tensor_info.dims
            self.output_dtype = self.output_tensor_info.dtype
            self.output_fix_point = self.output_tensor_info.get_attr("fix_point") if self.output_tensor_info.has_attr("fix_point") else 0
            # Scale for output: DPU INT8 to float32.
            # This is 1.0 / (2**fix_point)
            self.output_scale = 1.0 / (2**self.output_fix_point) if self.output_dtype.name == 'INT8' else 1.0
            
            print(f"Output Tensor: Name={self.output_tensor_info.name}, Shape={self.output_dims}, DType={self.output_dtype.name}, FixPoint={self.output_fix_point}, Scale={self.output_scale}")

            self.ready = True
            self.use_fallback = False
            
        except Exception as e:
            print(f"Initialization error: {e}")
            self.ready = False
            self.use_fallback = True
            print("DPU initialization failed, falling back to dummy detection.")

    def detect_safe(self, image):
        if not self.ready or self.use_fallback:
            return self.fallback_detect(image)
        
        try:
            resized_image = cv2.resize(image, (self.input_width, self.input_height))
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
            # Prepare input data based on expected DPU type
            if self.input_dtype.name == 'INT8':
                # If input_scale is 1.0, means uint8 is cast to int8, no division.
                # If your quantization script did (image / 255.0 * 2^fix_point), then input_scale should be (2^fix_point)/255.0
                input_data = (rgb_image * self.input_scale).astype(np.int8)
                # It's also possible that `vart` expects `uint8` and does the internal casting/quantization itself.
                # Let's try to pass `uint8` directly first if `self.input_fix_point` is 0 (meaning no explicit scaling needed)
                # Or if the input_dtype is actually DT_UINT8 (which is rare for DPU)
                if self.input_fix_point == 0: # Simple casting
                     input_data = rgb_image.astype(np.int8)
                else: # Apply specific scaling from quantization if any
                     input_data = (rgb_image * (2**self.input_fix_point) / 255.0).astype(np.int8)
                     # Or try: input_data = rgb_image.astype(np.float32) * self.input_scale if input_scale set to `(2**fix_point)/255.0`
                     # This depends on your specific quantization flow.
                     # A safer bet: The raw uint8 image usually is given to runner, and runner scales based on fix_point.
                     # Let's revert to a common pattern: feed uint8 and let runtime handle it, OR manually scale to fixed point.
                     # For now, let's stick with the original `uint8` and let the runner complain if it's wrong,
                     # or try (image * QUANT_SCALE).astype(INT8)
                     # Let's assume it needs to be scaled and cast
                     input_data = (rgb_image.astype(np.float32) / 255.0 * (2**self.input_fix_point)).astype(np.int8)

            elif self.input_dtype.name == 'FLOAT32':
                input_data = rgb_image.astype(np.float32) / 255.0
            else:
                raise ValueError(f"Unsupported input data type: {self.input_dtype.name}")

            input_data = np.expand_dims(input_data, axis=0) # Add batch dimension

            # Prepare output buffer in DPU's native data type
            output_buffer = np.zeros(self.output_dims, dtype=self.output_dtype.as_numpy_dtype)
            
            # Execute inference
            job_id = self.runner.execute_async([input_data], [output_buffer])
            self.runner.wait(job_id)
            
            # Post-process output: de-quantize if necessary
            if self.output_dtype.name == 'INT8':
                outputs = [output_buffer[0] * self.output_scale] # Apply de-quantization scale
            elif self.output_dtype.name == 'FLOAT32':
                outputs = [output_buffer[0]]
            else:
                raise ValueError(f"Unsupported output data type: {self.output_dtype.name}")
            
            detections = self.process_outputs(outputs, image.shape)
            result = self.draw_detections(image.copy(), detections)
            
            return result, detections
        
        except Exception as e:
            print(f"Inference error during DPU run: {e}")
            self.use_fallback = True
            return self.fallback_detect(image)

    # ... (rest of the class methods: fallback_detect, process_outputs, draw_detections,
    # and all functions outside the class: find_camera, camera_thread, generate, index, video_feed, main)
    # remain the same as the previous full code, except for `process_outputs` if it was already updated with NMS.
    # I'll just include the `process_outputs` with NMS and some minor adjustments to class_id/score logic.
    def process_outputs(self, outputs_raw, img_shape):
        detections = []
        try:
            if not outputs_raw or len(outputs_raw) == 0:
                return detections
            
            output = outputs_raw[0] 
            
            if output.ndim == 2:
                predictions = output
            elif output.ndim == 3: 
                # Expected YOLOv8 output from DPU after de-quantization could be (1, 84, 8400)
                # or (1, 8400, 84)
                if output.shape[1] == 84 and output.shape[2] == 8400: # (Batch, 84, Num_Boxes)
                    predictions = output[0].transpose(1, 0) # Transpose to (Num_Boxes, 84)
                elif output.shape[1] == 8400 and output.shape[2] == 84: # (Batch, Num_Boxes, 84)
                    predictions = output[0] # Take the first batch
                else:
                    print(f"Unexpected output shape: {output.shape}. Trying a common transpose.")
                    # Fallback to try a transpose if dimensions are swapped but not 84 vs 8400 directly
                    if output.shape[0] == 1 and output.shape[1] > output.shape[2]: # (1, Big, Small)
                        predictions = output[0].transpose(1,0) # Attempt transpose
                    elif output.shape[0] == 1 and output.shape[2] > output.shape[1]: # (1, Small, Big)
                        predictions = output[0] # No transpose needed
                    else:
                        return detections
            else:
                print(f"Unexpected output dimensions: {output.ndim}")
                return detections
            
            predictions = predictions.astype(np.float32)

            boxes = []
            scores = []
            class_ids = []

            for i in range(predictions.shape[0]): 
                row = predictions[i]
                
                # YOLOv8 output format: [cx, cy, w, h, class_score_0, ..., class_score_79] (84 elements)
                if len(row) >= 84:
                    cx, cy, w, h = row[0:4]
                    class_scores = row[4:84] 
                    
                    class_id = np.argmax(class_scores)
                    confidence = class_scores[class_id]
                    
                    if confidence > 0.4: # Adjustable confidence threshold
                        # Scale to image
                        scale_x = img_shape[1] / self.input_width
                        scale_y = img_shape[0] / self.input_height
                        
                        x1 = int((cx - w/2) * scale_x)
                        y1 = int((cy - h/2) * scale_y)
                        x2 = int((cx + w/2) * scale_x)
                        y2 = int((cy + h/2) * scale_y)
                        
                        # Clip coordinates to image boundaries
                        x1 = max(0, min(x1, img_shape[1]))
                        y1 = max(0, min(y1, img_shape[0]))
                        x2 = max(0, min(x2, img_shape[1]))
                        y2 = max(0, min(y2, img_shape[0]))
                        
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            scores.append(float(confidence))
                            class_ids.append(int(class_id))
            
            # Apply Non-Maximum Suppression (NMS)
            if len(boxes) > 0:
                boxes_np = np.array(boxes)
                scores_np = np.array(scores)
                class_ids_np = np.array(class_ids)

                final_detections = []
                for class_id_val in np.unique(class_ids_np):
                    indices = np.where(class_ids_np == class_id_val)[0]
                    selected_boxes = boxes_np[indices]
                    selected_scores = scores_np[indices]

                    # OpenCV NMSBoxes expects [x, y, width, height]
                    boxes_xywh = np.array([[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in selected_boxes])
                    
                    indices_keep = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), selected_scores.tolist(), score_threshold=0.4, nms_threshold=0.45) # Adjust thresholds as needed
                    
                    if len(indices_keep) > 0:
                        for idx in indices_keep.flatten():
                            original_idx = indices[idx] # Get original index
                            final_detections.append({
                                'bbox': boxes[original_idx],
                                'score': scores[original_idx],
                                'class': class_ids[original_idx]
                            })
                detections = final_detections

        except Exception as e:
            print(f"Output processing error: {e}")
            
        return detections
