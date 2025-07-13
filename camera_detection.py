#!/usr/bin/env python3
"""
Complete YOLOv8n object detection with DPU
"""

import cv2
import numpy as np
import xir
import vart
import time

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
        print("Loading model...")
        self.graph = xir.Graph.deserialize(model_path)
        subgraphs = self.graph.get_root_subgraph().toposort_child_subgraph()
        
        # Find DPU subgraph
        self.dpu_subgraph = None
        for sg in subgraphs:
            if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
                self.dpu_subgraph = sg
                break
                
        self.runner = vart.Runner.create_runner(self.dpu_subgraph, "run")
        self.input_tensors = self.runner.get_input_tensors()
        self.output_tensors = self.runner.get_output_tensors()
        
        # Get input shape
        self.input_shape = tuple(self.input_tensors[0].dims)
        print(f"Input shape: {self.input_shape}")
        
    def preprocess(self, image):
        """Preprocess image for YOLOv8"""
        # Resize to model input size
        resized = cv2.resize(image, (self.input_shape[2], self.input_shape[1]))
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Add batch dimension
        return np.expand_dims(rgb, axis=0).astype(np.uint8)
    
    def run_inference(self, image):
        """Run DPU inference"""
        # Preprocess
        input_data = self.preprocess(image)
        
        # Create output arrays
        outputs = []
        for tensor in self.output_tensors:
            shape = tuple(tensor.dims)
            output = np.empty(shape, dtype=np.float32)
            outputs.append(output)
        
        # Run DPU
        job_id = self.runner.execute_async(input_data, outputs)
        self.runner.wait(job_id)
        
        return outputs
    
    def postprocess(self, outputs, img_shape, conf_threshold=0.5):
        """Process YOLOv8 outputs to get detections"""
        detections = []
        
        # YOLOv8 output format varies, but typically:
        # outputs[0] shape: [1, 84, 8400] or [1, 8400, 84]
        output = outputs[0]
        
        if output.shape[-1] == 84:  # [1, N, 84]
            predictions = output[0]
        else:  # [1, 84, N]
            predictions = output[0].T
            
        # Process each detection
        for pred in predictions:
            # First 4 values are box coordinates
            x_center, y_center, width, height = pred[:4]
            
            # Next 80 values are class scores
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence > conf_threshold:
                # Convert to image coordinates
                h, w = img_shape[:2]
                x1 = int((x_center - width/2) * w / self.input_shape[2])
                y1 = int((y_center - height/2) * h / self.input_shape[1])
                x2 = int((x_center + width/2) * w / self.input_shape[2])
                y2 = int((y_center + height/2) * h / self.input_shape[1])
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'score': float(confidence),
                    'class': int(class_id)
                })
                
        return detections
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes on image"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            score = det['score']
            class_id = det['class']
            
            # Draw box
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{COCO_CLASSES[class_id]}: {score:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1-20), (x1+label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            
        return image

def main():
    # Initialize model
    model = YOLOv8DPU("yolov8n_kv260.xmodel")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nProcessing... Press Ctrl+C to stop")
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference
            start_time = time.time()
            outputs = model.run_inference(frame)
            detections = model.postprocess(outputs, frame.shape)
            inference_time = (time.time() - start_time) * 1000
            
            # Draw results
            result_frame = model.draw_detections(frame.copy(), detections)
            
            # Add info text
            info_text = f"Frame: {frame_count} | Time: {inference_time:.1f}ms | Objects: {len(detections)}"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save every 10th frame
            if frame_count % 10 == 0:
                filename = f"detection_{frame_count:04d}.jpg"
                cv2.imwrite(filename, result_frame)
                print(f"Saved {filename} - {len(detections)} objects detected")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nStopped by user")
        
    cap.release()
    print(f"\nProcessed {frame_count} frames")

if __name__ == "__main__":
    main()
