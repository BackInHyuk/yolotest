#!/usr/bin/env python3
"""
MJPEG streaming server for YOLOv8
"""

from flask import Flask, Response
import cv2
import threading
import time

app = Flask(__name__)
output_frame = None
lock = threading.Lock()

def capture_frames():
    global output_frame, lock
    
    # Initialize YOLOv8 model here
    # ... (model initialization code)
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Run YOLOv8 inference
        # ... (inference code)
        
        # Update global frame
        with lock:
            output_frame = frame.copy()

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
                
        # Return as MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.033)  # 30 FPS

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
        <head>
            <title>YOLOv8 Stream</title>
        </head>
        <body>
            <h1>YOLOv8 Live Detection</h1>
            <img src="/video_feed" width="640" height="480">
        </body>
    </html>
    '''

if __name__ == '__main__':
    t = threading.Thread(target=capture_frames)
    t.daemon = True
    t.start()
    
    # Run server
    app.run(host='0.0.0.0', port=5000, threaded=True)
