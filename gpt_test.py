#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOX-Nano + MJPEG HTTP 스트리밍 (Flask)
보드 IP: 192.168.x.x 라면 ⇒  http://192.168.x.x:5000  접속
"""

import os, cv2, time, random, colorsys, numpy as np, threading
from flask import Flask, Response, render_template_string
import xir, vitis_ai_library

# ────────────────────── 환경 설정 ──────────────────────
XMODEL   = "yolox_nano_pt.xmodel"
LABELS   = "img/coco2017_classes.txt"
IN_SIZE  = (416, 416)
CLASS_TH = 0.30
NMS_TH   = 0.45
NMS_SCTH = 0.10
CAM_ID   = 0           # USB 웹캠 번호
FPS_CAP  = 30          # Webcam 프레임 제한(없으면 0)

# ────────────────────── 클래스 이름 로드 ──────────────────────
with open(LABELS) as f:
    CLASSES = [l.strip() for l in f]

# ────────────────────── 전/후처리 함수 (필요 부분만) ──────────────────────
def preprocess(img, size):
    pad = np.ones((size[0], size[1], 3), dtype=np.uint8) * 114
    r = min(size[0]/img.shape[0], size[1]/img.shape[1])
    rs = cv2.resize(img, (int(img.shape[1]*r), int(img.shape[0]*r)),
                    interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    pad[:rs.shape[0], :rs.shape[1]] = rs
    return np.ascontiguousarray(pad, dtype=np.float32), r

def sigmoid(x): return 1/(1+np.exp(-x))
def softmax(x): e=np.exp(x - np.max(x)); return e/e.sum(axis=-1, keepdims=True)

def nms(boxes, scores, thr):
    x1, y1, x2, y2 = boxes.T
    area = (x2-x1+1)*(y2-y1+1)
    order = scores.argsort()[::-1]
    keep=[]
    while order.size:
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        w=np.maximum(0., xx2-xx1+1); h=np.maximum(0., yy2-yy1+1)
        inter=w*h; ovr=inter/(area[i]+area[order[1:]]-inter)
        inds=np.where(ovr<=thr)[0]; order=order[inds+1]
    return keep

def postprocess(outputs, img_size, r, w, h):
    p6=False; strides=[8,16,32] if not p6 else [8,16,32,64]
    hs=[img_size[0]//s for s in strides]; ws=[img_size[1]//s for s in strides]
    grids=[]; exps=[]
    for hsize,wsize,s in zip(hs,ws,strides):
        yv,xv=np.meshgrid(np.arange(hsize),np.arange(wsize))
        grid=np.stack((xv,yv),2).reshape(1,-1,2); grids.append(grid)
        shape=grid.shape[:2]; exps.append(np.full((*shape,1), s))
    grids=np.concatenate(grids,1); exps=np.concatenate(exps,1)
    outputs[...,:2]=(outputs[...,:2]+grids)*exps
    outputs[...,2:4]=np.exp(outputs[...,2:4])*exps
    pred=outputs[0]; boxes=pred[:,:4]
    scores=sigmoid(pred[:,4:5])*softmax(pred[:,5:])
    xyxy=np.empty_like(boxes)
    xyxy[:,0]=boxes[:,0]-boxes[:,2]/2; xyxy[:,1]=boxes[:,1]-boxes[:,3]/2
    xyxy[:,2]=boxes[:,0]+boxes[:,2]/2; xyxy[:,3]=boxes[:,1]+boxes[:,3]/2
    xyxy/=r
    cls_ids=scores.argmax(1); cls_scores=scores[np.arange(len(cls_ids)),cls_ids]
    mask=cls_scores>CLASS_TH
    if not mask.any(): return np.empty((0,6))
    xyxy,cls_scores,cls_ids = xyxy[mask],cls_scores[mask],cls_ids[mask]
    keep=nms(xyxy,cls_scores,NMS_TH)
    dets=np.concatenate([xyxy[keep], cls_scores[keep,None], cls_ids[keep,None]],1)
    dets[:,0]=np.clip(dets[:,0],0,w); dets[:,1]=np.clip(dets[:,1],0,h)
    dets[:,2]=np.clip(dets[:,2],0,w); dets[:,3]=np.clip(dets[:,3],0,h)
    return dets

def draw_bbox(img, dets):
    if not len(dets): return img
    n_cls=len(CLASSES)
    hsv=[(x/n_cls,1,1) for x in range(n_cls)]
    rgb=[tuple(int(c*255) for c in colorsys.hsv_to_rgb(*h)) for h in hsv]
    random.seed(0); random.shuffle(rgb)
    h,w,_=img.shape; thick=max(1,int(1.2*(h+w)/600))
    for x1,y1,x2,y2,sc,ci in dets:
        color=rgb[int(ci)]; cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),color,thick)
        label=f"{CLASSES[int(ci)]}:{sc:.2f}"
        t_size=cv2.getTextSize(label,0,0.5,thick//2)[0]
        cv2.rectangle(img,(int(x1),int(y1)-t_size[1]-4),(int(x1)+t_size[0],int(y1)),color,-1)
        cv2.putText(img,label,(int(x1),int(y1)-2),0,0.5,(0,0,0),thick//2,cv2.LINE_AA)
    return img

# ────────────────────── DPU 러너 1회 초기화 ──────────────────────
graph   = xir.Graph.deserialize(XMODEL)
runner  = vitis_ai_library.GraphRunner.create_graph_runner(graph)
in_t    = runner.get_input_tensors()[0]
out_ts  = runner.get_output_tensors()
in_shape=tuple(in_t.dims)
inp_buf = [np.empty(in_shape, dtype=np.float32, order='C')]
out_buf = [np.empty(tuple(t.dims), dtype=np.float32, order='C') for t in out_ts]

# ────────────────────── 글로벌 프레임 (스트리밍 공유) ──────────────────────
latest_jpeg = None
lock = threading.Lock()

# ────────────────────── 추론 스레드 ──────────────────────
def infer_loop():
    global latest_jpeg
    cap=cv2.VideoCapture(CAM_ID)
    if FPS_CAP: cap.set(cv2.CAP_PROP_FPS, FPS_CAP)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        h,w=frame.shape[:2]
        blob,r = preprocess(frame, IN_SIZE)
        inp_buf[0][0,...]=blob.transpose(2,0,1)  # NCHW
        jid=runner.execute_async(inp_buf, out_buf); runner.wait(jid)
        outs=np.concatenate([o.reshape(1,-1,o.shape[-1]) for o in out_buf],1)
        dets=postprocess(outs, IN_SIZE, r, w, h)
        vis=draw_bbox(frame.copy(), dets)
        _, jpeg=cv2.imencode('.jpg', vis, [int(cv2.IMWRITE_JPEG_QUALITY),80])
        with lock:
            latest_jpeg=jpeg.tobytes()
    cap.release()

threading.Thread(target=infer_loop, daemon=True).start()

# ────────────────────── Flask 서버 ──────────────────────
app = Flask(__name__)

HTML = """
<!doctype html><title>YOLOX Stream</title>
<h2 style="text-align:center;">YOLOX-Nano DPU 실시간 스트리밍</h2>
<img src="/video_feed" style="display:block;margin:auto;max-width:100%;">
"""

@app.route('/')
def index(): return render_template_string(HTML)

def mjpeg_generator():
    while True:
        with lock:
            frame = latest_jpeg
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)   # 100 FPS 이상 보내지 않도록

@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ────────────────────── 메인 ──────────────────────
if __name__ == '__main__':
    # 0.0.0.0 로 열어야 외부(호스트 PC)에서 접속 가능
    app.run(host='0.0.0.0', port=5000, threaded=True)
