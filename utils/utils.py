import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import matplotlib.pyplot as plt
import socket
import pickle
import requests
from pyzbar.pyzbar import decode, ZBarSymbol
import requests
import threading
from time import sleep
import time
from collections import deque

import h264decoder
import subprocess
frame = []
x = 0
y = 0
mode_status_data = { "mod": 2}
targetBox = [0,0,0,0]
lock_control = False
server_url = "http://192.168.31.60:7000"



class BaseEngine(object):
    def __init__(self, engine_path):
        self.CVData=deque(maxlen=20)
        self.safety=False
        self.mean = None
        self.std = None
        self.n_classes = 80
        self.class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})



    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        return data
    
    


    def detect_video(self, video_path, conf=0.5, end2end=False,autopilot=None):
        global mode_status_data,lock_control
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      
        print(width,height)
        
        fps = 0
        print("baglandi")
        boxes_array = np.zeros((4,4), dtype = int)
        center_x = width // 2
        center_y = height // 2
        draw_w = width//2
        draw_h = int(height*.8)
        right_text = (width - 140, height - 35)
        left_text = (10,height- 35)
        global index
        mainbox_x0 = center_x - (draw_w // 2)
        mainbox_y0 = center_y - (draw_h // 2)
        mainbox_x1 = center_x + (draw_w // 2)
        mainbox_y1 = center_y + (draw_h // 2)
        
        kamikaze = int(mode_status_data["mod"])
        
        while True:
            ret, frame = cap.read()
            kamikaze = int(mode_status_data["mod"])   
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  
            
            x = 0
            y = 0            
            timer_control = True                
                
            

            blob, ratio = preproc(frame, self.imgsz, self.mean, self.std)
            t1 = time.time()
            data = self.infer(blob)
            fps = (1. / (time.time() - t1))
            frame = cv2.putText(frame, "FPS:%d " %fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
            if end2end:
                num, final_boxes, final_scores, final_cls_inds = data
                final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
                dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
            else:
                predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
                dets = self.postprocess(predictions,ratio)

            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:,
                                                                :4], dets[:, 4], dets[:, 5]
                frame= vis(frame, final_boxes, final_scores, final_cls_inds,
                                conf=conf, class_names=self.class_names)        
                finalBoxes=[]
                #print(dets)
                for i in range(len(final_boxes)):
                    box = final_boxes[i]                
                    x0 = int(box[0])
                    y0 = int(box[1])
                    x1 = int(box[2])
                    y1 = int(box[3])
                    w = x1 - x0
                    h = y1 - y0
                    x = (w//2) + x0
                    y = (h//2) + y0

                    dist_center = int((abs(center_x - x)**2 + abs(center_y - y)**2) ** .5)
                    finalBoxes.append([x,y,w,h,dist_center])
                
                finalBoxes.sort(key= lambda a : a[4])
                
            if len(final_boxes) != 0:
                cv2.circle(frame,(x,y),1,(0,0,255),2)
                targetBox = finalBoxes[0]
                self.safety=False
                self.CVData.append({"bbox":targetBox,"time":time.time()})
                self.safety=True
                
                    
                cv2.line(frame,(center_x,center_y) , (targetBox[0],targetBox[1]), (255,0,0), 2)
                perc_width = (targetBox[2]*100) // width
                perc_height = (targetBox[3]*100) // height
                if perc_width > perc_height:
                    cv2.putText(frame, "Hedef Boyut: %" + str(perc_width), (10, 175),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 20, 20), 2)
                else:
                    cv2.putText(frame, "Hedef Boyut: %" + str(perc_height), (10, 175),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 20, 20), 2)
                if targetBox[0] > mainbox_x0 and targetBox[0] < mainbox_x1 and targetBox[1] > mainbox_y0 and targetBox[1] < mainbox_y1:
                    
                    if perc_width > 5 or perc_height > 5:
                        lock_control = True
                        cv2.putText(frame, "Kilitlenme Basladi", (center_x - 110, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (135, 255, 0), 2)                        
                        
                        timer_control = False
                        
                        cv2.putText(frame, str(int(time.time() - timerx)), (90, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 210), 2)                        
            else:
                
                self.safety=False
                self.CVData.append({"bbox":[0,0,0,0,0],"time":time.time()})
                self.safety=True
                cv2.line(frame,(center_x,center_y) , (center_x,center_y), (255,0,0), 2)
                cv2.putText(frame, "Taraniyor", (center_x - 80, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (135, 255, 0), 2)
                cv2.putText(frame, "Hedef Boyut: %" + str(0), (10, 175),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 20, 20), 2)
                lock_control = False
                
            
            
            cv2.rectangle(frame, (mainbox_x0, mainbox_y0), (mainbox_x1, mainbox_y1), (0, 255, 0), 2)
            
            cv2.line(frame,(center_x + 8,center_y - 8) , (center_x + 12,center_y - 12), (0, 0, 255), 1)
            cv2.line(frame,(center_x - 8,center_y + 8) , (center_x - 12,center_y + 12), (0, 0, 255), 1)
            cv2.line(frame,(center_x - 8,center_y - 8) , (center_x - 12,center_y - 12), (0, 0, 255), 1)
            cv2.line(frame,(center_x + 8,center_y + 8) , (center_x + 12,center_y + 12), (0, 0, 255), 1)
            cv2.circle(frame,(center_x,center_y),2,(0,0,255),2)
            cv2.putText(frame, "ABRA UAV", right_text,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (135, 255, 0), 2)
            cv2.putText(frame, "Timer:", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 20, 20), 2)
            if timer_control:
                cv2.putText(frame, str(0), (90, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 210), 2)
                timerx = time.time()
                
            cv2.putText(frame, "X: "+str(x), (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (89, 137, 0), 2)
            cv2.putText(frame, "Gorev: Otonom Takip ve Kilitlenme", left_text,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (89, 137, 0), 2)
            cv2.putText(frame, "Y: "+str(y), (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (89, 137, 0), 2)
            
                        
                
                    
                        
                
            
            cv2.imshow('frame', frame) 
            
        
        #out.release()
        # cap.release()

    def inference(self, img, conf=0.5, end2end=False):
        origin_img = img
        img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std)
        data = self.infer(img)
        if end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        else:
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            dets = self.postprocess(predictions,ratio)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,
                                                             :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=conf, class_names=self.class_names)
            return final_boxes
        return None

    @staticmethod
    def postprocess(predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        return dets

    def get_fps(self):
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(5):  # warmup
            _ = self.infer(img)

        t0 = time.perf_counter()
        for _ in range(100):  # calculate average time
            _ = self.infer(img)
        print(100/(time.perf_counter() - t0), 'FPS')


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # if use yolox set
    # padded_img = padded_img[:, :, ::-1]
    # padded_img /= 255.0
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def rainbow_fill(size=50):  # simpler way to generate rainbow color
    cmap = plt.get_cmap('jet')
    color_list = []

    for n in range(size):
        color = cmap(n/size)
        color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]

    return np.array(color_list)


_COLORS = rainbow_fill(80).astype(np.float32).reshape(-1, 3)


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format("iha", score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
        
        
        
    return img