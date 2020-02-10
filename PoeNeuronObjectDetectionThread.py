# Find the best attack object

from PIL import Image, ImageGrab
from models import *
from utils import *
from utils.datasets import *
import cv2
from sort import *
import copy

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import scipy.misc
import PoeNeuronScreenshotThread
from PoeNeuronData import ObjectDetection
from time import sleep
import time
import numpy

from PIL import Image

img_size=800
conf_thres=0.3
iou_thres=0.5
weights_path='weights/best.pt'
names_path='config/coco.names'
config_path='config/yolov3.cfg'
classify = False

def get_current_time_ms():
    return int(round(time.time() * 1000))
    
screenshot_count = 0

def prepare_image(img0):
    # Padded resize
#    img = img0 #cv2.cvtColor(numpy.array(img0), cv2.COLOR_RGB2BGR)
    img = numpy.array(img0) 
    # Convert RGB to BGR 
    #img = img[:, :, ::-1].copy() 
    
    img = letterbox(img, new_shape=img_size)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    half = False
    img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)  # uint8 to fp16/fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img
    
def detect_image(im0s, device, model):

    names = load_classes(names_path)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    t1a = get_current_time_ms()
    img = prepare_image(im0s)
    t = time.time()
    t1b = get_current_time_ms()

    # Get detections
    img = torch.from_numpy(img).to(device)
    t1c = get_current_time_ms()
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
        t1cd = get_current_time_ms()
        
    pred = model(img)[0]
    t1d = get_current_time_ms()

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
    t1e = get_current_time_ms()
    
    # Apply Classifier
    if classify:
        pred = apply_classifier(pred, modelc, img, im0s)

    # Process detections
    path = 'G:/PoeFollowBot/data/PathOfExileMonsters/images/0/zombie0.jpg'
    out = 'G:/PoeFollowBot/output'
    for i, det in enumerate(pred):  # detections per image
        t1f = get_current_time_ms()
        p, s, im0 = path, '', im0s

        save_path = str(Path(out) / Path(p).name)
        s += '%gx%g ' % img.shape[2:]  # print string
        t1g = get_current_time_ms()
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in det:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                if save_img or view_img:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        # Print time (inference + NMS)
        print('%sDone. (%.3fs)' % (s, time.time() - t))
        
        """
        cv2.imshow('detection (q exit):', im0)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration
        """
    print('a-b{}  b-c{} c-cd{} cd-d{} d-e{} e-f{} f-g{}'.format(t1b - t1a, t1c - t1b, t1cd - t1c, t1d - t1cd, t1e - t1d, t1f - t1e, t1g - t1f))
    return None
        
def PoeNeuronObjectDetectionThread(data):
    if not os.path.exists(weights_path):
        return
        
    torch.no_grad()
    model = Darknet(config_path, img_size)
    attempt_download(weights_path)
    device = torch_utils.select_device('')
    if weights_path.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights_path)
        
    # Second-stage classifier
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()
    model.to(device).eval()
            
    # Half precision
    half = False
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
        
    while True:
        
        image = ImageGrab.grab()
        image = image.resize((img_size,img_size))
        img = np.array(image)
        t1 = get_current_time_ms()
        detections = detect_image(image, device, model)
        t2 = get_current_time_ms()
        time_taken = t2 - t1
        print('fps: {}'.format(1000.0 / time_taken))
        
        
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x
        if detections is not None:
            tracked_objects = mot_tracker.update(detections.cpu())
            
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                box_h_half = int(((y2 - y1) / unpad_h) * img.shape[0]) / 2
                box_w_half = int(((x2 - x1) / unpad_w) * img.shape[1]) / 2
                y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                cls = classes[int(cls_pred)]
                print('detected {} at {}x{}'.format(cls, x1, y1))
                if cls not in data._detected_objects:
                    data._detected_objects[cls] = {}
                new_obj = ObjectDetection(x1 + box_w_half, y1 + box_h_half, get_current_time_ms())
                data._detected_objects[cls][obj_id] = new_obj 
                
        """
        copy_dets = copy.deepcopy(data._detected_objects)
        for cls in copy_dets:
            for obj_id in copy_dets[cls]:
                object_coordinate = copy_dets[cls][obj_id]
                if object_coordinate == None:
                    continue
                detected_time = object_coordinate._time
                if get_current_time_ms() - detected_time > 3000:
                    data._detected_objects[cls][obj_id] = None
        """
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    