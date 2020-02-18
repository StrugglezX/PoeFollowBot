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
    
def detect_image(data, im0s, device, model):
    detections = []
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
        im0 = numpy.array(im0s) 

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
                
            for *xyxy, conf, cls in det:
                x1 = int(xyxy[0])
                x2 = int(xyxy[2])
                
                y1 = int(xyxy[1])
                y2 = int(xyxy[3])
                
                av_x = (x1 + x2) / 2.0
                av_y = (y1 + y2) / 2.0
                det_name = names[int(c)]
                
                dist_center_x = abs(av_x - (img_size / 2))
                y_character_offset_from_center = -50
                dist_center_y = abs(av_y - ((img_size / 2) + y_character_offset_from_center))
                print('conf {} dist_center_x {} dist_center_y {}'.format(conf, dist_center_x, dist_center_y))
                if dist_center_x < 50 and dist_center_y < 50:
                    #deadzone
                    pass
                elif conf > 0.60:
                    detections.append( (av_x, av_y, det_name) )
                
    return detections
        
        
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
        
        image0 = ImageGrab.grab()
        image1 = image0.resize((img_size,img_size))
        img = np.array(image1)
        t1 = get_current_time_ms()
        detections = detect_image(data, image1, device, model)
        t2 = get_current_time_ms()
        time_taken = t2 - t1        
        
        for det in detections:
            converted_x = (det[0] / img_size) * image0.width
            converted_y = (det[1] / img_size) * image0.height
            det_name = det[2]
            print('detected {} at {}x{}'.format(det_name, converted_x, converted_y))
            if det_name not in data._detected_objects:
                data._detected_objects[det_name] = []
            data._detected_objects[det_name].append(ObjectDetection(converted_x, converted_y, get_current_time_ms()))
        
        for cls in list(data._detected_objects.keys()):
            for object_coordinate in list(data._detected_objects[cls]):
                if get_current_time_ms() - object_coordinate._time > 1000:
                    data._detected_objects[cls].remove(object_coordinate)
        
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    