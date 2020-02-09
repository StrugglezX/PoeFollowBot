# Find the best attack object

from PIL import Image, ImageGrab
from models import *
from utils import *
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

from PIL import Image

# load weights and set defaults
config_path='config/yolov3-tiny.cfg'
#weights_path='config/yolov3.weights'
weights_path='config/48.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# load model and put into eval mode
if os.path.exists(weights_path):
    print('available!')
    model = Darknet(config_path, img_size=img_size)
    model.load_weights(weights_path)
    model.cuda()
    model.eval()

classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor

mot_tracker = Sort() 

def get_current_time_ms():
    return int(round(time.time() * 1000))
    
def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 1, conf_thres, nms_thres)
    return detections[0]
        
def PoeNeuronObjectDetectionThread(data):
    if not os.path.exists(weights_path):
        return
        
    while True:
        
        image = PoeNeuronScreenshotThread.get_next_screenshot(data)
        image = image.resize((img_size,img_size))
        img = np.array(image)
        t1 = get_current_time_ms()
        detections = detect_image(image)
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
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    