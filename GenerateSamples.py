import os, sys, time, datetime, random
from PIL import Image
import numpy as np
import cv2

# load weights and set defaults
img_size=800
    
videopath = './poevidlowres.mp4'
vid = cv2.VideoCapture(videopath)
ret,frame=vid.read()

objnum = 0
while(True):
    ret, frame = vid.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (800, 800), interpolation = cv2.INTER_AREA)
    frame = Image.fromarray(frame)
    
    if objnum % 40 == 0:
        
        image_name = 'data/PathOfExileMonsters/images/zomebie{}.jpg'.format(objnum)
        frame.save(image_name, 'JPEG')
    objnum = objnum + 1
    
cv2.destroyAllWindows()
