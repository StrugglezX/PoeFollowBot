import os, sys, time, datetime, random

from PIL import Image
import cv2

# load weights and set defaults
img_size=800

def make_square(im, min_size=img_size, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im
    
videopath = './poevidlowres.mp4'
vid = cv2.VideoCapture(videopath)
ret,frame=vid.read()

objnum = 0
while(True):
    ret, frame = vid.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    pilimg = make_square(pilimg)
    detections = detect_image(pilimg)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    img = np.array(pilimg)

    im = Image.fromarray(img)
    im = make_square(im)
    
    #Hacks for making square inputs from the video
    if objnum % 40 == 0:
        
        image_name = 'forboxing/zomebie{}.jpg'.format(objnum)
        im.save(image_name, 'JPEG')
    objnum = objnum + 1
    
cv2.destroyAllWindows()
