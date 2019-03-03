# Responsible for detecting the thing to follow

import cv2
from mss import mss
from PIL import Image
import numpy as np
import os
import time

    
workspace_base='C:\\Users\\milbu\\Desktop\\PoeNeuron'
templates_base=os.path.join(workspace_base, 'templates')
ally_template=cv2.imread(os.path.join(templates_base, 'AllyX.png'), 1)
blue_ally_template=cv2.imread(os.path.join(templates_base, 'BlueAllyX.png'), 1)
player_template=cv2.imread(os.path.join(templates_base, 'PlayerX.png'), 1)
exit_template=cv2.imread(os.path.join(templates_base, 'Exit.png'), 1)
door_template=cv2.imread(os.path.join(templates_base, 'PoeDoor.png'), 1)
portal_template=cv2.imread(os.path.join(templates_base, 'Portal.png'), 1)


screen = mss()
monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 984}

def take_screenshot():
    sct_img = screen.grab(monitor)
    img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
    img = np.array(img)
    img = convert_rgb_to_bgr(img)

    return img

def get_image(path):
    return cv2.imread(path, 1)

def convert_rgb_to_bgr(img):
    return img[:, :, ::-1]

def match_template_to_position(img_grayscale, template, threshold=0.7):
    res = cv2.matchTemplate(img_grayscale, template, cv2.TM_CCOEFF_NORMED)
    matches = np.where(res >= threshold)
    position = None
    for pt in zip(*matches[::-1]):
        position = pt
        break
    return position

def load_example_file():
    return cv2.imread("C:\\Users\\milbu\\Desktop\\PoeNeuron\\templates\\Examples.png", 1)
    
def find_template_position(template, screenshot):
    pos = match_template_to_position(screenshot, template)
    return pos
    
def find_portal_position():
    return None
    
def PoeNeuronFollowPlayerThread(data):
    while True:
        data._text_command='follow'
        screenshot = take_screenshot()
        #screenshot = load_example_file()
        data._player_location = find_template_position(player_template, screenshot)
        if data._text_command == "follow":
            data._move_location_command = find_template_position(ally_template, screenshot)
            if data._move_location_command == None:
                data._move_location_command = find_template_position(blue_ally_template, screenshot)
        elif data._text_command == "portal":
            data._move_location_command = find_template_position(portal_template, screenshot)
        elif data._text_command == "door":
            data._move_location_command = find_template_position(door_template, screenshot)
        else:
            data._move_location_command = None
        time.sleep(0.2)