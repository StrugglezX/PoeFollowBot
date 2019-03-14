# Responsible for parsing the current text command

import time
from PIL import Image, ImageGrab
import matplotlib.pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

def try_hard_but_slower_config():
    return '--psm 12'
    
def find_text_on_image(image):    
    return pytesseract.image_to_string(image, config=try_hard_but_slower_config())  
    
def take_screenshot():
    image = ImageGrab.grab(bbox=(0, 500, 450, 500+200))
    return image
    
def command_list():
    return [
                'stop', # follow my character
                'follow', # follow my character
                'portal', # enter portal
                'door', # enter door
            ]
       
def get_command_from_text(full_text, data):
    for command in command_list():
        if command.upper() in full_text.upper():
            return command
    return data._text_command
            
def display_image(frame):
    imgplot = plt.imshow(frame)
    plt.show()
    
def PoeNeuronTextThread(data):
    while True:
        image = take_screenshot()
        display_image(image)
        full_text = find_text_on_image(image)
        data._text_command = get_command_from_text(full_text, data)
        time.sleep(5)