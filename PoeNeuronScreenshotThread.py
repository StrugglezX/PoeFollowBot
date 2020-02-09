# Take screenshots and store them in data

from PIL import ImageGrab
from time import sleep
import threading

def PoeNeuronScreenshotThread(data):
    while True:
        data._screenshot = ImageGrab.grab()
        data._screenshot_number = data._screenshot_number + 1
        sleep(data._screenshot_sleep_interval)
        
thread_last_screenshot_dict = {}

def get_next_screenshot(data):
    tid = threading.get_ident()
    if tid not in thread_last_screenshot_dict:
        thread_last_screenshot_dict[tid] = 0
    
    threads_last_screenshot_count = thread_last_screenshot_dict[tid]
    
    while threads_last_screenshot_count > data._screenshot_number or data._screenshot == None:
        print('threads_last_screenshot_count {} < data._screenshot_number {} or data._screenshot == None {}'.format(threads_last_screenshot_count, data._screenshot_number, data._screenshot == None))
        sleep(data._screenshot_sleep_interval)
        
    return data._screenshot