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
def get_next_screenshot():
    tid = threading.get_ident()
    if threads_last_screenshot_count:
        while threads_last_screenshot_count < data._screenshot_number:
            sleep(data._screenshot_sleep_interval)
    return (data._screenshot, data._screenshot_number)