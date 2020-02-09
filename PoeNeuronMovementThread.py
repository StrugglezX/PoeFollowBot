
import time
import math
from pynput.mouse import Button, Controller as MouseController
import keyboard
import pynput
import time
import random
from time import sleep
from PoeNeuronData import ObjectDetection
import copy

mouse = MouseController()


def move_mouse(x, y):
    def set_mouse_position(x, y):
        mouse.position = (int(x), int(y))
    def smooth_move_mouse(from_x, from_y, to_x, to_y, speed=0.2):
        steps = 40
        sleep_per_step = speed // steps
        x_delta = (to_x - from_x) / steps
        y_delta = (to_y - from_y) / steps
        for step in range(steps):
            new_x = x_delta * (step + 1) + from_x
            new_y = y_delta * (step + 1) + from_y
            set_mouse_position(new_x, new_y)
            time.sleep(sleep_per_step)
    return smooth_move_mouse(
        mouse.position[0],
        mouse.position[1],
        x,
        y
    )

def left_mouse_click():
    mouse.click(Button.left)
    
def right_mouse_click():
    mouse.click(Button.right)
    
def get_monster_list():
    return [
            'ZOMBIES',
        ]
        
def get_closest_monster(data):
    coodinate = None
    detected_objects = copy.deepcopy(data._detected_objects)
    
    for key in detected_objects.keys():
        if key not in get_monster_list():
            continue
        
        object_coordinates = detected_objects[key]
        for object_coordinate in list(object_coordinates):
            #deadzone check
            staleness = get_current_time_ms() - object_coordinate._time
            if staleness < 3000:                
                print( 'Attacking {} at {}x{}'.format(key, object_coordinate._x, object_coordinate._y ) )
                return (object_coordinate._x, object_coordinate._y)
            else:
                print('removing stale {} at {}x{}'.format(key, object_coordinate._x, object_coordinate._y ) )
                object_coordinates.remove(object_coordinate)
    
    return coodinate
    
def get_closest_lootable(data):
    return None
    
def get_current_time_ms():
    return int(round(time.time() * 1000))
    
def is_cooldown_over(last_time, total_cooldown_time):
    current_time = get_current_time_ms()
    time_elapsed = current_time - last_time
    return time_elapsed > total_cooldown_time
    
def PoeNeuronMovementThread(data):

    last_attack_time = 0
    total_attack_cooldown = 0 #ms
    
    while True:
        sleep_time = random.random() * 0.5
        sleep( sleep_time )
        
        """
        if keyboard.is_pressed('a'):
            print('sleeping for 10s')
            sleep(10)
        """
        
        
        ## ATTACK
        coordinate = get_closest_monster( data )
        if is_cooldown_over(last_attack_time, total_attack_cooldown) and coordinate != None:
            move_mouse( coordinate[0], coordinate[1] );
            right_mouse_click()
            last_attack_time = get_current_time_ms()
            continue
        
        ## LOOT
        
        
        ## MOVE
        if data._fog_coordinate != None:  
            print('clicking fog at {}x{}'.format(data._fog_coordinate[0], data._fog_coordinate[1]))
            move_mouse( data._fog_coordinate[0], data._fog_coordinate[1] );
            left_mouse_click()
            continue
        
        