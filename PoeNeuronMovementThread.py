
import time
import math
from pynput.mouse import Button, Controller as MouseController
from pynput import keyboard
import pynput

mouse = MouseController()
keyboardController = keyboard.Controller()

def valid_move_commands():
    return [
            'stop',
            'follow',
            'portal',
            'door',
            ]
            
deadzone = 100
movement_space = deadzone * 2

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
potion_idx = 0;
    
def potions():
    return [
        keyboard.KeyCode.from_char('1'), 
        keyboard.KeyCode.from_char('2'), 
        keyboard.KeyCode.from_char('3'), 
        keyboard.KeyCode.from_char('4'), 
        keyboard.KeyCode.from_char('5'),
        ]
def use_potion():
    global potion_idx
    potion_idx = potion_idx + 1
    if potion_idx > len(potions()) - 1:
        potion_idx = 0
    keyboardController.press(potions()[potion_idx])
    keyboardController.release(potions()[potion_idx])
    
potion_toggle_counter = 0;
def PoeNeuronMovementThread(data):
    while True:
        global potion_toggle_counter
        potion_toggle_counter = potion_toggle_counter + 1
        if potion_toggle_counter == 10:
            potion_toggle_counter = 0
            use_potion()
            
        if data._text_command in valid_move_commands() and data._player_location and data._move_location_command:
            change_in_x = data._move_location_command[0]-data._player_location[0]
            change_in_y = data._move_location_command[1]-data._player_location[1]
            move_angle_radians = math.atan2(change_in_y, change_in_x)
            mouse_delta_x = math.cos(move_angle_radians) * movement_space
            mouse_delta_y = math.sin(move_angle_radians) * movement_space
            mouse_pos = (data._player_location[0] + mouse_delta_x, data._player_location[1] + mouse_delta_y)
            move_mouse(mouse_pos[0], mouse_pos[1])
            left_mouse_click()
            
        time.sleep(0.2)