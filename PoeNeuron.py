import os
import sys
sys.path.append(os.getcwd())
import threading
import PoeNeuronData
#import PoeNeuronTextThread
#import PoeNeuronFollowPlayerThread
import PoeNeuronMovementThread
import PoeNeuronObjectDetectionThread
import PoeNeuronScreenshotThread
import PoeNeuronFogThread
import tkinter
import pynput


def hook_keyboard(data):
    from pynput.keyboard import Key
    from pynput.keyboard import Listener


    def on_press(key):
        pass

    def on_release(key):
        try:
            if key.char == 'a' or key == Key.esc:
                data._escape = True
        except:
            return

    # Collect events until released
    with Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()
        
if __name__ == "__main__":
    data = PoeNeuronData.PoeNeuronData()
    """
    text_thread = threading.Thread(target=PoeNeuronTextThread.PoeNeuronTextThread, args=(data,))
    text_thread.setDaemon(True)
    text_thread.start()
    """
    
    
    """
    follow_player_thread = threading.Thread(target=PoeNeuronFollowPlayerThread.PoeNeuronFollowPlayerThread, args=(data,))
    follow_player_thread.setDaemon(True)
    follow_player_thread.start()
    """
    
    movement_thread = threading.Thread(target=PoeNeuronMovementThread.PoeNeuronMovementThread, args=(data,))
    movement_thread.setDaemon(True)
    movement_thread.start()
    
    screenshot_thread = threading.Thread(target=PoeNeuronScreenshotThread.PoeNeuronScreenshotThread, args=(data,))
    screenshot_thread.setDaemon(True)
    screenshot_thread.start()
    
    detection_thread = threading.Thread(target=PoeNeuronObjectDetectionThread.PoeNeuronObjectDetectionThread, args=(data,))
    detection_thread.setDaemon(True)
    detection_thread.start()
    
    fog_thread = threading.Thread(target=PoeNeuronFogThread.PoeNeuronFogThread, args=(data,))
    fog_thread.setDaemon(True)
    fog_thread.start()
    
    keyboard_thread = threading.Thread(target=hook_keyboard, args=(data,))
    keyboard_thread.setDaemon(True)
    keyboard_thread.start()
    
    
       
    
    root = tkinter.Tk()
    text_command = tkinter.StringVar()
    player_position_str = tkinter.StringVar()

    def update_display():
        text_command.set(data._text_command)
        
        if data._move_location_command:
            player_position_str.set("{} x {}".format(data._move_location_command[0], data._move_location_command[1]))
        else:
            player_position_str.set(None)
            
        root.after(1000,update_display)
        root.update_idletasks()
        
    label1 = tkinter.Label(root, textvariable=text_command)
    label1.pack()
    
    label2 = tkinter.Label(root, textvariable=player_position_str)
    label2.pack()
    
    root.after(1000,update_display)
    
    root.mainloop()
    
    
 