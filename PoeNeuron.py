import os
import sys
sys.path.append(os.getcwd())
import threading
import PoeNeuronData
#import PoeNeuronTextThread
#import PoeNeuronFollowPlayerThread
import PoeNeuronMovementThread
import PoeNeuronAttackThread
import PoeNeuronScreenshotThread
import tkinter

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
    
    attack_thread = threading.Thread(target=PoeNeuronAttackThread.PoeNeuronAttackThread, args=(data,))
    attack_thread.setDaemon(True)
    attack_thread.start()
    
    
    
       
    
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
    
    
 