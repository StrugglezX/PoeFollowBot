# Defines the multi threaded data type

class PoeNeuronData(): 
    def __init__(self):
        self._text_command = None
        self._player_location = None
        self._move_location_command = None
        self._screenshot = None
        self._screenshot_number = 0
        self._frames_per_second = 10
        self._screenshot_sleep_interval = 1.0 / self._frames_per_second
        self._detected_objects = {}
        self._fog_coordinate = None
        self._escape = False
        
class ObjectDetection():
    def __init__(self, x, y, time):
        self._x = x
        self._y = y
        self._time = time