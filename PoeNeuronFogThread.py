
import PoeNeuronScreenshotThread

def PoeNeuronFogThread(data):
    while True:
        image = PoeNeuronScreenshotThread.get_next_screenshot(data)
        total_width, total_height = image.size
        start_x = int(total_width * 0.2)
        end_x = int(total_width * 0.8)
        start_y = int(total_height * 0.1)
        end_y = int(total_height * 0.9)
        current_width = start_x
        current_height = start_y
        width_increment = 4
        height_increment = 4
        
        red_tolerance = 5
        green_tolerance = 5
        blue_tolerance = 5
        
        expected_pixel = (38, 133, 180)
        found_coordinate = None
        
        while current_width < end_x and not found_coordinate:
        
            while current_height < end_y and not found_coordinate:
                coordinate = (current_width, current_height)
                pixel = image.getpixel(coordinate)
                red_diff = abs(pixel[0] - expected_pixel[0])
                green_diff = abs(pixel[1] - expected_pixel[1])
                blue_diff = abs(pixel[2] - expected_pixel[2])
                
                if red_diff < red_tolerance and green_diff < green_tolerance and blue_diff < blue_tolerance:
                    found_coordinate = coordinate
            
                current_height = current_height + height_increment
            current_width = current_width + width_increment
            current_height = start_y
        data._fog_coordinate = found_coordinate