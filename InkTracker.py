from Singleton import Singleton
import time, threading
from threading import Thread
import numpy as np
import cv2

from ImageTransformer import ImageTransformer

class InkTracker(metaclass=Singleton):

    def __init__(self):
        self.debug = False

        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0
        
        self.previous_state = False
        self.current_state = False
        self.have_printed_character = False
        self.red_high_hsv = np.array([23, 232, 232])
        self.red_low_hsv = np.array([0, 199, 156])

        self.video_stream = None

        self.data_exporter = None


        self.__go_flag = threading.Event() # Create new flag 'Event'
        self.__go_flag.set() # set flag to True

        self.__is_running = threading.Event()
        self.__is_running.set()

    def set_video_stream(self, video_stream):
        """
        Sets self.video_stream to a VideoStream object
        reference. This will be used in the thread operations
        to retrieve the most recent frame.
        """
        self.video_stream = video_stream

    def set_start_x(self, x: int):
        self.start_x = x

    def set_start_y(self, y: int):
        self.start_y = y

    def set_end_x(self, x: int):
        self.end_x = x

    def set_end_y(self, y: int):
        self.end_y = y
    
    def set_roi(self, startx: int, starty: int, endx: int, endy: int):
        self.start_x = startx
        self.start_y = starty
        self.end_x = endx
        self.end_y = endy

    def get_roi(self):
        return (self.start_x, self.start_y, self.end_x, self.end_y)

    def set_red_hsv_ranges(self, red_high_hsv: np.array, red_low_hsv: np.array):
        self.red_high_hsv = red_high_hsv
        self.red_low_hsv = red_low_hsv

    def set_red_high_hsv(self, red_high_hsv: np.array):
        self.red_high_hsv = red_high_hsv

    def set_red_low_hsv(self, red_low_hsv: np.array):
        self.red_low_hsv = red_low_hsv

    def get_current_state(self):
        """
        Returns variable current_state
        It is True if a physical keystroke was detected.
        Prints False otherwise.
        """
        return self.current_state

    def get_previous_state(self):
        """
        Returns variable previous_state
        It is True if a physical keystroke was detected.
        Prints False otherwise.
        """
        return self.previous_state


    def start(self):
        """
        Creates specific thread for running the OCR.
        """
        Thread(target=self.ink).start()
        return self

    def is_running(self):
        """
        Returns whether the thread is running or not.
        It will return True if the thread is running.
        It will return False if it is paused or permanently
            stopped or exited.
        """
        return self.__is_running.is_set() & self.__go_flag.is_set()

    def pause(self):
        """
        If running, the thread will pause operatoins.
        If stop() or exit() have been called, the
        thread cannot be resumed or paused.
        """
        self.__go_flag.clear() # Sets __go_flag to FALSE

    def resume(self):
        """
        If paused, the thread will resume operations.
        If stop() or exit() have been called, the
        thread cannot be resumed or paused.
        """
        self.__go_flag.set() # Sets __go_flag to True

    def stop(self):
        """
        Permanently stops thread, then runs self.exit_gracefully()
        """
        self.resume()
        self.__is_running.clear()
    
    def exit(self):
        """
        Calls self.stop()
        """
        self.stop()

    def exit_gracefully(self):
        # what should we do? 
        # maybe free up some memory???
        # idk something to do with the model.
        print(f'Goodbye from {self.__class__.__name__}!')

    def set_data_exporter(self, data_exporter):
        """
        Configure data exporter for saving frames and stuff.
        """
        self.data_exporter = data_exporter

    def new_data_exporter(self):
        return 

    def ink(self):
        """
        InkTracker Processing loop. Runs in separate thread.
        Start thread using start()
        """
        while self.__is_running.is_set():
            self.__go_flag.wait() # If __go_flag is False it will not run
                                    # until set to True, therefore 'pausing'
                                    # the thread.
                
            
            #print(f'{time.ctime(time.time())}' )
            if self.video_stream is not None:

                frame = self.video_stream.frame.copy()

                self.previous_state = self.current_state
                self.data_exporter["previous_state"] = self.data_exporter["current_state"]
                
                # Ensure ink ROI is within bounds before cropping
                c_start_x = min(frame.shape[1], self.start_x)
                c_start_y = min(frame.shape[0], self.start_y)
                c_end_x = min(frame.shape[1], self.end_x)
                c_end_y = min(frame.shape[0], self.end_y)

                
                cropped_frame = ImageTransformer.frame_crop_frame(frame, c_start_x, c_start_y, c_end_x, c_end_y)

                # CRITICAL FRAME CHECK INSIDE LIVE OCR LOOP
                if cropped_frame is None or cropped_frame.size == 0:
                    print("Warning: Ink ROI frame is empty.")
                    time.sleep(0.01)
                    continue
                    
                hsv_frame = ImageTransformer.frame_bgr_to_hsv(cropped_frame)
                
                #mask = cv2.inRange(hsv_frame, self.red_low_hsv, self.red_high_hsv)
                mask = ImageTransformer.in_range_hsv(hsv_frame, self.red_low_hsv, self.red_high_hsv)
                
                # make rectangle around red color
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cropped_frame = cv2.drawContours(cropped_frame, contours, -1, (255,0,0), 3)
                self.data_exporter["filtered_frames"]["0"] = cropped_frame.copy()

                area = 0
                if len(contours) > 0:
                    shape = contours[0]
                    area = cv2.contourArea(shape)
                    
                width = c_end_x - c_start_x
                height = c_end_y - c_start_y
                area_roi = width*height

                if(area >= 0.05*area_roi):
                    self.current_state = True
                    self.data_exporter["current_state"] = True
                    #print('COLORRRRRRRRRRR')
                else:
                    self.current_state = False
                    self.data_exporter["current_state"] = False
                
                #ocr.show_target_resized(cropped_frame)

                time.sleep(0.01)
            else:
                print(f"ink() thread: video_stream not set. :(")
                time.sleep(0.02)
                # SELFFF

        self.exit_gracefully()
