from Singleton import Singleton
import time, threading
from threading import Thread
import numpy as np
import cv2

from ImageTransformer import ImageTransformer

class OpticalTracker(metaclass=Singleton):

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

        self.shiTomasiCornerParams = dict(
            maxCorners = 5,
            qualityLevel = 0.2,
            minDistance = 60,
            blockSize = 7 )

        self.lucasKanadeParams = dict(
            winSize = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                10,
                0.03 ) )

        self.randomColors = np.random.randint(0, 255, (100, 3) )

        self.firstFrame = None

        self.frame_gray_previous = None
        self.previous_corners = None

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
        Thread(target=self.OpticalTracker).start()
        return self

    def is_running(self):
        """
        Returns whether the thread is running or not.
        It will return True if the thread is running.
        It will return False if it is paused or permanently
            stopped or exited.
        """
        return self.__is_running.isSet() & self.__go_flag.isSet()

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

    def OpticalTracker(self):
        """
        InkTracker Processing loop. Runs in separate thread.
        Start thread using start()
        """
        while self.__is_running.isSet():
            self.__go_flag.wait() # If __go_flag is False it will not run
                                    # until set to True, therefore 'pausing'
                                    # the thread.
                
            
            #print(f'{time.ctime(time.time())}' )
            if self.video_stream is not None:

                frame = self.video_stream.frame.copy()
                frame = ImageTransformer.frame_crop_frame(frame, 200, 50, 600, 546)

                # setup
                if self.firstFrame is None:
                    self.firstFrame = frame
                    if self.firstFrame is None:
                        print(f'OpticalTracker(): unreadable frame.')
                        continue

                    self.frame_gray_previous = ImageTransformer.frame_bgr_to_gray(self.firstFrame)
                    self.previous_corners = cv2.goodFeaturesToTrack( self.frame_gray_previous,
                                                                mask=None,
                                                                **self.shiTomasiCornerParams)
                    
                    # new_corners = cv2.selectROI(self.firstFrame) 

                    # previous_corners = np.array([[[new_corners[0], new_corners[1] ]]], dtype=np.float32)
                    
                    mask = np.zeros_like(self.firstFrame)
                
                # end setup
                
                # main loop
                
                frame_gray_current = ImageTransformer.frame_bgr_to_gray(frame)

                current_corners, found_status, _ = cv2.calcOpticalFlowPyrLK(
                    self.frame_gray_previous, frame_gray_current, self.previous_corners, None,
                    **self.lucasKanadeParams)
        
                if current_corners is not None:
                    cornersMatchedCur = current_corners[found_status==1]
                    cornersMatchedPrev = self.previous_corners[found_status==1]

                for i, (curCorner, prevCorner) in enumerate(zip(
                    cornersMatchedCur, cornersMatchedPrev)):
                    xCur, yCur = curCorner.ravel()
                    xPrev, yPrev = prevCorner.ravel()

                    mask = cv2.line(mask, ( int(xCur), int(yCur) ),
                    ( int(xPrev), int(yPrev) ), self.randomColors[1].tolist(), 2)

                    frame = cv2.circle(frame, (int(xCur), int(yCur)), 5,
                        self.randomColors[i].tolist(), -1)
                    
                    frame = cv2.add(frame, mask)
                
                self.frame_gray_previous = frame_gray_current.copy()
                self.previous_corners = cornersMatchedCur.reshape(-1, 1, 2)

                self.data_exporter["frames"]["result"] = frame.copy()
                 

                

            else:
                print(f"OpticalTracker() thread: video_stream not set. :(")
                time.sleep(0.02)
                # SELFFF

        self.exit_gracefully()
