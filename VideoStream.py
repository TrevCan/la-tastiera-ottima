import cv2
import threading
from threading import Thread

from Singleton import Singleton

class VideoStream (metaclass=Singleton):
    """
    Class for grabbing frames from CV2 video capture in a dedicated thread.
    Generate a new object with VideoStream.VideoStream(src)
    Indicate the index for the video as an integer.
    Then from the object call .start()
    Now you can grab whenever you want from the object the frame
    To stop the thread, run stop_process()
    """
    def __init__(self, src):
        """
        Generates a new VideoStream object.
        @param src. Mandatory parameter for the video source index
                    as an integer.
        """
        print(f'Starting video stream on source: {src}')
        self.stream = cv2.VideoCapture(src)
        
        # Check if stream opened successfully
        if not self.stream.isOpened():
            print(f"ERROR: Cannot open video stream source {src}.")
            self.grabbed = False
            self.frame = None
        else:
            (self.grabbed, self.frame) = self.stream.read()
            
        ### --Start-- Thread operation flags
        ## Flags for
        ## RESUME, PAUSE, and STOP
        self.__go_flag = threading.Event() # Create new flag 'Event'
        self.__go_flag.set() # set flag to True

        self.__is_running = threading.Event()
        self.__is_running.set()
        ### --End-- Thread operation flags

    def start(self):
        """
        Creates a thread targeted at get(), which reads frames from CV2 VideoCapture
        """
        if self.stream.isOpened():
            Thread(target=self.get, args=()).start()
        return self

           
    def get_video_dimensions(self):
        """
        Gets the width and height of the video stream frames
        """
        if self.stream.isOpened():
            width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
            return int(width), int(height)
        return 0, 0

    def shape(self):
        """
        Get frame shape in (y, x) format.
        cv2 likes it like that!!!
        """
        x, y = self.get_video_dimensions()
        return (y, x)

    def release(self):
        """
        Calls self.stop()
        """
        self.stop()
    
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
        Permanently stops thread, then runs self.exit_gracefully() from the thread.
        """
        self.resume()
        self.__is_running.clear()
    
    def exit(self):
        """
        Calls self.stop()
        """
        self.stop()

    def exit_gracefully(self):
        if self.stream.isOpened():
            self.stream.release()
            print(f"VideoStream {self.stream} successfully closed.")

        print(f'\nGoodbye from {self.__class__.__name__}!')

    def get(self):
        """
        Continuously gets frames from CV2 VideoCapture and sets them as self.frame attribute
        """
        while self.__is_running.isSet():
            self.__go_flag.wait() # If __go_flag is False it will not run
                                    # until set to True, therefore 'pausing'
                                    # the thread.
                
            # We explicitly check if the stream is still open
            if self.stream.isOpened():
                (self.grabbed, self.frame) = self.stream.read()
                #print(f'Grab frame: {self.grabbed}')
            else:
                self.frame = None
        
        # After closing, run exit_gracefully
        self.exit_gracefully()
 
