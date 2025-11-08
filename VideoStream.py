import cv2
import threading
from threading import Thread


class VideoStream:
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
            
        self.stopped = False

    def start(self):
        """
        Creates a thread targeted at get(), which reads frames from CV2 VideoCapture
        """
        if self.stream.isOpened():
            Thread(target=self.get, args=()).start()
        return self

    def get(self):
        """
        Continuously gets frames from CV2 VideoCapture and sets them as self.frame attribute
        """
        while not self.stopped:
            # We explicitly check if the stream is still open
            if self.stream.isOpened():
                (self.grabbed, self.frame) = self.stream.read()
                #print(f'Grab frame: {self.grabbed}')
            else:
                self.frame = None
                self.stopped = True # Stop the thread if stream is no longer open
            
    def get_video_dimensions(self):
        """
        Gets the width and height of the video stream frames
        """
        if self.stream.isOpened():
            width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
            return int(width), int(height)
        return 0, 0

    def stop_process(self):
        """
        Sets the self.stopped attribute as True and kills the VideoCapture stream read
        """
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()
            print(f"VideoStream {self.stream} successfully closed.")

    def release(self):
        """
        Calls self.stop_process()
        """
        self.stop_process()
    
    def close(self):
        """
        Calls self.stop_process()
        """
        self.stop_process()

    def exit(self):
        """
        Calls self.stop_process()
        """
        self.stop_process()
