import numpy as np
import cv2
import typing
from typing import Optional
from typing import Tuple

class ImageTransformer():
    @staticmethod
    def frame_resize(frame, dimensions: Tuple[int, int]):
        """
        Resizes to dimensions.
        frame is an image frame
        dimensions is an integer tuple for (width, height)
        """
        return cv2.resize(frame, dsize=dimensions)

    @staticmethod
    def frame_crop_frame(frame, startx, starty, endx, endy):
        """
        Crops frame to coordinate rectangle.
        """
        return frame[ starty:endy, startx:endx ]
    
    @staticmethod
    def frame_bgr_to_gray(frame):
        """
        Converts BGR frame to Grayscale.
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def frame_bgr_to_hsv(frame):
        """
        Converts BGR frame to HSV.
        """
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    @staticmethod
    def frame_gaussian_blur(frame, kernel_size: Optional[Tuple[int, int]] = (5, 5), kernel_deviation: Optional[Tuple[int, int]] = (0, 0) ):
        """
        Applies Gaussian Blur to frame.
        Optional parameters
        """
        return cv2.GaussianBlur(frame, ksize=kernel_size, sigmaX=kernel_deviation[0], sigmaY=kernel_deviation[1])

    @staticmethod
    def frame_threshold_binary(frame, max_value: Optional[int] = 255, block_size: Optional[int] = 21, c: Optional[int] = 10): 
        """
        Applies Adaptive Gaussian Threshold to frame,
        using the cv2.ADAPTIVE_THRESH_GAUSSIAN_C and
            cv2.THRESH_BINARY
        """
        return cv2.adaptiveThreshold(
                    frame,
                    max_value,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block_size,
                    c)

    @staticmethod
    def frame_morphology_ex(frame, operation: int = cv2.MORPH_OPEN, iterations: int = 1):
        """
        Applies erosion or dilation filters to frame.
        """
        kernel = np.ones( (4, 4), np.uint8 )
        return cv2.morphologyEx(frame, operation, kernel, iterations=iterations)

    @staticmethod
    def frame_morph_open(frame, iterations: Optional[int] = 2):
        """
        uses cv2.MORPH_OPEN
        defaults to 2 iterations
        calls frame_morphology_ex
        """
        return ImageTransformer.frame_morphology_ex(frame, operation=cv2.MORPH_OPEN, iterations=iterations)

    @staticmethod
    def frame_morph_close(frame, iterations: Optional[int] = 1):
        """
        uses cv2.MORPH_CLOSE
        defaults to 1 iteration
        calls frame_morphology_ex
        """
        return ImageTransformer.frame_morphology_ex(frame, operation=cv2.MORPH_CLOSE, iterations=iterations)


    @staticmethod
    def inRange(hsv_frame, low_hsv, high_hsv):
        """
        cv2.inRange(hsv_frame, low_hsv, high_hsv)
        returns binary mask
        """
        return cv2.inRange(hsv_frame, low_hsv, high_hsv)
