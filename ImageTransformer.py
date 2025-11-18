import numpy as np
import cv2
import typing
from typing import Optional
from typing import Tuple

class ImageTransformer:
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
    def frame_bgr_to_rgba(frame):
        """
        Converts BGR frame to RGBA (Red, Green, Blue, Alpha)
        uses opencv's cv2.cvtColor
        Returns RGBA frame
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        return frame

    @staticmethod
    def frame_gray_to_rgba(frame):
        """
        Converts Gray (binary) frame to RGBA (Red, Green, Blue, Alpha)
        uses opencv's cv2.cvtColor
        Returns RGBA frame
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGBA)
        return frame

    @staticmethod
    def frame_bgr_to_rgba_normalized(frame):
        """
        Converts BGR frame into a normalized (0 to 1 float)
        RGBA frame.
        This is useful for rendering it in DearPyGUI
        """
        video_texture = ImageTransformer.frame_bgr_to_rgba(frame)
        video_texture = video_texture.ravel()
        video_texture = np.asarray(video_texture, dtype=np.float32)
        video_texture = np.true_divide(video_texture, 255.0)
        return video_texture

    @staticmethod
    def frame_gray_to_rgba_normalized(frame):
        """
        Converts Gray (binary) frame into a normalized (0 to 1 float)
        RGBA frame.
        This is useful for rendering it in DearPyGUI
        """
        video_texture = ImageTransformer.frame_gray_to_rgba(frame)
        video_texture = video_texture.ravel()
        video_texture = np.asarray(video_texture, dtype=np.float32)
        video_texture = np.true_divide(video_texture, 255.0)
        return video_texture

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
    def in_range_hsv(hsv_frame, low_hsv, high_hsv):
        """
        cv2.in_range(hsv_frame, low_hsv, high_hsv)
        returns binary mask
        """
        return cv2.inRange(hsv_frame, low_hsv, high_hsv)

    @staticmethod
    def get_binary_mask_polygon(frame_shape, polygon_coordinates):
        """
        Returns mask with width and height based on @param frame_shape.
        It will set every pixel value within the polygon to 255.
        The polygon must be passed as an array of coordinates in
        clockwise order or counter clockwise order as per cv2.fillPoly().
        [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ]
        """
        if len(polygon_coordinates) < 3:
            # this is NOT a polygon!
            return None
        mask = np.zeros(frame_shape, dtype=np.uint8)
        vertices = np.array(polygon_coordinates, dtype=np.int32)

        cv2.fillPoly(mask, [vertices], 255)

        return mask


    @staticmethod
    def center_image_in_texture(frame, target_width, target_height):
        """
        Creates a frame of width target_width and height target_height.
        Centers `frame`. (Must be of equal or smaller width and height).
        The rest will be black.

        Args:
            frame: Input image.
            target_width: desired width of the frame
            target_height: desired height of the frame

        Returns:
            Padded image of width target_width, height target_height.
            The channel format is returned as in frame.
        """
        h, w = frame.shape[:2]
        pad_top = (target_height - h) // 2
        pad_left = (target_width - w) // 2
        pad_bottom = target_height - h - pad_top
        pad_right = target_width - w - pad_left

        # Pad with black (0s)
        padded = cv2.copyMakeBorder(
            frame,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0, 0]  # Black for RGBA
        )
        return padded


