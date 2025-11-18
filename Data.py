from Singleton import Singleton
import numpy as np

class Data(metaclass=Singleton):
    """
    Stores data. Allows for cross-function
    transfer without the need of using 'global'
    variables. Instead, this class is a Singleton
    OOP-based class. It will always return the
    same instance of the class no matter what.
    Of course, the first time it is called the
    instance is created.
    """

    def __init__(self):
        self.mouse_x = 0
        self.mouse_y = 0
        self.ocr_roi = {
            "x": 475,
            "y": 191,
            "ex": 509,
            "ey": 250,
            "w": 34,
            "h": 59,
            "amplification": 0.35417
        }

        self.ink_roi = {
            "x": 594,
            "y": 285,
            "ex": 611,
            "ey": 293
        }

        self.ocr_data = {
            "filtered_frames": {
                "0": [],
                "1": [],
                "2": [],
                "3": [],
                "4": [],
                "5": [],
                "6": [],
                "7": [],
                "8": []
            },
            "parameters": {
                "a": []
            }
        }

        self.ink_data = {
            "filtered_frames": {
                "0": [],
                "1": [],
                "2": [],
                "3": [],
                "4": [],
                "5": [],
                "6": [],
                "7": [],
                "8": []
            },
            "current_state": False,
            "previous_state": False
        }

        self.optical_data = {
            # defines the crop area that will
            # always be read and applied before processing
            # the image in OpticalTracker.py
            "crop_points": (-1, -1, -1, -1),
            "frames": {
                "cropped_frame": None,
                "0": [],
                "result": []
            },
            "results": {
                "average_displacement_magnitude": 0,
                "average_displacement_x": 0,
                "average_displacement_x_rounded": 0,
                "average_displacement_y": 0,
                "average_displacement_y_rounded": 0,
                "shape_p0": 0,
                "shape_p1": 0,
                "shape_p1_n,2": 0,
                "chars_moved": 0
            },
            "new_corners": [],
            "current_status": "black",
            "user_parameters": {
                "pixels_per_inch": -1
            }
        }

        self.commander = {
            "frame": None,
            "checkbox_01": {
                "name": "",
                "state": [False]
            },
            "checkbox_02": {
                "name": "",
                "state": [False]
            },
            "button_optical_set_crop_region": {
                "name": "Set rectangle to new Optical Crop Region",
                "state": False
            },
            "button_optical_set_crop_roi": {
                "name": "Set rectangle to ROI point tracker region",
                "state": False
            }
        }

        ##    if d.optical_roi_set_inch_initiate:
        #        x, y, w, h = cv2.selectROI("select_inch", VideoStream().frame.copy(), showCrosshair=False, fromCenter=False)
        #        d.optical_data["user_parameters"]["pixels_per_inch"]=w
        #

        self.optical_roi_initiate = False
        self.optical_roi_points = []
        self.optical_roi_max_points = 4

        self.optical_roi_mask = None
        self.optical_roi_mask_initiate = False
        self.optical_roi_mask_points = []
        self.optical_roi_mask_max_points = 4

        self.enable_model = False

        self.status_frames = {
            "green": np.full((50, 50, 3), [0, 255, 0], dtype=np.uint8),
            "red": np.full((50, 50, 3), [0, 0, 255], dtype=np.uint8),
            "black": np.full((50, 50, 3), [0], dtype=np.uint8),
            "blue": np.full((50, 50, 3), [255, 0, 0], dtype=np.uint8)
        }

        self.debug = {
            "all": False,
            "simulate_keystroke_wayland": False,
            "rodent_handler_optical_roi_initiate": False,
            "rodent_handler_optical_roi_mask_initiate": False,
            "ocr_stream_dimensions": False,
            "ocr_stream_roi_or_crop_dimensions": False,
        }
