import os
import torch
from fastai.vision.all import *
from fastai.data.core import TfmdLists
from fastai.vision.core import PILImageBW
from PIL import Image
import torch.nn.functional as F
import threading
from threading import Thread
import cv2

from Singleton import Singleton



import ImageTransformer
from ImageTransformer import ImageTransformer

class OCR( metaclass=Singleton ):
    """
    Runs Optical Character Recognition on specific
    image frames.
    """

    OCR_MODEL_PATH = "./typewriter_ocr_model"
    
    def __init__(self):
        self.debug = False
        self.start_x = 0
        self.start_y = 0
        self.crop_width = None
        self.crop_height = None
        self.default_width = None
        self.default_height = None
        self.predicted_char = None
        self.predicted_confidence = None

        self.__go_flag = threading.Event() # Create new flag 'Event'
        self.__go_flag.set() # set flag to True

        self.__is_running = threading.Event()
        self.__is_running.set()
        self.frame = None
        self.transformed_frame = None
        self.video_stream = None
        self.data_exporter = None


        self.learn = None
        
    def start(self):
        """
        Creates specific thread for running the OCR.
        """
        Thread(target=self.ocr).start()
        return self

    def start_ai_model(self, model_path = OCR_MODEL_PATH):
        """
        Starts AI Model
        """
        # Attempt to load the real learner, fallback to mock if path fails
        try:
            self.learn = load_learner(model_path, cpu=True)
            self.learn.model.eval()
        except (FileNotFoundError, Exception) as e:
            print(f"ERROR: OCR model not found at {model_path}.")
            print(f"{e}")
    
    def set_video_stream(self, video_stream):
        """
        Sets self.video_stream to a VideoStream object
        reference. This will be used in the thread operations
        to retrieve the most recent frame.
        """
        self.video_stream = video_stream

    def set_data_exporter(self, data_exporter):
        """
        Sets data_exporter to a dictionary reference that will
        be used to export specific frames.
        """
        self.data_exporter = data_exporter
    
    def generate_new_data_exporter(self):
        """
        Generates a new data exporter in case there was none
        passed through set_data_exporter
        """
        return {
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

    def set_crop_location(self, start_x: int, start_y: int):
        self.start_x = start_x
        self.start_y = start_y
        return (self.start_x, self.start_y)

    def get_crop_location(self):
        return (self.start_x, self.start_y)

    def set_crop_dimensions(self, width: int, height: int):
        """
        Set crop dimensions from frame.
        """
        self.crop_width = width
        self.crop_height = height
        return (self.crop_width, self.crop_height)

    def set_crop_width(self, width: int):
        self.crop_width = width

    def set_crop_height(self, height: int):
        self.crop_height = height

    def set_default_dimensions(self, default_width: int, default_height: int):
        """
        Sets default dimensions for input frame.
        The OCR function will automatically rescale to these dimensions.
        """
        self.default_width = default_width
        self.default_height = default_height
        return (self.default_width, self.default_height)
    
    def get_default_dimensions(self):
        """
        Returns a tuple of default_width and default_height
        """
        return (self.default_width, self.default_height)

    def get_crop_dimensions(self):
        """
        Returns a tuple of width and height
        """
        return (self.crop_width, self.crop_height)

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
        print(f'Goodbye from OCR!')

    def predict_single_character(self, frame):
        """
        Predicts a single character from a numpy array (image data) using a fastai Learner.

        Args:
            frame (np.ndarray): The input image data (grayscale)

        Returns:
            tuple: (predicted_character_string, confidence_float)
        """

        try:
            # Convert NumPy array (grayscale/binary) back to PILImageBW format for fastai
            # 1. Ensure the image data is grayscale (2D)
            if frame.ndim == 3:
                # Assuming it's BGR, convert to grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 2. FIX: Convert the NumPy array to a PIL Image object first.
            # The PILImageBW constructor expects an existing PIL Image, not the raw array data.
            pil_img_object = Image.fromarray(frame)
            
            # 3. Instantiate the fastai wrapper class with the PIL Image object
            img = PILImageBW(pil_img_object)

            # Create a test DataLoader
            test_dl = self.learn.dls.test_dl([img], 
                                        after_item=self.learn.dls.valid.after_item, 
                                        after_batch=self.learn.dls.valid.after_batch)

            # Get the processed input tensor (xb)
            xb, = test_dl.one_batch()

        except Exception as e:
            return f"Error during image processing: {e}", 0.0

        # --- 3. Get Raw Prediction and Decode
        try:
            with torch.no_grad():
                raw_output = self.learn.model(xb) # Get raw logits

                if raw_output.ndim == 1:
                    raw_output = raw_output.unsqueeze(0)

                predicted_index = raw_output.argmax(dim=1).item()

                vocab = self.learn.dls.vocab
                predicted_char = vocab[predicted_index]

                # Calculate confidence using softmax
                confidence = F.softmax(raw_output, dim=1)[0, predicted_index].item()

                #print(f'********* char is {predicted_char}')
                #print(f'********* conf is {confidence}')

                return predicted_char, confidence
                
        except Exception as e:
            return f"Error during model inference character recognition: {e}", 0.0

    def get_latest_prediction(self):
        return (self.predicted_char, self.predicted_confidence)

    def ocr(self):
        """
        OCR Processing loop. Runs in separate thread.
        Start thread using start()
        """
        while self.__is_running.isSet():
            self.__go_flag.wait() # If __go_flag is False it will not run
                                    # until set to True, therefore 'pausing'
                                    # the thread.
            if self.debug:
                print(f'Running loop {time.ctime(time.time())}')
            if self.video_stream is not None:
                frame = self.video_stream.frame 

                if frame is None:
                    print(f"ocr() thread: frame is None. BAD :(")
                    continue

                if self.data_exporter is None:
                    self.data_exporter = self.generate_new_data_exporter()
                    print("""Generated new data exporter because none was
                            given before""")

                
                # crop image
                end_x = self.start_x + self.crop_width
                end_y = self.start_y + self.crop_height

                if self.debug:
                    print(f"OCR:\n\t{self.start_x}, {self.start_y}\n\t{end_x}, {end_y}")

                self.data_exporter["filtered_frames"]["0"] = frame.copy()

                frame = ImageTransformer.frame_crop_frame(frame, self.start_x, self.start_y, end_x, end_y)
                #cv2.imshow("f1", frame)

                if (self.crop_width != self.default_width) or (self.crop_height != self.default_height):
                    frame = ImageTransformer.frame_resize(frame, 
                        (self.default_width, self.default_height) )

                #cv2.imshow("f2", frame)

                frame = ImageTransformer.frame_bgr_to_gray(frame)
                #cv2.imshow("f3", frame)
                frame = ImageTransformer.frame_gaussian_blur(frame)
                #cv2.imshow("f4", frame)
                frame = ImageTransformer.frame_threshold_binary(frame)
                #cv2.imshow("f5", frame)
                frame = ImageTransformer.frame_morph_open(frame)
                #cv2.imshow("f6", frame)
                frame = ImageTransformer.frame_morph_close(frame)
                #cv2.imshow("f7", frame)

                self.transformed_frame = frame

               # self.transformed_frame = ImageTransformer.frame_crop_frame(frame, 50, 50, 100, 100)
               # run predict_single_character(frame, learn)
                self.predicted_char, self.predicted_confidence = self.predict_single_character(frame)

                
                time.sleep(0.01)
                # SELFFF
                
            else:

                print(f"ocr() thread: video_stream not set. :(")
                time.sleep(0.02)
                continue


        self.exit_gracefully()


