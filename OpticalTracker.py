from Singleton import Singleton
import time, threading
from threading import Thread
import numpy as np
import cv2

from ImageTransformer import ImageTransformer as img

from round_if_in_range import round_if_in_range

def inRange(n, lower_bound, upper_bound):

    def up(n, min_add):
        return int(n + copysign(min_add, n))
    


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

        self.video_stream = None

        self.data_exporter = None

        self.mask = None
        self.corners = None

        self.p0 = []

        self.target_displacement = 0

        # this should be based off a 
        # percentage from the size of an inch in pixels
        # or any other measure, really, just make it
        # adjustable every time the program is ran.
        self.delta_distance = 10

#        self.shiTomasiCornerParams = dict(
#            maxCorners = 10,
#            qualityLevel = 0.30,
#            minDistance = 20,
#            blockSize = 7 )
#
#        self.lucasKanadeParams = dict(
#            winSize = (15, 15),
#            maxLevel = 2,
#            criteria = (cv2.TERM_CRITERIA_EPS |
#                cv2.TERM_CRITERIA_COUNT,
#                10,
#                0.03 ) )

        self.shiTomasiCornerParams = dict(
            maxCorners = 100,
            qualityLevel = 0.04,
            minDistance = 12,
            blockSize = 7 )

        self.lucasKanadeParams = dict(
            winSize = (23, 23),
            maxLevel = 3,
            criteria = (cv2.TERM_CRITERIA_EPS |
                cv2.TERM_CRITERIA_COUNT,
                10,
                0.03 ),
                minEigThreshold=1e-4 )

# Brave AI's recommendations for tracking a metal ruler with dominant X-motion and small Y-vibrations:
#Shi-Tomasi Parameters
#maxCorners=200 → Reduce to 50–100. A ruler has limited texture; too many points may include noise.
#qualityLevel=0.01 → Increase to 0.02–0.05 to select only stronger corners.
#minDistance=2 → Increase to 10–15 to avoid clustering on edges.
#blockSize=7 → Good for a ruler’s sharp edges; keep it.
#🌀 Lucas-Kanade Parameters
#winSize=(15, 15) → Slightly small. Use (21, 21) or (25, 25) for better patch matching on structured surfaces.
#maxLevel=5 → High. Use 2–3 unless motion is very fast. Higher levels add computation with diminishing returns.
#criteria=(..., 10, 0.03) → Good. Ensures convergence without excessive iterations.
#✅ Additional Tip
#Add minEigThreshold=1e-4 to lucasKanadeParams to filter out poorly tracked points:
#
#lucasKanadeParams = dict(
#    winSize=(21, 21),
#    maxLevel=3,
#    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
#    minEigThreshold=1e-4  # Add this
#)
#

        self.random_colors = np.random.randint(0, 255, (100, 3) )

        self.initial_frame = None

        self.previous_frame_bn = None
        self.previous_corners = None



        ### --Start-- Thread operation flags
        ## Flags for
        ## RESUME, PAUSE, and STOP
        self.__go_flag = threading.Event() # Create new flag 'Event'
        self.__go_flag.set() # set flag to True

        self.__is_running = threading.Event()
        self.__is_running.set()
        ### --End-- Thread operation flags

    def set_video_stream(self, video_stream):
        """
        Sets self.video_stream to a VideoStream object
        reference. This will be used in the thread operations
        to retrieve the most recent frame.
        """
        self.video_stream = video_stream

    def set_mask(self, mask):
        self.mask = mask

    def set_ppi(self, ppi):
        self.ppi = ppi

    def set_corners(self, corners):
        self.corners = corners

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
                # cropping is somewhat unnecessary
                #frame = img.frame_crop_frame(frame, 200, 50, 600, 546)

                # setup
                if self.initial_frame is None:
                    self.initial_frame = frame
                    if self.initial_frame is None:
                        print(f'OpticalTracker(): unreadable frame.')
                        continue

#                    if self.mask is not None:
#                        print(f'self.mask is not None')

                    self.previous_frame_bn = img.frame_bgr_to_gray(self.initial_frame)
                    #print(f"previous_frame_bn.shape is {self.previous_frame_bn.shape}")
                    self.previous_corners = cv2.goodFeaturesToTrack( self.previous_frame_bn,
                                                                mask=self.mask,
                                                                **self.shiTomasiCornerParams)

                    print(f'OG: previous_corners:\n {self.previous_corners}')

                    
                    # user user provided corners if available
                    if self.corners is not None:
                        if len(self.corners) > 0:
                            # used to be this, but really depends on how
                            # you are inputing your data, sometimes
                            # you don't need the []
                            # self.previous_corners = np.array([self.corners], dtype=np.float32)
                            self.previous_corners = np.array(self.corners, dtype=np.float32)

                    # initial position of [0] corner will be the trigger when [1] corner
                    # hits this position.
                    # this means that the carriage has moved 1 inch.
                    # this should of course have correct gui implementation
                    
                    # print(f'previous_corners:\n {self.previous_corners}')
                    
                    # new_corners = cv2.selectROI(self.initial_frame) 

                    # previous_corners = np.array([[[new_corners[0], new_corners[1] ]]], dtype=np.float32)

                    # self.target_displacement
                    # self.delta_distance = 3 pixels
                    # TODO make self.delta_distance editable by the GUI
                    
                    self.target_displacement = self.previous_corners[1]
                    print(f'target_displacement: {self.target_displacement}')
                    self.delta_distance = 5
                    
                    self.p0 = self.previous_corners


                    # do not 'retrack' on same frame
                
                # end setup
                
                mask = np.zeros_like(self.initial_frame)

                # main loop
                
                current_frame_bn = img.frame_bgr_to_gray(frame)

                current_corners, found_status, error = cv2.calcOpticalFlowPyrLK(
                    self.previous_frame_bn, current_frame_bn, self.previous_corners, None,
                    **self.lucasKanadeParams)

                p1 = current_corners
        
                if current_corners is not None:
                    cornersMatchedCur = current_corners[found_status==1]
                    cornersMatchedPrev = self.previous_corners[found_status==1]

                if found_status is None:
                    found_status = []

                if len(found_status) > 0:
                    found_mask = found_status.astype(bool).flatten()
                    self.p0 = self.p0[found_mask]

                p1_found1 = cornersMatchedCur

                
                if self.p0 is not None:
                    self.data_exporter["results"]["shape_p0"] = self.p0.shape
                if p1 is not None:
                    self.data_exporter["results"]["shape_p1"] = p1.shape
                if p1_found1 is not None:
                    self.data_exporter["results"]["shape_p1_n,2"] = p1_found1.shape

                for i, (curCorner, prevCorner) in enumerate(zip(
                    cornersMatchedCur, cornersMatchedPrev)):
                    xCur, yCur = curCorner.ravel()
                    xPrev, yPrev = prevCorner.ravel()


#                    mask = cv2.line(mask, ( int(xCur), int(yCur) ),
#                    ( int(xPrev), int(yPrev) ), self.random_colors[i].tolist(), 2)
                    mask = mask

                    # show circles escalonated
                    frame = cv2.circle(frame, (int(xCur), int(yCur)), 5,
                        self.random_colors[i].tolist(), -1)
                    
                    frame = cv2.add(frame, mask)


                magnitude = 0 
                mean_dx = 0
                mean_dy = 0

                if found_status is None or len(found_status) == 0 or np.sum(found_status) == 0:
                    mean_dx = 0
                    mean_dy = 0
                else:
                    p1_filtered = p1_found1

                    p0_filtered = self.p0
                    
                    p1_squeezed = p1_filtered.reshape(-1, 2)
                    p0_squeezed = p0_filtered.reshape(-1, 2)
                    
                    mean_dx = np.mean(p1_squeezed[:, 0] - p0_squeezed[:, 0])
                    mean_dy = np.mean(p1_squeezed[:, 1] - p0_squeezed[:, 1])

                ## TODO make function that can adjust the way the displacement x is measured,
                ## TODO ADD cropbox at the start of OpticalTracker 
                ## TODO when points start to diverge more than 70% off the center, get new points
                ##      preferably in the center region.
                ## having points being there floating will deteriorate the overall displacement
                ## average
                average_motion_vector = (mean_dx, mean_dy)
                magnitude = np.linalg.norm(average_motion_vector)

                self.data_exporter["results"]["average_displacement_magnitude"] = magnitude
                self.data_exporter["results"]["average_displacement_x"] = mean_dx
                #print(f"displacement x is : {mean_dx}")
                self.data_exporter["results"]["average_displacement_y"] = mean_dy
                pp_tenth_of_an_inch = self.ppi / 10

                chars_moved = round_if_in_range(mean_dx/pp_tenth_of_an_inch, 0.09, 0.9)
                print(f"PPI {self.ppi} / 10 = {pp_tenth_of_an_inch}, chars_moved = {chars_moved}")

                self.data_exporter["results"]["average_displacement_x_rounded"] = round_if_in_range(mean_dx, 0.1, 0.9) 
                self.data_exporter["results"]["average_displacement_y_rounded"] = round_if_in_range(mean_dy, 0.1, 0.9) 
                self.data_exporter["results"]["chars_moved"] = chars_moved


                

                if( magnitude <= self.delta_distance):
                    self.data_exporter["current_status"] = "green"
                else:
                    self.data_exporter["current_status"] = "red"

                
                self.previous_frame_bn = current_frame_bn.copy()
                self.previous_corners = cornersMatchedCur.reshape(-1, 1, 2)

                #self.data_exporter["frames"]["result"] = frame.copy()
                #self.data_exporter["frames"]["result"] = current_frame_bn.copy()
                #self.data_exporter["frames"]["result"] = None
                 
                self.data_exporter["frames"]["result"] = frame.copy()

                #time.sleep(0.01)

                

            else:
                print(f"OpticalTracker() thread: video_stream not set. :(")
                time.sleep(0.02)
                # SELFFF

        self.exit_gracefully()
