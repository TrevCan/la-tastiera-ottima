import sys
import cv2
import time
import numpy as np

import VideoStream
from VideoStream import VideoStream
import OCR
from OCR import OCR
import ModelVocabulary

from InkTracker import InkTracker

import subprocess

from Singleton import Singleton

from OpticalTracker import OpticalTracker

from ImageTransformer import ImageTransformer

from gui import GUI

import dearpygui.dearpygui as dpg

from Data import Data

def isKeyDown(key):
    """
    isKeyDown(key)
    key must be format dpg.mvKey_.*
    """
    return dpg.is_key_down(key)

def safe_imshow(name, frame):
    if frame is not None and hasattr(frame, 'shape') and frame.shape != (0,) and frame.size > 0:
        cv2.imshow(name, frame)   

def get_variables_with_values():
    tmp = globals().copy()
    data = ""
    for k, v in tmp.items():
        if not k.startswith('_'):
            data += f"{k} : {v} type: {type(v).__name__}\n"
            ## TODO make __str__ function in all classes
            ## to print important variables!!!
    return data


d = Data()

        
def simulate_keystroke_wayland(character):
    """
    Simulates keystrokes on Wayland by calling the ydotool command-line utility.
    """
    # ydotool uses a specific keycode or character mapping
    # Note: This requires the 'ydotoold' daemon to be running.

    if d.debug["all"] or d.debug["simulate_keystroke_wayland"]:
        print(f'CHAR IS: {character}')

    character = ModelVocabulary.vocabulary_to_char[character]
    
    if len(character) == 1:
        # Example command to type a single character: ydotool type "H"
        command = ['ydotool', 'type', character]
    elif character == '\n':
        # Example command for Enter: ydotool key 28:1 28:0
        command = ['ydotool', 'key', '28:1', '28:0']
    else:
        print(f"Unsupported character for ydotool: {character}")
        return

    try:
        # Execute the command
        subprocess.run(command, check=True)
        if d.debug["all"] or d.debug["simulate_keystroke_wayland"]:
            print(f"Simulated keypress via ydotool: {character}")
    except subprocess.CalledProcessError as e:
        print(f"ydotool failed: {e}")
        print("Ensure the ydotool daemon (ydotoold) is running and accessible.")
        return

def rodent_handler(sender, app_data, user_data):
    d.mouse_x = app_data[0] + GUI().d.regionCreator.draw_offset_x
    d.mouse_y = app_data[1] + GUI().d.regionCreator.draw_offset_y
    
#    if event == cv2.EVENT_LBUTTONDOWN:
#        sx, sy, ex, ey = InkTracker().get_roi()
#        dx = ex - sx
#        dy = ey - sy
#        InkTracker().set_start_x(x)
#        InkTracker().set_start_y(y)
#        InkTracker().set_end_x(x + dx)
#        InkTracker().set_end_y(y + dy)
#
#    if event == cv2.EVENT_MBUTTONDOWN:
#        InkTracker().set_end_x(x)
#        InkTracker().set_end_y(y)
#
#    if d.optical_roi_initiate:
#        if len(d.optical_roi_points) <= d.optical_roi_max_points:
#            if event == cv2.EVENT_LBUTTONDOWN:
#
#                # this is for adding points as corners,
#                # NOT for setting a mask.
#                d.optical_roi_points.append([[x, y]])
#                if d.debug["all"] or d.debug["rodent_handler_optical_roi_initiate"]:
#                    print("Added current mouseX, mouseY to optical_roi_points")
#        else:
#            x, y, w, h = d.optical_data["crop_points"]
#
#            shape = (h, w)
#
#            for i, _ in enumerate(d.optical_roi_mask_points):
#                d.optical_roi_mask_points[i][0] -= x
#                d.optical_roi_mask_points[i][1] -= y
#
#            if d.debug["all"] or d.debug["rodent_handler_optical_roi_initiate"]:
#                print(f"Points are: {d.optical_roi_points} (Relative to crop box.)")
#            d.optical_roi_initiate = False
#
#            if d.debug["all"] or d.debug["rodent_handler_optical_roi_initiate"]:
#                print(f"ROI Mask creation for Optical Tracking.\n###END###")
#
#    if d.optical_roi_mask_initiate:
#        if len(d.optical_roi_mask_points) < d.optical_roi_mask_max_points:
#            if event == cv2.EVENT_LBUTTONDOWN:
#                # this is for setting a mask!
#                d.optical_roi_mask_points.append([x, y])
#
#        else:
#            if d.debug["all"] or d.debug["rodent_handler_optical_roi_mask_initiate"]:
#                print(f"Points are: {d.optical_roi_mask_points}")
#
#                print(f"ROI Mask creation for Optical Tracking.\n###END###")
#            d.optical_roi_mask_initiate = False


#           Key Detector           
# L_CLICK   Move ROI (ink) to mouse location.
# M_CLICK   Move ROI (ink) corner to mouse location.



def ocr_stream(source: int = 0):


    # already in line above somewhere in the void
    # maybe go watch True Detective S01
    ##d = Data()
    ink = InkTracker()
    
    # TODO make video_stream a singleton or make object available to rodent
    # handler
    video_stream = VideoStream(source)
    video_stream.start() # starts new thread dedicated to getting the frame.

    o_tracker = OpticalTracker()
    o_tracker.set_video_stream(video_stream)
    o_tracker.set_data_exporter(d.optical_data)
    o_tracker.pause()
    o_tracker.start()
    width, height = video_stream.get_video_dimensions()
    # default crop box should be nothing
    d.optical_data["crop_points"] = (0, 0, width, height)
    # don't actually run the thread, only initiate it
    # resume with o_tracker.resume()

    gui = GUI(width, height)

    ink.set_video_stream(video_stream)
    ink.set_roi(d.ink_roi["x"], d.ink_roi["y"], d.ink_roi["ex"], d.ink_roi["ey"])
    ink.set_data_exporter(d.ink_data)
    ink.start()
    ink.pause()

    ocr = OCR()
    ocr.set_data_exporter(d.ocr_data)
    ocr.set_video_stream(video_stream)
    ocr.set_default_dimensions(96, 164)
    ocr.set_crop_dimensions(d.ocr_roi["w"], d.ocr_roi["h"])
    ocr.set_crop_location(d.ocr_roi["x"], d.ocr_roi["y"])
    ocr.start_ai_model()
    ocr.start()
    ocr.pause()


    dims = ocr.get_default_dimensions()
    if d.debug["all"] or d.debug["ocr_stream_dimensions"]:
        print(f"default\n{dims}")
    dims = ocr.get_crop_dimensions()

    if d.debug["all"] or d.debug["ocr_stream_roi_or_crop_dimensions"]:
        print(f"crop\n{dims}")

    d.ocr_roi["w"] = dims[0] 
    d.ocr_roi["h"] = dims[1] 

    # Check if the camera stream is even available to proceed
    if not video_stream.stream.isOpened():
        sys.exit("FATAL ERROR: Video stream could not be opened. Exiting.")
        return # Exit the function
        
    cv2.namedWindow('stream', flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    #cv2.namedWindow('stream')
    #cv2.setMouseCallback('stream', rodent_handler)

    cv2.namedWindow('output', flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

    cv2.namedWindow('optical', flags=cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)


    def graceful_exit():
        print("\nUser requested exit.")
        gui.exit()
        ocr.exit()
        ink.exit()
        o_tracker.exit()
        video_stream.exit()
        
        cv2.destroyAllWindows()
        print("OCR stream stopped")
        #print(f"{captures} image(s) captured and saved to current directory")
        sys.exit(0)
    

    ocr.set_crop_dimensions(d.ocr_roi['w'], d.ocr_roi['h'])
    ocr.set_crop_location(d.ocr_roi['x'], d.ocr_roi['y'])

    with dpg.handler_registry():
        dpg.add_mouse_move_handler(
            callback=rodent_handler,
        )

    with dpg.texture_registry():
        z = np.zeros((700, 700, 4), dtype=np.float32 )
        dpg.add_raw_texture(tag="texture_cropped_optical_tag", width=700, height=700, format=dpg.mvFormat_Float_rgba, default_value=z)

    with dpg.window(label="CROP", tag="cropped_optical_window", pos=(800, 300)):
        dpg.draw_image(tag="draw_ttt", texture_tag="texture_cropped_optical_tag", pmin=[0,0], pmax=[700,700])
        #dpg.add_image(texture_tag="texture_cropped_optical_tag", width=300, height=300)


    while gui.loop_condition():

        
#           KEYSSSS
#
# v         turn ON/off Live OCR Keyboard
# c         Capture and save image from ROI (OCR)
# f         Test OCR on ROI. 

#           OCR
# w         Move ROI (OCR) up.
# s         Move ROI (OCR) down.
# a         Move ROI (OCR) left.
# d         Move ROI (OCR) right. 
# SPACE ' ' Move ROI (OCR) to mouse location.
# 9         shrink ROI  (OCR)
# 0         enlarge ROI (OCR)

# h         set ROI (OCR) hsv ranges (dirty).

#           ALIGN
# t         set ROI (ALIGN) with mousex, mousey(align)
# o         set offset based on ROI (OCR)
# i         turn ON/off tracking ROI (align)



#           Key Detector           
# L_CLICK   Move ROI (ink) to mouse location.
# M_CLICK   Move ROI (ink) corner to mouse location.



# q         Exit


        #if pressed_key == ord('q'):
        if isKeyDown(dpg.mvKey_Q):
            graceful_exit()

#        # wasd control ROI COR
#        if pressed_key == ord('w'):
#            d.ocr_roi["y"] -= 1
#        if pressed_key == ord('a'):
#            d.ocr_roi["x"] -= 1
#        if pressed_key == ord('s'):
#            d.ocr_roi["y"] += 1
#        if pressed_key == ord('d'):
#            d.ocr_roi["x"] += 1

        # wasd control ROI COR
        if isKeyDown(dpg.mvKey_W):
            d.ocr_roi["y"] -= 1
        if isKeyDown(dpg.mvKey_A):
            d.ocr_roi["x"] -= 1
        if isKeyDown(dpg.mvKey_S):
            d.ocr_roi["y"] += 1
        if isKeyDown(dpg.mvKey_D):
            d.ocr_roi["x"] += 1



        #if pressed_key == ord(' '):
        if isKeyDown(dpg.mvKey_Spacebar):
            d.ocr_roi['x'] = round(d.mouse_x - d.ocr_roi['w']/2)
            d.ocr_roi['y'] = round(d.mouse_y - d.ocr_roi['h']/2)

        #if pressed_key == ord('9'):
        if isKeyDown(dpg.mvKey_9):
            f = 1
            d.ocr_roi['amplification'] -= 0.02

            ocr.set_crop_width(round(ocr.default_width * d.ocr_roi['amplification']))
            ocr.set_crop_height(round(ocr.default_height * d.ocr_roi['amplification']))
            deltax = ocr.crop_width - d.ocr_roi['w']
            deltay = ocr.crop_height - d.ocr_roi['h']

            d.ocr_roi['w'] = ocr.crop_width
            d.ocr_roi['h'] = ocr.crop_height
            
            d.ocr_roi['x'] -= round(f * deltax/2)
            d.ocr_roi['y'] -= round(f * deltay/2)

        #if pressed_key == ord('0'):
        if isKeyDown(dpg.mvKey_0):
            f = 1
            d.ocr_roi['amplification'] += 0.02

            ocr.set_crop_width(round(ocr.default_width * d.ocr_roi['amplification']))
            ocr.set_crop_height(round(ocr.default_height * d.ocr_roi['amplification']))
            deltax = ocr.crop_width - d.ocr_roi['w']
            deltay = ocr.crop_height - d.ocr_roi['h']

            d.ocr_roi['w'] = ocr.crop_width
            d.ocr_roi['h'] = ocr.crop_height
            
            d.ocr_roi['x'] -= round(f * deltax/2)
            d.ocr_roi['y'] -= round(f * deltay/2)
            
            ocr.set_crop_location(d.ocr_roi['x'], d.ocr_roi['y'])
            print(f'ocr {ocr.crop_width}, {ocr.crop_height}')
            print(f'0wh {d.ocr_roi['w']}, {d.ocr_roi['h']}')
            print(f'0xy {d.ocr_roi['x']}, {d.ocr_roi['y']}')


       


#        if pressed_key == ord('1'):
#            simulate_keystroke_wayland('q')

        #if pressed_key == ord('p'):
        if isKeyDown(dpg.mvKey_P):
            if ocr.is_running():
                ocr.pause()
                print(f"OCR ||")
            elif not ocr.is_running():
                ocr.resume()
                print(f"OCR >")
            else:
                print('ERROR: Unable to resume or pause OCR', file=sys.stderr)



        #if pressed_key == ord('v'):
        if isKeyDown(dpg.mvKey_V):
            d.enable_model = not d.enable_model
            print(f"OCR Model: {d.enable_model}")

        out_frame = None

        if d.enable_model:
            ink.resume()
            #if d.ink_data["previous_state"] and (not d.ink_data["current_state"]):
            if not d.ink_data["previous_state"] and d.ink_data["current_state"]:
                ocr.resume()
                time.sleep(0.25)
                ocr.pause()
                character, confidence = ocr.get_latest_prediction()
                out_frame = d.ocr_data["filtered_frames"]["0"].copy()
                print(f'{character} with {confidence} confidence.')
                ocr.resume()
                simulate_keystroke_wayland(character)
        else:
            ocr.pause()
            ink.pause()


        #if pressed_key == ord('f'):
        if isKeyDown(dpg.mvKey_F):
            x, y = ocr.set_crop_location(d.ocr_roi['x'], d.ocr_roi['y'])
            w, h = ocr.get_crop_dimensions()
            if not ocr.is_running():
                ocr.resume()
            time.sleep(0.01)
            ocr.pause()
            character, confidence = ocr.get_latest_prediction()
            #character, confidence = ocr.predict_single_character(out_frame)
            print(f'Char is {character} with {confidence} confidence.')
#            cv2.displayStatusBar('output', f'crop ({x}, {y}): ({w}, {h})', 0)

#            cv2.imshow("f0", d.ocr_data["filtered_frames"]["0"])
            # TODO add window for filtered frame in dearpygui

            dpg.set_value("log_text", f'Char is {character} with {confidence} confidence.')

        #if pressed_key == ord('x'):
        if isKeyDown(dpg.mvKey_X):
            d.ocr_roi['x'] = round(d.mouse_x - d.ocr_roi['w']/2)
            d.ocr_roi['y'] = round(d.mouse_y - d.ocr_roi['h']/2)

            x, y = ocr.set_crop_location(d.ocr_roi['x'], d.ocr_roi['y'])
            w, h = ocr.get_crop_dimensions()
            d.ocr_roi["w"] = w
            d.ocr_roi["h"] = h
            #print(f'D: {w}, {h}')

            cv2.displayStatusBar('output', f'crop ({x}, {y}): ({w}, {h})', 0)

        #if pressed_key == ord('k'):
        if isKeyDown(dpg.mvKey_K):
            file_name = f"data_{time.ctime(time.time())}.txt"
            with open(file_name, "w") as file:
                file.write(get_variables_with_values())
            print(f"Wrote variables to {file_name}")

        #if pressed_key == ord('n'):
        if isKeyDown(dpg.mvKey_N):
            #d.optical_roi_mask = None
            d.optical_roi_initiate = True
            d.optical_roi_points = []
            d.optical_roi_max_points = 4

        #if pressed_key == ord('m'):
        if isKeyDown(dpg.mvKey_M):
            d.optical_roi_mask = None
            d.optical_roi_mask_initiate = True
            d.optical_roi_mask_points = []
            d.optical_roi_mask_max_points = 4


#         
#        self.optical_roi_mask = None
#        self.optical_roi_mask_initiate = False
#        self.optical_roi_mask_points = []
#        self.optical_roi_mask_max_points = 4
#
            
        optical_tracker_frame = None

        #if pressed_key == ord('u'):
        if isKeyDown(dpg.mvKey_U):
            #print(f"VS dimensions: {video_stream.get_video_dimensions()}")
            # we only want a binary image, so only receive the first two elements.


            o_tracker.set_corners(d.optical_roi_points)

            x, y, w, h = d.optical_data["crop_points"]

            shape = (h, w)

            d.optical_roi_mask = ImageTransformer.get_binary_mask_polygon(
                shape,
                d.optical_roi_mask_points )
            if d.optical_roi_mask is not None:
                cv2.imshow('mask', d.optical_roi_mask)
                # TODO maybe remove this....

            o_tracker.set_mask(d.optical_roi_mask)

            o_tracker.set_mask(d.optical_roi_mask)

            if not o_tracker.is_running():
                o_tracker.resume()
                print(f">\tRESUME Optical Tracker")
                time.sleep(0.01)
            else:
                print(f"||\tPAUSE Optical Tracker")
                o_tracker.pause()

        #if pressed_key==ord('l'):
        if isKeyDown(dpg.mvKey_L):
            was_running = False
            if o_tracker.is_running():
                o_tracker.pause()
                was_running = True

            print('start Le Cropbox Calibration')
            x, y, w, h = cv2.selectROI("Select Cropbox Optical Tracking", video_stream.frame.copy(), showCrosshair=True, fromCenter=False)
            d.optical_data["crop_points"] = (x, y, x+w, y+h)
            cv2.destroyWindow("Select Cropbox Optical Tracking")

            print("Start Le Inch Calibration")
            frame = video_stream.frame.copy()

            x1, y1, x2, y2 = d.optical_data["crop_points"]

            frame = ImageTransformer.frame_crop_frame(frame, x1, y1, x2, y2)

            x, y, w, h = cv2.selectROI("select_inch", frame, showCrosshair=False, fromCenter=False)
            d.optical_data["user_parameters"]["pixels_per_inch"]=w
            print(f'PPI: {d.optical_data["user_parameters"]["pixels_per_inch"]}')
            cv2.destroyWindow("select_inch")

            o_tracker.create_new_corners()

            if was_running:
                o_tracker.resume()

#         
#        self.optical_roi_mask = None
#        self.optical_roi_mask_initiate = False
#        self.optical_roi_mask_points = []
#        self.optical_roi_mask_max_points = 4
#




        if o_tracker.is_running():
            optical_tracker_frame = d.optical_data["frames"]["result"]
            
#
#        d_mag = d.optical_data["results"]["average_displacement_magnitude"]
#        d_x = d.optical_data["results"]["average_displacement_y"]
#        d_y = d.optical_data["results"]["average_displacement_x"]
#
#        cv2.displayStatusBar('optical', f'MAG {d_mag}. DISP X: {d_x}\tDISP Y: {d_y}', 0)
        cv2.displayStatusBar("optical", f'charsM: {d.optical_data["results"]["chars_moved"]}')

        #cv2.displayStatusBar('output', f'crop ({x}, {y}): ({w}, {h})', 0)
#                 "results": {
#                "average_displacement_magnitude": 0,
#                "average_displacement_x": 0,
#                "average_displacement_y": 0
#            },           

        #out_frame = ocr.transformed_frame
        if out_frame is not None:
            cv2.imshow("output", out_frame)
            # TODO maybe remove this too.
            

        if video_stream.frame is None:
            # If the background thread hasn't provided a frame yet, wait briefly and try again
            time.sleep(0.01)
            continue

#        cv2.imshow("status optical", d.status_frames[d.optical_data["current_status"]])
        # TODO add this to info window, maybe as colored text.
        
        frame = video_stream.frame.copy()  # Grabs the most recent frame read by the VideoStream class


        if optical_tracker_frame is not None:
            f = ImageTransformer.frame_bgr_to_rgba(frame)
            f_centered = ImageTransformer.center_image_in_texture(f, target_width=width, target_height=height)  # Assuming 200x200 texture
            gui.update_optical_stream_frame(f_centered)
    ## TODO make this work!!! add button to enable optical tracking.

#            cv2.imshow("optical", optical_tracker_frame) 
        #safe_imshow("optical", optical_tracker_frame)
        # TODO add optical window


        d.ocr_roi["ex"] = d.ocr_roi["x"] + d.ocr_roi["w"]
        d.ocr_roi["ey"] = d.ocr_roi["y"] + d.ocr_roi["h"]


        gui.update_raw_stream_frame(frame)

        cv2.rectangle(frame, (d.ocr_roi["x"], d.ocr_roi["y"]), ( d.ocr_roi["ex"], d.ocr_roi["ey"]), (0, 255, 0), thickness=2)
        sx, sy, ex, ey = ink.get_roi()

        cv2.rectangle(frame, (sx, sy), (ex, ey), (0, 0, 255), thickness=2)

        gui.update_roi_stream_frame(frame)

        gui.loop()

    graceful_exit()



if __name__ == "__main__":
    # If using a specific camera index (e.g., /dev/videoN), change '0' to '1' or the correct index.
    # The default 0 is usually the first webcam.
    #ocr_stream(source=0, _ink_roi_x=594, _ink_roi_y=285, _ink_roi_end_x=611, _ink_roi_end_y=293 ) 
    ocr_stream(source="/dev/video1")