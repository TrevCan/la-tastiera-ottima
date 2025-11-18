import sys
from typing import Optional

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from ImageTransformer import ImageTransformer
from Datum import Datum
from OpticalTracker import OpticalTracker
from Singleton import Singleton
from VideoStream import VideoStream
from Data import Data


class GUI(metaclass=Singleton):

    def __init__(self, width, height):
        self.cropped_frame = None
        self.frame = None
        self.width_stream = width
        self.height_stream = height
        self.d = Datum()

        self.DEBUG_GUI = False

        dpg.create_context()

        with dpg.font_registry():
            #default_font = dpg.add_font( "fonts/JetBrainsMono/fonts/ttf/JetBrainsMono-Regular.ttf", 31)
            default_font = dpg.add_font( "fonts/JetBrainsMono/fonts/ttf/JetBrainsMono-Regular.ttf", 13)

        dpg.bind_font(default_font)

        dpg.create_viewport(title="live-ocr", width=1920, height=1000)

        self.init_video(width, height)


        with dpg.window(**self.d.regionCreator.window_parameters) as self.draw_frame:
            dpg.set_item_pos(self.draw_frame, [0, 0])


            with dpg.drawlist(**self.d.regionCreator.draw_list_parameters) as draw_list:
                dpg.set_item_pos(draw_list, [0, 0])
                # IMPORTANT! Use draw_image to draw images inside a drawlist,
                # this way, the image will be in the same position as
                # the subsequent layers inside the same drawlist.
                #dpg.add_image("texture_tag2")
                dpg.draw_image(self.d.tag_stream_texture, [0, 0], [width, height])
                for i in range(len(self.d.regionCreator.layers)):
                    with dpg.draw_layer(tag=self.d.regionCreator.layers[i]["tag"]) as tag_draw_layer:
                        dpg.draw_rectangle([0, 0], [0, 0], color=(255, 255, 255, 255),
                                           tag=f"{tag_draw_layer}_rectangle")

                        # dpg.add_text(default_value=d.regionCreator.layers[i]["label"])
        #        with dpg.draw_layer(tag="draw_layer") as draw_layer:
        #            dpg.draw_circle(center=[50, 50], radius=10, color=[255,255,255])


        # with dpg.theme() as theme:
        #    with dpg.theme_component(dpg.mvAll):
        #        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0)
        # dpg.bind_item_theme("w1", theme)


        with dpg.window(label="info", width=700, height=500, no_collapse=True) as window_tag:
            dpg.set_item_pos(window_tag, [0, 600])
            dpg.add_tab_bar(tag="bartag")
            dpg.add_tab_button(label="iuse;", parent="bartag")
            dpg.add_text("tuning")

            dpg.add_text(tag="mouse_position_draw", label="Draw Mouse x y", show_label=True)
            dpg.add_text(tag="mouse_position_drag", label="Drag Mouse x y", show_label=True)

            with dpg.group(horizontal=True):
                items = list(map(lambda x: x["label"], self.d.regionCreator.layers))
                dpg.add_radio_button(items=items, callback=self.layer_use, tag="layer_radio_button", default_value=items[2])

                dpg.add_button(label="Send Crop Layer", tag="button_send_crop_layer", callback=self.button_send_crop_layer,
                               user_data={"layer_index": 2,
                                          "texture": "texture_cropped_optical_tag" })
                dpg.add_button(label="Send Optical ROI Mask", tag="button_send_optical_roi_mask", callback=self.button_send_optical_roi_mask,
                               user_data={"layer_index": 1,
                                      "texture": "texture_cropped_optical_tag"})
            dpg.add_spacer()
            for layer in self.d.regionCreator.layers:
                dpg.add_checkbox(label=f"Show {layer["label"]}", tag=layer["tag_checkbox_show"], default_value=True,
                                 callback=self.layer_show, user_data=layer["tag"])
            dpg.add_spacer(height=80)
            dpg.add_checkbox(label="Show ROI", default_value=True, callback=self.toggle_roi)
            dpg.add_text("Offset Draw")
            with dpg.group(horizontal=True):
                # default padding is like 8 pixels, I tried modifying it to no avail.
                # should you find a better solution, let me know!
                dpg.add_input_int(tag="offset_x", label="⟷ x axis", callback=self.set_offset_draw, width=115,
                                  default_value=-8)
                # trigger callback function
                self.set_offset_draw("offset_x", dpg.get_value("offset_x"), [])
                dpg.add_spacer(width=10, height=5)
                dpg.add_input_int(tag="offset_y", label="↕ Y axis", callback=self.set_offset_draw, width=115,
                                  default_value=8)
                self.set_offset_draw("offset_y", dpg.get_value("offset_y"), [])

            # dpg.add_input_text(default_value="hhhh", tag="input_text1", wrap=100 )
            dpg.add_text(label="LOG", show_label=True, default_value="", wrap=100, tag="log_text")

            # dpg.add_color_picker(tag="color_picker")

            # dpg.add_input_text(hint="candy wrapper", multiline=True, wrap=0)


        dpg.setup_dearpygui()

#        dpg.show_style_editor()
#        dpg.show_item_registry()

        dpg.show_viewport()

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(callback=self.dragging_mouse)
            dpg.add_mouse_click_handler(callback=self.click_mouse)

        with dpg.value_registry():
            dpg.add_int_value(default_value=0, tag="mouse_click_x")
            dpg.add_int_value(default_value=0, tag="mouse_click_y")


    def init_video(self, width, height):
        with dpg.texture_registry(show=False):
            zeros = np.zeros((width, height, 4), dtype=np.float32 )
            zeros = zeros.ravel()
            zeros = np.asarray(zeros, dtype=np.float32)
            dpg.add_raw_texture(width=width, height=height, tag=self.d.tag_stream_texture,
                default_value=zeros, format=dpg.mvFormat_Float_rgba)
            dpg.add_raw_texture(width=width, height=height, tag=self.d.tag_mask_texture,
                default_value=zeros, format=dpg.mvFormat_Float_rgba)
            dpg.add_raw_texture(width=width, height=height, tag=self.d.tag_roi_w_stream_texture,
                default_value=zeros, format=dpg.mvFormat_Float_rgba)
            dpg.add_raw_texture(width=width, height=height, tag=self.d.tag_optical_texture,
                                default_value=zeros, format=dpg.mvFormat_Float_rgba)

        with dpg.window(label="raw stream") as raw_stream_window:
            dpg.set_item_pos(raw_stream_window, [720, 0])
            #dpg.add_text("Hello, world1:")
            with dpg.drawlist(width, height):
                with dpg.draw_layer():
                    dpg.draw_image(self.d.tag_stream_texture, pmin=[0,0], pmax=[width, height])
                with dpg.draw_layer(tag=f"layer_{self.d.tag_roi_w_stream_texture}"):
                    dpg.draw_image(self.d.tag_roi_w_stream_texture, pmin=[0,0], pmax=[width,height])
                with dpg.draw_layer(tag=f"layer_{self.d.tag_optical_texture}", show=False):
                    dpg.draw_image(self.d.tag_optical_texture, pmin=[0,0], pmax=[width, height])
            dpg.add_slider_doublex()

        with dpg.window(label="mask") as mask_window:
            dpg.set_item_pos(mask_window, [720, height + 15])
            #dpg.add_text("Hello, world1:")
            dpg.add_image(self.d.tag_mask_texture)
            dpg.add_slider_doublex()

    def toggle_roi(self, sender, app_data, user_data):
        print('aaaaa')
        if app_data:
            dpg.show_item(f"layer_{self.d.tag_roi_w_stream_texture}")
        else:
            dpg.hide_item(f"layer_{self.d.tag_roi_w_stream_texture}")


    def button_send_crop_layer(self, sender, app_data, user_data):
        rectangle = self.rectangle(layer_index=user_data["layer_index"])
        # TODO find startx, starty, endx, endy through mins and maxs
        # right now the code below ONLY works when the rectangle is
        # created starting in the upper left corner and ending in the lower right corner
        startx, starty = rectangle["pmin"][0], rectangle["pmin"][1]
        endx, endy = rectangle["pmax"][0], rectangle["pmax"][1]
        startx = int(startx) + self.d.regionCreator.draw_offset_x
        starty = int(starty) + self.d.regionCreator.draw_offset_y
        endx = int(endx) + self.d.regionCreator.draw_offset_x
        endy = int(endy) + self.d.regionCreator.draw_offset_y
        frame = self.frame.copy()
        print()
        print(startx)
        print(starty)
        print(endx)
        print(endy)
        print()
        cropped_frame = ImageTransformer.frame_crop_frame(frame, startx, starty, endx, endy)
        self.cropped_frame = cropped_frame.copy()
        height, width, _ = cropped_frame.shape
        # Thanks for using Singletons!
        Data().optical_data["crop_points"] = startx, starty, endx, endy

        dpg.delete_item("texture_cropped_optical_tag")
        dpg.delete_item("draw_ttt")
        with dpg.texture_registry():
            #z = np.zeros((height, width, 4), dtype=np.float32)
            z = cropped_frame
            z = ImageTransformer.frame_bgr_to_rgba_normalized(z)
            dpg.add_raw_texture(tag="texture_cropped_optical_tag", width=width, height=height,
                                format=dpg.mvFormat_Float_rgba, default_value=z)

        dpg.configure_item("cropped_optical_window", width=width+10, height=height+10)
        dpg.draw_image(parent="cropped_optical_window" ,tag="draw_ttt", texture_tag="texture_cropped_optical_tag", pmin=[0, 0], pmax=[width, height])
        # dpg.add_image(texture_tag="texture_cropped_optical_tag", width=300, height=300)

#        f = ImageTransformer.frame_resize(frame, (1000, 1000))
#        f = ImageTransformer.frame_bgr_to_rgba_normalized(f)
#        dpg.configure_item("draw_ttt", pmax=[1000, 1000])
#        dpg.set_value("texture_cropped_optical_tag", f)


#        resized = ImageTransformer.frame_resize(self.frame, (self.width_stream, self.height_stream) )
#        rgba = ImageTransformer.frame_bgr_to_rgba_normalized(resized)

        #dpg.configure_item(item="cropped_optical_window", width=self.height_stream, height=self.width_stream)
#        dpg.set_item_width(item="cropped_optical_window", width=self.width_stream)
#        dpg.set_item_height(item="cropped_optical_window", height=self.height_stream)
        #dpg.set_item_width("texture_cropped_optical_tag", resized.shape[1])
#        dpg.set_item_width("texture_cropped_optical_tag", 300)
#        dpg.set_item_height("texture_cropped_optical_tag", 300)
        #dpg.set_item_height("texture_cropped_optical_tag", resized.shape[0])
        ##################################
#
#        dpg.delete_item("texture_cropped_optical_tag")
#        dpg.delete_item("draw_ttt")
#        with dpg.texture_registry():
#            z = np.zeros((1000, 1000, 4), dtype=np.float32)
#            dpg.add_raw_texture(tag="texture_cropped_optical_tag", width=1000, height=1000,
#                                format=dpg.mvFormat_Float_rgba, default_value=z)
#
#        dpg.draw_image(parent="cropped_optical_window" ,tag="draw_ttt", texture_tag="texture_cropped_optical_tag", pmin=[0, 0], pmax=[1000, 1000])
#
#        f = ImageTransformer.frame_resize(frame, (1000, 1000))
#        f = ImageTransformer.frame_bgr_to_rgba_normalized(f)
#        dpg.configure_item("draw_ttt", pmax=[1000, 1000])
#        dpg.set_value("texture_cropped_optical_tag", f)
#
    ##############
#        if True:
#            y, x, layers = cropped_frame.shape
#            print(f"x {x}, y {y}")
#            dpg.configure_item(item="cropped_optical_window", width=x, height=y)
#            dpg.configure_item(item="cropped_optical_tag", texture_tag="texture_cropped_optical_tag",
#                          width=x,
#                          height=y)
#            one_d_frame = ImageTransformer.frame_bgr_to_rgba_normalized(cropped_frame)
#            #one_d_frame = ImageTransformer.frame_gray_to_rgba_normalized(cropped_frame)
#            dpg.set_value("texture_cropped_optical_tag", one_d_frame)
#            #texture = dpg.get_value("texture_cropped_optical_tag")
#            #print(f"shape is {texture.shape}")

    def button_send_optical_roi_mask(self, sender, app_data, user_data):
        points_main_crop = self.rectangle_points(layer_index=2)
        offset_x = points_main_crop[0][0]
        offset_y = points_main_crop[0][1]
        points = self.rectangle_points(layer_index=user_data["layer_index"])
        for point in points:
            point[0] = point[0] - offset_x
            point[1] = point[1] - offset_y

        print(points)

        # TODO add verification for existence of self.cropped_frame
        height, width, _ = self.cropped_frame.shape

        the_mask = ImageTransformer.get_binary_mask_polygon((height, width), points)
        OpticalTracker().set_mask(the_mask)


        dpg.delete_item("texture_cropped_optical_tag")
        dpg.delete_item("draw_ttt")
        with dpg.texture_registry():
            #z = np.zeros((height, width, 4), dtype=np.float32)
            z = self.cropped_frame
            m = cv2.cvtColor(the_mask, cv2.COLOR_GRAY2BGR)
            z = cv2.add(z, m)
            z = ImageTransformer.frame_bgr_to_rgba_normalized(z)
            dpg.add_raw_texture(tag="texture_cropped_optical_tag", width=width, height=height,
                                format=dpg.mvFormat_Float_rgba, default_value=z)

        dpg.configure_item("cropped_optical_window", width=width+10, height=height+10)
        dpg.draw_image(parent="cropped_optical_window" ,tag="draw_ttt", texture_tag="texture_cropped_optical_tag", pmin=[0, 0], pmax=[width, height])

#        the_mask = ImageTransformer.frame_gray_to_rgba_normalized(the_mask)



    def layer_show(self, sender, app_data, user_data):
        # user_data is the specific layer name for each checkbox name
        # it is automatically set in the checkbox generation, since the
        # checkbox tag and the layer tag correspond to the same index in the
        # layers array.
        if not app_data:
            dpg.hide_item(user_data)
        else:
            dpg.show_item(user_data)

    def layer_use(self, sender, app_data, user_data):
        index = int(dpg.get_item_configuration(sender)["items"].index(app_data))
        self.d.regionCreator.current_tag_layer = self.d.regionCreator.layers[index]["tag"]
        self.d.regionCreator.current_layer_index = index

        # in theory only the last parameter is used by the function but the other
        # parameters are specified in order to track down where it comes from
        # should one be debugging.
        # this makes the layer actually be shown.
        self.layer_show(sender, app_data, self.d.regionCreator.current_tag_layer)

        # this sets the checkbox value and makes it visible in the GUI
        dpg.set_value(self.d.regionCreator.layers[index]["tag_checkbox_show"], True)

        self.send_layer_to_mask()



    def set_offset_draw(self, sender, app_data, user_data):
        if self.DEBUG_GUI:
            print(f"int input callback: set_offset_draw()")
            print(f"\tsender is {sender}")
            print(f"\tdata is {app_data}")
        if sender == self.d.regionCreator.tag_draw_value_offset_x:
            self.d.regionCreator.draw_offset_x = app_data
        elif sender == self.d.regionCreator.tag_draw_value_offset_y:
            self.d.regionCreator.draw_offset_y = app_data


    def isInFrame(self,x, y, frame_window):
        frame_x, frame_y = dpg.get_item_pos(frame_window)
        frame_end_x = frame_x + dpg.get_item_width(frame_window)
        frame_end_y = frame_y + dpg.get_item_height(frame_window)

        if self.DEBUG_GUI:
            print("isInframe(x, y, frame_window)")
            print(f"\t{frame_x},{frame_y} : {frame_end_x},{frame_end_y}")

        return frame_x <= x <= frame_end_x and frame_y <= y <= frame_end_y


    def dragging_mouse(self, sender, app_data):

        draw_layer = self.d.regionCreator.layers[0]["tag"]
        draw_layer = self.d.regionCreator.current_tag_layer

    #    dpg.get_value("mouse_position_drag")
        #dpg.draw_circle(center=[50, 50], radius=10, color=[255, 255, 255])
        x, y = dpg.get_value("mouse_click_x"), dpg.get_value("mouse_click_y")
        ex = x + app_data[1]
        ey = y + app_data[2]
        if self.DEBUG_GUI:
            print(f"dragging_mouse(sender, app_data)")
            print(f"\tEP: {ex}, {ey}")
    #    is_draw_window = dpg.get_item_label(dpg.get_active_window()) is d.regionCreator.tag_window

        window=dpg.get_active_window()
    #    print(dpg.get_item_label(d.regionCreator.window_parameters["label"]))
    #    print(f"WWW {dpg.get_item_label(draw_frame)}")
    #    print(f"draw frame {draw_frame} window {window}")
    #    print(f"isdrawin {is_draw_window}")
    #    print(f"isdrawin {d.regionCreator.tag_window}")
    #    print(f"isdrawin {dpg.get_item_label(d.regionCreator.tag_window)}")
    # dpg.get_item_alias(integer_id) integer id TAG  to tag ALIAS
    # dpg.get_alias_id("string_alias") tag ALIAS to tag ID integer
        is_draw_window = dpg.get_alias_id(self.d.regionCreator.tag_window) is dpg.get_active_window()
        #print(f"{dpg.get_alias_id(d.regionCreator.tag_window)}, {dpg.get_active_window()}")
    #    is_draw_window = dpg.get_item_label(dpg.get_active_window()) == dpg.get_item_label(d.regionCreator.tag_window)

        if self.isInFrame(x, y, self.draw_frame) and self.isInFrame(ex, ey, self.draw_frame) and is_draw_window:
            # clears frame
            # don't delete draw_layer... for now
    #        if dpg.does_item_exist(draw_layer):
    #            dpg.delete_item(draw_layer)
    #            #draw_layer = dpg.add_draw_layer(parent=draw_list, tag=draw_layer)
    #            dpg.add_draw_layer(parent=draw_list, tag=draw_layer)
            x = x + self.d.regionCreator.draw_offset_x
            y = y + self.d.regionCreator.draw_offset_y

            ex = ex + self.d.regionCreator.draw_offset_x
            ey = ey + self.d.regionCreator.draw_offset_y

            if self.DEBUG_GUI:
                print(f"dragging_mouse(sender, app_data)")
                print(f"offset_x  is {self.d.regionCreator.draw_offset_x}")
                print(f"offset_y  is {self.d.regionCreator.draw_offset_y}")

            #dpg.draw_rectangle([x, y], [ex, ey], color=(255, 255, 255, 255), parent=draw_layer)
            dpg.configure_item(f"{draw_layer}_rectangle", pmin=[x, y], pmax=[ex, ey])

            self.send_layer_to_mask()

        dpg.set_value("log_text",
                      (f"\n"
                       f"    ######################\n"
                       f"    sender {sender}\n"
                       f"    app_data {app_data}\n"
                       f"    children {len(dpg.get_item_children(draw_layer))}\n") )

        #for child in dpg.get_item_children("draw_layer", slot=dpg.get_item_slot(draw_layer)):
    #    for child in dpg.get_item_children(draw_layer, slot=dpg.get_item_slot(draw_layer)):
    #        type_child = dpg.get_item_type(child)
    #        #value = dpg.get_value(child)
    #        #print(f"T: {type_child} L:  V: ")


    def click_mouse(self, sender, app_data):
        x, y = dpg.get_mouse_pos()
        if not dpg.is_mouse_button_dragging(button=dpg.mvMouseButton_Left, threshold=0.01):

            dpg.set_value("mouse_click_x", x)
            dpg.set_value("mouse_click_y", y)

    #    dpg.set_value("mouse_click_x", x)
    #    dpg.set_value("mouse_click_y", y)

    def rectangle(self, layer_index):
        tag_draw_layer = self.d.regionCreator.layers[layer_index]["tag"]
        tag_rectangle = f"{tag_draw_layer}_rectangle"
        #return dpg.get_value(tag_rectangle)
        return dpg.get_item_configuration(tag_rectangle)

    def rectangle_position(self, layer_index):
        rectangle = self.rectangle(layer_index=layer_index)
        return rectangle["pmin"], rectangle["pmax"]

    def rectangle_points(self, layer_index):
        rectangle = self.rectangle(layer_index=layer_index)
        p1 = [int(rectangle["pmin"][0]), int(rectangle["pmin"][1]) ]
        p3 = [int(rectangle["pmax"][0]), int(rectangle["pmax"][1]) ]
        rectangle_width = int(p3[0]) - int(p1[0])
        rectangle_height = p3[1] - p1[1]
        p2 = [ p1[0] + rectangle_width, p1[1]]
        p4 = [ p1[0], p1[1] + rectangle_height]

#        print(self.rectangle(layer_index=layer_index))
#        print(p1)
#        print(p2)
#        print(p3)
#        print(p4)

        return [p1, p2, p3, p4]

    def loop_condition(self):
        return dpg.internal_dpg.is_dearpygui_running()


    def run_loop(self):
        while dpg.internal_dpg.is_dearpygui_running():
            self.loop()

    def loop(self):
        mouse_position_dragged = dpg.get_mouse_drag_delta()
        mouse_position = dpg.get_drawing_mouse_pos()

        dpg.set_value("mouse_position_drag", mouse_position_dragged)
        dpg.set_value("mouse_position_draw", mouse_position)

        # dpg.mvMouseDragHandler(dragging_mouse)
        # print(mousepos)

        # print(dpg.get_item_configuration(d.regionCreator.tag_window))

        dpg.internal_dpg.render_dearpygui_frame()

    def exit(self):
        dpg.destroy_context()

    def send_layer_to_mask(self):
        points = self.rectangle_points(self.d.regionCreator.current_layer_index)

        the_mask = ImageTransformer.get_binary_mask_polygon((self.height_stream, self.width_stream), points)
        the_mask = ImageTransformer.frame_gray_to_rgba_normalized(the_mask)

        dpg.set_value(self.d.tag_mask_texture, the_mask)

        return the_mask, points

    @staticmethod
    def is_key_down_down_arrow(self):
        return dpg.is_key_down(dpg.mvKey_Down)

    @staticmethod
    def is_key_down_up_arrow():
        return dpg.is_key_down(dpg.mvKeyUp)

    @staticmethod
    def is_key_down_right_arrow():
        return dpg.is_key_down(dpg.mvKey_Right)

    @staticmethod
    def is_key_down_left_arrow():
        return dpg.is_key_down(dpg.mvKey_Left)

    @staticmethod
    def is_key_down_0():
        return dpg.is_key_down(dpg.mvKey_0)

    @staticmethod
    def is_key_down_9():
        return dpg.is_key_down(dpg.mvKey_9)

    def update_raw_stream_frame(self, frame):
        if frame is not None:
            self.frame = frame.copy()
            frame = ImageTransformer.frame_bgr_to_rgba_normalized(frame)
            dpg.set_value(self.d.tag_stream_texture, frame)
            return True
        return False

    def update_roi_stream_frame(self, frame):
        if frame is not None:
            f = frame.copy()
            f = ImageTransformer.frame_bgr_to_rgba_normalized(f)
            dpg.set_value(self.d.tag_roi_w_stream_texture, f)

    def update_optical_stream_frame(self, optical_tracker_frame):
        if frame is not None:
            optical_tracker_frame = optical_tracker_frame.copy()
            optical_tracker_frame = ImageTransformer.frame_bgr_to_rgba_normalized(optical_tracker_frame)
            dpg.set_value(self.d.tag_optical_texture, optical_tracker_frame)


if __name__ == '__main__':

    video = VideoStream("/dev/video1")
    video.start()
    width, height = video.get_video_dimensions()
    g = None
    if width==0 or height==0:
        sys.exit(f"ERROR: unable to get width or height. {width}, {height}")
    g = GUI(width, height)

    #print(g.rectangle(0) )
    while g.loop_condition():
        frame = video.frame
        if frame is not None:
            gui_frame = ImageTransformer.frame_bgr_to_rgba_normalized(frame)
            dpg.set_value(g.d.tag_stream_texture, gui_frame)
        g.loop()
    g.exit()
    video.exit()


