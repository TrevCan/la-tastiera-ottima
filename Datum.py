from Singleton import Singleton

class RegionCreator:
    def __init__(self):
        self.tag_window = "region_creator"
        self.tag_draw_list = "draw_list"

        self.tag_draw_value_offset_x = "offset_x"
        self.tag_draw_value_offset_y = "offset_y"

        # this should compensate for any weird padding in windows!
        self.draw_offset_x = 0
        self.draw_offset_y = 0

        self.window_parameters = {
            'label': 'DRAW_WIN',
#            "width" : 700,
#            "height" : 500,
            "no_move" : True,
            "no_collapse" : True,
            "no_close" : True,
            "no_background" : True,
            "no_title_bar" : True,
            "no_resize" : True,
            "tag": self.tag_window
        }

        self.draw_list_parameters = {
            "width": 700,
            "height": 500,
            "tag": self.tag_draw_list
        }

        self.current_tag_layer = "o_m_c_m"
        self.current_layer_index = 2

        self.tag_optical_main_crop_mask = "o_m_c_m"
        self.is_showing_layer_optical_main_crop_mask = [True]

        self.tag_optical_roi_crop_mask = "o_r_c_m"
        self.is_showing_layer_optical_roi_crop_mask = [True]

        self.tag_ocr_roi = "ocr_m"
        self.is_showing_layer_ocr_roi = [True]


        self.layers = [
            { "tag": self.tag_ocr_roi, "isShowing": self.is_showing_layer_ocr_roi, "label": "OCR ROI", "tag_checkbox_use": f"radio_button_{self.tag_ocr_roi}",
              "tag_checkbox_show": f"show_{self.tag_ocr_roi}"},
            { "tag": self.tag_optical_roi_crop_mask,
                "isShowing": self.is_showing_layer_optical_roi_crop_mask, "label": "Optical ROI Mask", "tag_checkbox_use":f"radio_button_{self.tag_optical_roi_crop_mask}",
              "tag_checkbox_show": f"show_{self.tag_optical_roi_crop_mask}"},
            { "tag": self.tag_optical_main_crop_mask, "isShowing": self.is_showing_layer_optical_main_crop_mask, "label": "Optical Crop Mask", "tag_checkbox_use": f"radio_button_{self.tag_optical_main_crop_mask}",
              "tag_checkbox_show": f"show_{self.tag_optical_main_crop_mask}" }
        ]


class Datum:
    def __init__(self):
        self.x = 0
        self.regionCreator = RegionCreator()

        self.tag_stream_texture = "raw_stream_texture"
        self.tag_mask_texture = "mask_texture"
        self.tag_roi_w_stream_texture = "roi_stream_texture"
        self.tag_optical_texture = "optical_texture"

