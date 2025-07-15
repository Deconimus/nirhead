# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
import numpy as np
import imgui
import eg3d.dnnlib as dnnlib
from eg3d.gui_utils import imgui_utils


# ----------------------------------------------------------------------------

class EyeOpenWidget:
    
    def __init__(self, viz):
        self.viz = viz
        self.eye_open = dnnlib.EasyDict(x=0.5, anim=False, speed=0.25)
        self.eye_open_def = dnnlib.EasyDict(self.eye_open)
        
    def drag(self, dx, dy):
        viz = self.viz
        self.eye_open.x += -dy / viz.font_size * 4e-2
        self.limit_values()
    
    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text('Eye Open')
            imgui.same_line(viz.label_w)
            self.limit_values()
            eye_open = self.eye_open.x
            with imgui_utils.item_width(viz.font_size * 5):
                changed, new_eye_open = imgui.input_float('##frac', eye_open, format='%+.2f', flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
                if changed:
                    self.eye_open.x = new_eye_open
                    self.limit_values()
            imgui.same_line(viz.label_w + viz.font_size * 13 + viz.spacing * 2)
            _clicked, dragging, dx, dy = imgui_utils.drag_button('Drag', width=viz.button_w)
            if dragging:
                self.drag(dx, dy)
            imgui.same_line()
            snapped = dnnlib.EasyDict(self.eye_open, x=round(self.eye_open.x, 1))
            if imgui_utils.button('Snap', width=viz.button_w, enabled=(self.eye_open != snapped)):
                self.eye_open = snapped
            imgui.same_line()
            if imgui_utils.button('Reset', width=-1, enabled=(self.eye_open != self.eye_open_def)):
                self.gaze = dnnlib.EasyDict(self.eye_open_def)
        
        viz.args.eye_open = self.eye_open.x
    
    def limit_values(self):
        self.eye_open.x = min(max(self.eye_open.x, 0.0), 1.0)

# ----------------------------------------------------------------------------
