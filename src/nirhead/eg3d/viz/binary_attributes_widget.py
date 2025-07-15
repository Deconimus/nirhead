# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import numpy as np
import imgui
import eg3d.dnnlib as dnnlib
from eg3d.gui_utils import imgui_utils

#----------------------------------------------------------------------------

class BinaryAttributesWidget:
    def __init__(self, viz):
        self.viz          = viz
        self.bright_pupil = False
        self.glasses      = False
        self.facial_hair  = False

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            #imgui.text('Binary Attributes:')
            #imgui.same_line()
            _clicked, self.bright_pupil = imgui.checkbox('Bright Pupil', self.bright_pupil)
            imgui.same_line()
            _clicked, self.glasses = imgui.checkbox('Glasses', self.glasses)
            imgui.same_line()
            _clicked, self.facial_hair = imgui.checkbox('Facial Hair', self.facial_hair)

        viz.args.bright_pupil = self.bright_pupil
        viz.args.glasses      = self.glasses
        viz.args.facial_hair  = self.facial_hair

#----------------------------------------------------------------------------
