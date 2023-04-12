'''
Methods implementing different conversions from the RGB to the YCbCr
colour spaces with primaries from different ITU-R recommendations.
Implemented so far is: ITU-R BT.601 an ITU-R B.709, in full range mode
ITU-R BT.601: https://www.itu.int/rec/R-REC-BT.601/
ITU-R BT.709: https://www.itu.int/rec/R-REC-BT.709/

Copyright(c) 2022 Matteo Naccari
All Rights Reserved.

email: matteo.naccari@gmail.com | matteo.naccari@polimi.it | matteo.naccari@lx.it.pt

The copyright in this collection of software modules is being made available under the BSD
License, included below. This software may be subject to other third party
and contributor rights, including patent rights, and no such rights are
granted under this license.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the author may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES LOSS OF USE, DATA, OR PROFITS OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
'''

from typing import Any

import numpy as np
from nptyping import NDArray


def rgb_to_ycbcr_bt601(red: NDArray[(Any, Any), np.uint8],
                       green: NDArray[(Any, Any), np.uint8],
                       blue: NDArray[(Any, Any), np.uint8], bpp: int = 8) -> NDArray[(Any, Any, 3), np.int32]:
    max_value = (1 << bpp) - 1
    mid_range = 1 << (bpp - 1)
    red = red.astype(np.float64) / max_value
    green = green.astype(np.float64) / max_value
    blue = blue.astype(np.float64) / max_value
    T = np.array([[0.299, 0.587, 0.114],
                  [-0.299, -0.587, 0.886],
                  [0.701, -0.587, -0.114]])

    T[1, :] /= 1.772
    T[2, :] /= 1.402

    ycbcr_image = np.zeros((red.shape[0], red.shape[1], 3), np.float64)
    ycbcr_image[:, :, 0] = max_value * (T[0, 0] * red + T[0, 1] * green + T[0, 2] * blue)
    ycbcr_image[:, :, 1] = max_value * (T[1, 0] * red + T[1, 1] * green + T[1, 2] * blue) + mid_range
    ycbcr_image[:, :, 2] = max_value * (T[2, 0] * red + T[2, 1] * green + T[2, 2] * blue) + mid_range
    ycbcr_image = np.clip(ycbcr_image + 0.5, 0, max_value)

    return ycbcr_image.astype(np.int32)


def rgb_to_ycbcr_bt709(red: NDArray[(Any, Any), np.uint8],
                       green: NDArray[(Any, Any), np.uint8],
                       blue: NDArray[(Any, Any), np.uint8], bpp: int = 8) -> NDArray[(Any, Any, 3), np.int32]:
    max_value = (1 << bpp) - 1
    mid_range = 1 << (bpp - 1)
    red = red.astype(np.float64) / max_value
    green = green.astype(np.float64) / max_value
    blue = blue.astype(np.float64) / max_value
    T = np.array([[0.2126, 0.7152, 0.0722],
                  [-0.2126, -0.7152, 0.9278],
                  [0.7874, -0.7152, -0.0722]])

    T[1, :] /= 1.8556
    T[2, :] /= 1.5748

    ycbcr_image = np.zeros((red.shape[0], red.shape[1], 3), np.float64)
    ycbcr_image[:, :, 0] = max_value * (T[0, 0] * red + T[0, 1] * green + T[0, 2] * blue)
    ycbcr_image[:, :, 1] = max_value * (T[1, 0] * red + T[1, 1] * green + T[1, 2] * blue) + mid_range
    ycbcr_image[:, :, 2] = max_value * (T[2, 0] * red + T[2, 1] * green + T[2, 2] * blue) + mid_range
    ycbcr_image = np.clip(ycbcr_image + 0.5, 0, max_value)

    return ycbcr_image.astype(np.int32)


def ycbcr_to_rgb_bt601(y: NDArray[(Any, Any), np.int32],
                       cb: NDArray[(Any, Any), np.int32],
                       cr: NDArray[(Any, Any), np.int32], bpp: int = 8) -> NDArray[(Any, Any, 3), np.int32]:
    max_value = (1 << bpp) - 1
    mid_range = 1 << (bpp - 1)
    y = y.astype(np.float64) / max_value
    cb = (cb.astype(np.float64) - mid_range) / max_value
    cr = (cr.astype(np.float64) - mid_range) / max_value
    Tf = np.array([[0.299, 0.587, 0.114],
                   [-0.299, -0.587, 0.886],
                   [0.701, -0.587, -0.114]])
    Tf /= 1.772
    Tf /= 1.402
    T = np.linalg.inv(Tf)

    rgb_image = np.zeros((y.shape[0], y.shape[1], 3), np.float64)
    rgb_image[:, :, 0] = max_value * (T[0, 0] * y + T[0, 1] * cb + T[0, 2] * cr)
    rgb_image[:, :, 1] = max_value * (T[1, 0] * y + T[1, 1] * cb + T[1, 2] * cr)
    rgb_image[:, :, 2] = max_value * (T[2, 0] * y + T[2, 1] * cb + T[2, 2] * cr)
    rgb_image = np.clip(rgb_image + 0.5, 0, max_value)

    return rgb_image.astype(np.int32)


def ycbcr_to_rgb_bt709(y: NDArray[(Any, Any), np.int32],
                       cb: NDArray[(Any, Any), np.int32],
                       cr: NDArray[(Any, Any), np.int32], bpp: int = 8) -> NDArray[(Any, Any, 3), np.int32]:
    max_value = (1 << bpp) - 1
    mid_range = 1 << (bpp - 1)
    y = y.astype(np.float64) / max_value
    cb = (cb.astype(np.float64) - mid_range) / max_value
    cr = (cr.astype(np.float64) - mid_range) / max_value
    Tf = np.array([[0.2126, 0.7152, 0.0722],
                   [-0.2126, -0.7152, 0.9278],
                   [0.7874, -0.7152, -0.0722]])
    Tf[1, :] /= 1.8556
    Tf[2, :] /= 1.5748
    T = np.linalg.inv(Tf)

    rgb_image = np.zeros((y.shape[0], y.shape[1], 3), np.float64)
    rgb_image[:, :, 0] = max_value * (T[0, 0] * y + T[0, 1] * cb + T[0, 2] * cr)
    rgb_image[:, :, 1] = max_value * (T[1, 0] * y + T[1, 1] * cb + T[1, 2] * cr)
    rgb_image[:, :, 2] = max_value * (T[2, 0] * y + T[2, 1] * cb + T[2, 2] * cr)
    rgb_image = np.clip(rgb_image + 0.5, 0, max_value)

    return rgb_image.astype(np.int32)
