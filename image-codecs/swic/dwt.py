'''
Forward and inverse Discrete Wavelet Transformation (DWT) using the following Wavelet type:
 * Haar Wavelet
 * LeGall 5/3 as used in the JPEG 2000 (ITU-T T.800) standard for lossless encoding
 * Cohen Dabeuchies Feauveau (CDF) 9/7 as used in the JPEG 2000 standard for lossy
   encoding

Inputs a assumed to be 3D arrays representing image pixels with (up to) three colour
component (e.g. Y, Cb and Cr). All DWTs are implemented using the lifting technique
proposed by W. Sweldens

Copyright(c) 2023 Matteo Naccari
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

import numpy as np
from typing import Any, Tuple
from nptyping import NDArray

'''
Haar Wavelet
'''


def forward_haar_dwt(image: NDArray[(Any, Any, Any), np.int32],
                     range_expansion: bool = False) -> Tuple[NDArray[(Any, Any, Any), np.int32],
                                                             NDArray[(Any, Any, Any), np.int32],
                                                             NDArray[(Any, Any, Any), np.int32],
                                                             NDArray[(Any, Any, Any), np.int32]]:

    rows, cols = image.shape[0], image.shape[1]
    rows_pad, cols_pad = ((rows + 1) >> 1) << 1, ((cols + 1) >> 1) << 1

    coefficients = np.zeros(image.shape, np.int32)
    coefficients[:rows, :cols] = image

    if rows_pad != rows:
        coefficients[rows:rows_pad, :, :] = coefficients[rows, :, :]

    if cols_pad != cols:
        coefficients[:, cols:cols_pad, :] = coefficients[:, cols, :]

    if range_expansion:
        # Transform on columns
        coefficients[:, ::2] = coefficients[:, ::2] + coefficients[:, 1::2]
        coefficients[:, 1::2] = coefficients[:, 1::2] - coefficients[:, ::2] // 2

        # Transform on rows
        coefficients[::2, :] = coefficients[::2, :] + coefficients[1::2, :]
        coefficients[1::2, :] = coefficients[1::2, :] - coefficients[::2, :] // 2
    else:
        # Transform on columns
        coefficients[:, 1::2] = coefficients[:, 1::2] - coefficients[:, ::2]
        coefficients[:, ::2] = coefficients[:, ::2] + coefficients[:, 1::2] // 2

        # Transform on rows
        coefficients[1::2, :] = coefficients[1::2, :] - coefficients[::2, :]
        coefficients[::2, :] = coefficients[::2, :] + coefficients[1::2, :] // 2

    # Deinterleaving
    ll = coefficients[::2, ::2]
    hl = coefficients[::2, 1::2]
    lh = coefficients[1::2, ::2]
    hh = coefficients[1::2, 1::2]

    return ll, hl, lh, hh


def inverse_haar_dwt(ll: NDArray[(Any, Any, Any), np.int32],
                     hl: NDArray[(Any, Any, Any), np.int32],
                     lh: NDArray[(Any, Any, Any), np.int32],
                     hh: NDArray[(Any, Any, Any), np.int32], range_expansion: bool = False) -> NDArray[(Any, Any, Any), np.int32]:
    rows, cols = ll.shape[0], ll.shape[1]
    if len(ll.shape) == 3:
        samples = np.zeros((2 * rows, 2 * cols, 3), np.int32)
    else:
        samples = np.zeros((2 * rows, 2 * cols), np.int32)

    # Interleaving
    samples[::2, ::2] = ll
    samples[::2, 1::2] = hl
    samples[1::2, ::2] = lh
    samples[1::2, 1::2] = hh

    if range_expansion:
        # Transform on rows
        samples[1::2, :] = samples[1::2, :] + samples[::2, :] // 2
        samples[::2, :] = samples[::2, :] - samples[1::2, :]

        # Transform on columns
        samples[:, 1::2] = samples[:, 1::2] + samples[:, ::2] // 2
        samples[:, ::2] = samples[:, ::2] - samples[:, 1::2]
    else:
        samples[::2, :] = samples[::2, :] - samples[1::2, :] // 2
        samples[1::2, :] = samples[1::2, :] + samples[::2, :]

        # Transform on columns
        samples[:, ::2] = samples[:, ::2] - samples[:, 1::2] // 2
        samples[:, 1::2] = samples[:, 1::2] + samples[:, ::2]

    return samples
