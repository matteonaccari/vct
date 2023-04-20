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
from typing import Any, List, Tuple
from nptyping import NDArray
from enum import IntEnum


class DwtType(IntEnum):
    Haar = 0,
    LeGall5_3 = 1,
    CDF9_7 = 2

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return DwtType[s]
        except KeyError:
            raise ValueError()


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


'''
LeGall 5/3 Wavelet
'''


def extend_1D(samples_1D: NDArray[(Any), np.int32], table_left: List[int] = [2, 1],
              table_right: List[int] = [1, 2]) -> Tuple[NDArray[(Any), np.int32], int]:
    i0, i1 = 0, samples_1D.size
    i_left, i_right = table_left[i0 & 1], table_right[i1 & 1]

    samples_1D_ext = np.pad(samples_1D, (i_left, i_right), "reflect")
    return samples_1D_ext, i_left


def forward_filter_5_3(samples_ext: NDArray[(Any), np.int32], i0: int, i1: int, i_left: int) -> NDArray[(Any), np.int32]:
    samples_filtered = np.zeros_like(samples_ext)
    n_start, n_stop = (i0 + 1) // 2 - 1, (i1 + 1) // 2
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    samples_filtered[i_left + 2 * n + 1] = samples_ext[i_left + 2 * n + 1] - ((samples_ext[i_left + 2 * n] + samples_ext[i_left + 2 * n + 2]) // 2)

    n_start = (i0 + 1) // 2 - 1
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    samples_filtered[i_left + 2 * n] = samples_ext[i_left + 2 * n] + ((samples_filtered[i_left + 2 * n - 1] + samples_filtered[i_left + 2 * n + 1] + 2) // 4)
    return samples_filtered[i0 + i_left:i1 + i_left]


def inverse_filter_5_3(samples_ext: NDArray[(Any), np.int32], i0: int, i1: int, i_left: int) -> NDArray[(Any), np.int32]:
    samples_filtered = np.zeros_like(samples_ext)
    n_start, n_stop = i0 // 2, i1 // 2 + 1
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    samples_filtered[i_left + 2 * n] = samples_ext[i_left + 2 * n] - (samples_ext[i_left + 2 * n - 1] + samples_ext[i_left + 2 * n + 1] + 2) // 4

    n_stop = i1 // 2
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    samples_filtered[i_left + 2 * n + 1] = samples_ext[i_left + 2 * n + 1] + (samples_filtered[i_left + 2 * n] + samples_filtered[i_left + 2 * n + 2]) // 2
    return samples_filtered[i0 + i_left:i1 + i_left]


def forward_legall_5_3_dwt(image: NDArray[(Any, Any, Any), np.int32]) -> Tuple[NDArray[(Any, Any, Any), np.int32],
                                                                               NDArray[(Any, Any, Any), np.int32],
                                                                               NDArray[(Any, Any, Any), np.int32],
                                                                               NDArray[(Any, Any, Any), np.int32]]:
    rows, cols = image.shape[0], image.shape[1]
    components = 3 if len(image.shape) == 3 else 1
    if components == 3:
        coefficients = np.zeros((rows, cols, components), np.int32)
    else:
        coefficients = np.zeros((rows, cols), np.int32)

    # Transform on columns
    for c in range(cols):
        for comp in range(components):
            samples_column = image[:, c, comp] if components > 1 else image[:, c]
            # Extend
            samples_column_ext, i_left = extend_1D(samples_column)

            # Apply LeGall 5/3 filter with lifting
            filtered_column = forward_filter_5_3(samples_column_ext, 0, samples_column.size, i_left)
            if components == 1:
                coefficients[:, c] = filtered_column
            else:
                coefficients[:, c, comp] = filtered_column

    # Transform on rows
    for r in range(rows):
        for comp in range(components):
            samples_row = coefficients[r, :, comp] if components > 1 else coefficients[r, :]
            # Extend
            samples_row_ext, i_left = extend_1D(samples_row)

            # Apply LeGall 5/2 filter with lifting
            filtered_row = forward_filter_5_3(samples_row_ext, 0, samples_row.size, i_left)
            if components == 1:
                coefficients[r, :] = filtered_row
            else:
                coefficients[r, :, comp] = filtered_row

    # Deinterleaving
    ll = coefficients[::2, ::2]
    hl = coefficients[::2, 1::2]
    lh = coefficients[1::2, ::2]
    hh = coefficients[1::2, 1::2]

    return ll, hl, lh, hh


def inverse_legall_5_3_dwt(ll: NDArray[(Any, Any, Any), np.int32],
                           hl: NDArray[(Any, Any, Any), np.int32],
                           lh: NDArray[(Any, Any, Any), np.int32],
                           hh: NDArray[(Any, Any, Any), np.int32]) -> NDArray[(Any, Any, Any), np.int32]:
    rows, cols = ll.shape[0] << 1, ll.shape[1] << 1
    if len(ll.shape) == 3:
        samples = np.zeros((rows, cols, 3), np.int32)
        components = 3
    else:
        samples = np.zeros((rows, cols), np.int32)
        components = 1

    # Interleaving
    samples[::2, ::2] = ll
    samples[::2, 1::2] = hl
    samples[1::2, ::2] = lh
    samples[1::2, 1::2] = hh

    # Transform on rows
    for r in range(rows):
        for comp in range(components):
            coefficients_row = samples[r, :, comp] if components > 1 else samples[r, :]

            # Extend
            coefficients_row_ext, i_left = extend_1D(coefficients_row, [1, 2], [2, 1])

            # Filter
            filtered_row = inverse_filter_5_3(coefficients_row_ext, 0, coefficients_row.size, i_left)

            if components == 1:
                samples[r, :] = filtered_row
            else:
                samples[r, :, comp] = filtered_row

    # Transform on columns
    for c in range(cols):
        for comp in range(components):
            coefficients_column = samples[:, c, comp] if components > 1 else samples[:, c]

            # Extend
            coefficients_column_ext, i_left = extend_1D(coefficients_column, [1, 2], [2, 1])

            # Filter
            filtered_column = inverse_filter_5_3(coefficients_column_ext, 0, coefficients_column.size, i_left)

            if components == 1:
                samples[:, c] = filtered_column
            else:
                samples[:, c, comp] = filtered_column

    return samples
