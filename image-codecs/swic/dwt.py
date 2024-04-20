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

from enum import IntEnum
from typing import List, Tuple

import numpy as np
from nptyping import NDArray, Shape


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


class Direction(IntEnum):
    Horizontal = 0,
    Vertical = 1

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


def forward_haar_dwt(image: NDArray[Shape["*, *, *"], np.int32],
                     range_expansion: bool = False) -> Tuple[NDArray[Shape["*, *, *"], np.int32],
                                                             NDArray[Shape["*, *, *"], np.int32],
                                                             NDArray[Shape["*, *, *"], np.int32],
                                                             NDArray[Shape["*, *, *"], np.int32]]:

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


def inverse_haar_dwt(ll: NDArray[Shape["*, *, *"], np.int32],
                     hl: NDArray[Shape["*, *, *"], np.int32],
                     lh: NDArray[Shape["*, *, *"], np.int32],
                     hh: NDArray[Shape["*, *, *"], np.int32], range_expansion: bool = False) -> NDArray[Shape["*, *, *"], np.int32]:
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


def extend(samples_2D: NDArray[Shape["*, *, *"], np.int32], table_left: List[int] = [2, 1],
           table_right: List[int] = [1, 2], d: Direction = Direction.Vertical) -> Tuple[NDArray[Shape["*, *, *"], np.int32], int]:
    i0, i1 = 0, samples_2D.shape[0] if d == Direction.Vertical else samples_2D.shape[1]
    components = 1 if len(samples_2D.shape) == 2 else 3
    i_left, i_right = table_left[i0 & 1], table_right[i1 & 1]

    extension_params = [[i_left, i_right], [0, 0]] if d == Direction.Vertical else [[0, 0], [i_left, i_right]]
    if components > 1:
        extension_params += [[0, 0]]
    samples_2D_ext = np.pad(samples_2D, extension_params, "reflect")

    return samples_2D_ext, i_left


def forward_filter_5_3(samples_ext: NDArray[Shape["*, *, *"], np.int32], i0: int, i1: int, i_left: int, d: Direction = Direction.Vertical) -> NDArray[Shape["*, *, *"], np.int32]:
    samples_filtered = np.zeros_like(samples_ext)
    n_start, n_stop = (i0 + 1) // 2 - 1, (i1 + 1) // 2
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n + 1, :] = samples_ext[i_left + 2 * n + 1, :] - ((samples_ext[i_left + 2 * n, :] + samples_ext[i_left + 2 * n + 2, :]) // 2)
    else:
        samples_filtered[:, i_left + 2 * n + 1] = samples_ext[:, i_left + 2 * n + 1] - ((samples_ext[:, i_left + 2 * n] + samples_ext[:, i_left + 2 * n + 2]) // 2)

    n_start = (i0 + 1) // 2 - 1
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n, :] = samples_ext[i_left + 2 * n, :] + ((samples_filtered[i_left + 2 * n - 1, :] + samples_filtered[i_left + 2 * n + 1, :] + 2) // 4)
        return samples_filtered[i0 + i_left:i1 + i_left, :]
    else:
        samples_filtered[:, i_left + 2 * n] = samples_ext[:, i_left + 2 * n] + ((samples_filtered[:, i_left + 2 * n - 1] + samples_filtered[:, i_left + 2 * n + 1] + 2) // 4)
        return samples_filtered[:, i0 + i_left:i1 + i_left]


def inverse_filter_5_3(samples_ext: NDArray[Shape["*, *, *"], np.int32], i0: int, i1: int, i_left: int, d: Direction = Direction.Vertical) -> NDArray[Shape["*, *, *"], np.int32]:
    samples_filtered = np.zeros_like(samples_ext)
    n_start, n_stop = i0 // 2, i1 // 2 + 1
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n, :] = samples_ext[i_left + 2 * n, :] - (samples_ext[i_left + 2 * n - 1, :] + samples_ext[i_left + 2 * n + 1, :] + 2) // 4
    else:
        samples_filtered[:, i_left + 2 * n] = samples_ext[:, i_left + 2 * n] - (samples_ext[:, i_left + 2 * n - 1] + samples_ext[:, i_left + 2 * n + 1] + 2) // 4

    n_stop = i1 // 2
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n + 1, :] = samples_ext[i_left + 2 * n + 1, :] + (samples_filtered[i_left + 2 * n, :] + samples_filtered[i_left + 2 * n + 2, :]) // 2
        return samples_filtered[i0 + i_left:i1 + i_left, :]
    else:
        samples_filtered[:, i_left + 2 * n + 1] = samples_ext[:, i_left + 2 * n + 1] + (samples_filtered[:, i_left + 2 * n] + samples_filtered[:, i_left + 2 * n + 2]) // 2
        return samples_filtered[:, i0 + i_left:i1 + i_left]


def forward_legall_5_3_dwt(image: NDArray[Shape["*, *, *"], np.int32]) -> Tuple[NDArray[Shape["*, *, *"], np.int32],
                                                                                NDArray[Shape["*, *, *"], np.int32],
                                                                                NDArray[Shape["*, *, *"], np.int32],
                                                                                NDArray[Shape["*, *, *"], np.int32]]:
    rows, cols = image.shape[0], image.shape[1]

    # Transform on columns
    # Extend
    image_ext, i_left = extend(image)
    # Apply LeGall 5/3 filter with lifting
    coefficients = forward_filter_5_3(image_ext, 0, rows, i_left)

    # Transform on rows
    # Extend
    coefficients_ext, i_left = extend(coefficients, d=Direction.Horizontal)
    # Apply LeGall 5/3 filter with lifting
    coefficients = forward_filter_5_3(coefficients_ext, 0, cols, i_left, Direction.Horizontal)

    # Deinterleaving
    ll = coefficients[::2, ::2]
    hl = coefficients[::2, 1::2]
    lh = coefficients[1::2, ::2]
    hh = coefficients[1::2, 1::2]

    return ll, hl, lh, hh


def inverse_legall_5_3_dwt(ll: NDArray[Shape["*, *, *"], np.int32],
                           hl: NDArray[Shape["*, *, *"], np.int32],
                           lh: NDArray[Shape["*, *, *"], np.int32],
                           hh: NDArray[Shape["*, *, *"], np.int32]) -> NDArray[Shape["*, *, *"], np.int32]:
    rows, cols = ll.shape[0] << 1, ll.shape[1] << 1
    if len(ll.shape) == 3:
        samples = np.zeros((rows, cols, 3), np.int32)
    else:
        samples = np.zeros((rows, cols), np.int32)

    # Interleaving
    samples[::2, ::2] = ll
    samples[::2, 1::2] = hl
    samples[1::2, ::2] = lh
    samples[1::2, 1::2] = hh

    # Transform on rows
    # Extend
    samples_ext, i_left = extend(samples, [1, 2], [2, 1], Direction.Horizontal)
    # Filter
    samples = inverse_filter_5_3(samples_ext, 0, cols, i_left, Direction.Horizontal)

    # Transform on columns
    samples_ext, i_left = extend(samples, [1, 2], [2, 1])
    # Filter
    samples = inverse_filter_5_3(samples_ext, 0, rows, i_left)

    return samples


'''
Cohen Dabeuchies Feauveau (CDF) 9/7 Wavelet
'''
alpha, beta, gamma, delta, K = -1.586134342059924, -0.052980118572961, 0.882911075530934, 0.443506852043971, 1.230174104914001


def forward_filter_9_7(samples_ext: NDArray[Shape["*, *, *"], np.float64], i0: int, i1: int, i_left: int, d: Direction = Direction.Vertical) -> NDArray[Shape["*, *, *"], np.float64]:
    samples_filtered = np.zeros(samples_ext.shape)

    # Step 1
    n_start, n_stop = ((i0 + 1) >> 1) - 2, ((i1 + 1) >> 1) + 1
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n + 1, :] = samples_ext[i_left + 2 * n + 1, :] + alpha * (samples_ext[i_left + 2 * n, :] + samples_ext[i_left + 2 * n + 2, :])
    else:
        samples_filtered[:, i_left + 2 * n + 1] = samples_ext[:, i_left + 2 * n + 1] + alpha * (samples_ext[:, i_left + 2 * n] + samples_ext[:, i_left + 2 * n + 2])

    # Step 2
    n_start, n_stop = ((i0 + 1) >> 1) - 1, ((i1 + 1) >> 1) + 1
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n, :] = samples_ext[i_left + 2 * n, :] + beta * (samples_filtered[i_left + 2 * n - 1, :] + samples_filtered[i_left + 2 * n + 1, :])
    else:
        samples_filtered[:, i_left + 2 * n] = samples_ext[:, i_left + 2 * n] + beta * (samples_filtered[:, i_left + 2 * n - 1] + samples_filtered[:, i_left + 2 * n + 1])

    # Step 3
    n_start, n_stop = ((i0 + 1) >> 1) - 1, ((i1 + 1) >> 1)
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n + 1, :] += gamma * (samples_filtered[i_left + 2 * n, :] + samples_filtered[i_left + 2 * n + 2, :])
    else:
        samples_filtered[:, i_left + 2 * n + 1] += gamma * (samples_filtered[:, i_left + 2 * n] + samples_filtered[:, i_left + 2 * n + 2])

    # Step 4
    n_start, n_stop = ((i0 + 1) >> 1), ((i1 + 1) >> 1)
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n, :] += delta * (samples_filtered[i_left + 2 * n - 1, :] + samples_filtered[i_left + 2 * n + 1, :])
    else:
        samples_filtered[:, i_left + 2 * n] += delta * (samples_filtered[:, i_left + 2 * n - 1] + samples_filtered[:, i_left + 2 * n + 1])

    # Step 5
    n_start = ((i0 + 1) >> 1) - 1
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n + 1, :] *= K
    else:
        samples_filtered[:, i_left + 2 * n + 1] *= K

    # Step 6
    n_start = ((i0 + 1) >> 1)
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n, :] *= (1 / K)
        return samples_filtered[i_left + i0:i1 + i_left, :]
    else:
        samples_filtered[:, i_left + 2 * n] *= (1 / K)
        return samples_filtered[:, i_left + i0:i1 + i_left]


def inverse_filter_9_7(samples_ext: NDArray[Shape["*, *, *"], np.float64], i0: int, i1: int, i_left: int, d: Direction = Direction.Vertical) -> NDArray[Shape["*, *, *"], np.float64]:
    samples_filtered = np.zeros(samples_ext.shape)

    # Step 1
    n_start, n_stop = (i0 >> 1) - 1, (i1 >> 1) + 2
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n, :] = K * samples_ext[i_left + 2 * n, :]
    else:
        samples_filtered[:, i_left + 2 * n] = K * samples_ext[:, i_left + 2 * n]

    # Step 2
    n_start, n_stop = (i0 >> 1) - 2, (i1 >> 1) + 2
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n + 1, :] = (1 / K) * samples_ext[i_left + 2 * n + 1, :]
    else:
        samples_filtered[:, i_left + 2 * n + 1] = (1 / K) * samples_ext[:, i_left + 2 * n + 1]

    # Step 3
    n_start, n_stop = (i0 >> 1) - 1, (i1 >> 1) + 2
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n, :] -= delta * (samples_filtered[i_left + 2 * n - 1, :] + samples_filtered[i_left + 2 * n + 1, :])
    else:
        samples_filtered[:, i_left + 2 * n] -= delta * (samples_filtered[:, i_left + 2 * n - 1] + samples_filtered[:, i_left + 2 * n + 1])

    # Step 4
    n_start, n_stop = (i0 >> 1) - 1, (i1 >> 1) + 1
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n + 1, :] -= gamma * (samples_filtered[i_left + 2 * n, :] + samples_filtered[i_left + 2 * n + 2, :])
    else:
        samples_filtered[:, i_left + 2 * n + 1] -= gamma * (samples_filtered[:, i_left + 2 * n] + samples_filtered[:, i_left + 2 * n + 2])

    # Step 5
    n_start, n_stop = i0 >> 1, (i1 >> 1) + 1
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n, :] -= beta * (samples_filtered[i_left + 2 * n - 1, :] + samples_filtered[i_left + 2 * n + 1, :])
    else:
        samples_filtered[:, i_left + 2 * n] -= beta * (samples_filtered[:, i_left + 2 * n - 1] + samples_filtered[:, i_left + 2 * n + 1])

    # Step 6
    n_start, n_stop = i0 >> 1, i1 >> 1
    n = np.array([i for i in range(n_start, n_stop)], np.int32)
    if d == Direction.Vertical:
        samples_filtered[i_left + 2 * n + 1, :] -= alpha * (samples_filtered[i_left + 2 * n, :] + samples_filtered[i_left + 2 * n + 2, :])
        return samples_filtered[i_left + i0:i1 + i_left, :]
    else:
        samples_filtered[:, i_left + 2 * n + 1] -= alpha * (samples_filtered[:, i_left + 2 * n] + samples_filtered[:, i_left + 2 * n + 2])
        return samples_filtered[:, i_left + i0:i1 + i_left]


def forward_cdf_9_7_dwt(image: NDArray[Shape["*, *, *"], np.int32]) -> Tuple[NDArray[Shape["*, *, *"], np.float64],
                                                                             NDArray[Shape["*, *, *"], np.int32],
                                                                             NDArray[Shape["*, *, *"], np.int32],
                                                                             NDArray[Shape["*, *, *"], np.int32]]:
    rows, cols = image.shape[0], image.shape[1]

    # Transform on columns
    # Extend
    image_ext, i_left = extend(image, [4, 3], [3, 4])
    # Apply CDF 9/7 filter with lifting
    coefficients = forward_filter_9_7(image_ext, 0, rows, i_left)

    # Transform on rows
    # Extend
    coefficients_ext, i_left = extend(coefficients, [4, 3], [3, 4], d=Direction.Horizontal)
    # Apply CDF 9/7 filter with lifting
    coefficients = forward_filter_9_7(coefficients_ext, 0, cols, i_left, Direction.Horizontal)

    # Deinterleaving
    ll = coefficients[::2, ::2]
    hl = (np.round(coefficients[::2, 1::2])).astype(np.int32)
    lh = (np.round(coefficients[1::2, ::2])).astype(np.int32)
    hh = (np.round(coefficients[1::2, 1::2])).astype(np.int32)

    return ll, hl, lh, hh


def inverse_cdf_9_7_dwt(ll: NDArray[Shape["*, *, *"], np.float64],
                        hl: NDArray[Shape["*, *, *"], np.float64],
                        lh: NDArray[Shape["*, *, *"], np.float64],
                        hh: NDArray[Shape["*, *, *"], np.float64]) -> NDArray[Shape["*, *, *"], np.float64]:
    rows, cols = ll.shape[0] << 1, ll.shape[1] << 1
    if len(ll.shape) == 3:
        samples = np.zeros((rows, cols, 3), np.float64)
    else:
        samples = np.zeros((rows, cols), np.float64)

    # Interleaving
    samples[::2, ::2] = ll.astype(np.float64)
    samples[::2, 1::2] = hl.astype(np.float64)
    samples[1::2, ::2] = lh.astype(np.float64)
    samples[1::2, 1::2] = hh.astype(np.float64)

    # Transform on rows
    # Extend
    samples_ext, i_left = extend(samples, [3, 4], [4, 3], Direction.Horizontal)
    # Filter
    samples = inverse_filter_9_7(samples_ext, 0, cols, i_left, Direction.Horizontal)

    # Transform on columns
    samples_ext, i_left = extend(samples, [3, 4], [4, 3])
    # Filter
    samples = inverse_filter_9_7(samples_ext, 0, rows, i_left)

    return samples
