'''
Implementation of the uniform quantiser specified by the H.264/AVC, H.265/HEVC and
H.266/VVC standards(*) whereby the quantisation step (delt) increases exponentially
with the Quantisation Parameter (QP), according to the following approximation:

    delta = 2**((QP - 4) / 6)

Inputs a assumed to be 3D arrays representing Wavelet transform coefficients.
(*) H.266/VVC extends the QP range to [0, 62] but here only the old range [0, 51] is
implemented.

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
from nptyping import NDArray, Shape

quantiser_scale = np.array([26214, 23302, 20560, 18396, 16384, 14564], np.int32)
reconstruction_scale = np.array([40, 45, 51, 57, 64, 72], np.int32)


def quantise_plane(subband_plane: NDArray[Shape["*, *"], np.int32], qp: int) -> NDArray[Shape["*, *"], np.int32]:
    # Definitions
    quantiser_shift = 14
    qp_rem, qp_per = qp % 6, qp // 6
    shift_amount = quantiser_shift + qp_per
    offset = 1 << (shift_amount - 1)

    # Quantisation
    s = np.sign(subband_plane)
    levels_plane = (np.abs(subband_plane) * quantiser_scale[qp_rem] + offset) >> shift_amount
    levels_plane *= s

    return levels_plane


def reconstruct_plane(levels_plane: NDArray[Shape["*, *"], np.int32], qp: int) -> NDArray[Shape["*, *"], np.int32]:
    # Definitions
    quantiser_shift = 6
    qp_rem, qp_per = qp % 6, qp // 6
    shift_amount = quantiser_shift - qp_per

    # Reconstruction
    if shift_amount > 0:
        offset = 1 << (shift_amount - 1)
        coefficients_rec = (levels_plane * reconstruction_scale[qp_rem] + offset) >> shift_amount
    else:
        shift_amount = -shift_amount
        coefficients_rec = (levels_plane * reconstruction_scale[qp_rem]) << shift_amount

    return coefficients_rec
