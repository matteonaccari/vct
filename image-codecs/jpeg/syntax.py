'''
Helper functions to handle the high level syntax by the JPEG standard:
https://www.itu.int/rec/T-REC-T.81-199209-I/en, that is markers and metadata
required by the decoder.

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
from bit_io import BitWriter
from nptyping import NDArray


def write_jfif_header(bw: BitWriter) -> None:
    jfif_data = [0xFF, 0xD8, 0xFF, 0xE0, 0, 16, ord('J'), ord('F'), ord('I'), ord('F'), 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]
    bw.write_bytes(jfif_data)


def write_segment_marker(bw: BitWriter, marker_id: int, length: int) -> None:
    data = [0xFF, marker_id]
    if length:
        data.extend([length >> 8, length & 0xFF])
    bw.write_bytes(data)


def write_comment(bw: BitWriter, comment: str) -> None:
    write_segment_marker(bw, 0xFE, 2 + len(comment))
    comment_numbers = [ord(i) for i in comment]
    bw.write_bytes(comment_numbers)


def write_quantisation_tables(bw: BitWriter, qy: NDArray[(64), np.int32], qc: NDArray[(64), np.int32]) -> None:
    y_data = [0] + qy.tolist()
    c_data = [1] + qc.tolist()

    write_segment_marker(bw, 0xDB, 2 + 2 * 65)
    bw.write_bytes(y_data)
    bw.write_bytes(c_data)


def write_start_of_frame(bw: BitWriter, frame_height: int, frame_width: int, components: int = 3) -> None:
    sof_data = [8, frame_height >> 8, frame_height & 255, frame_width >> 8, frame_width & 255, components, 1, 0x11, 0, 2, 0x11, 1, 3, 0x11, 1]
    write_segment_marker(bw, 0xC0, 2 + 6 + 3 * components)
    bw.write_bytes(sof_data)


def write_huffman_table(bw: BitWriter,
                        dc_y_bits: NDArray[(Any), np.int32], dc_y_values: NDArray[(Any), np.int32],
                        ac_y_bits: NDArray[(Any), np.int32], ac_y_values: NDArray[(Any), np.int32],
                        dc_c_bits: NDArray[(Any), np.int32], dc_c_values: NDArray[(Any), np.int32],
                        ac_c_bits: NDArray[(Any), np.int32], ac_c_values: NDArray[(Any), np.int32]) -> None:
    ly = 1 + len(dc_y_bits) + len(dc_y_values) + 1 + len(ac_y_bits) + len(ac_y_values)
    lc = 1 + len(dc_c_bits) + len(dc_c_values) + 1 + len(ac_c_bits) + len(ac_c_values)
    write_segment_marker(bw, 0xC4, 2 + ly + lc)

    # Luma
    bw.write_bytes([0x00])
    bw.write_bytes(dc_y_bits.tolist())
    bw.write_bytes(dc_y_values.tolist())
    bw.write_bytes([0x10])
    bw.write_bytes(ac_y_bits.tolist())
    bw.write_bytes(ac_y_values.tolist())

    # Chroma
    bw.write_bytes([0x01])
    bw.write_bytes(dc_c_bits.tolist())
    bw.write_bytes(dc_c_values.tolist())
    bw.write_bytes([0x11])
    bw.write_bytes(ac_c_bits.tolist())
    bw.write_bytes(ac_c_values.tolist())


def write_start_of_scan(bw: BitWriter) -> None:
    components = 3
    write_segment_marker(bw, 0xDA, 2 + 1 + 2 * components + 3)
    sos_data = [components, 1, 0, 2, 0x11, 3, 0x11, 0, 63, 0]
    bw.write_bytes(sos_data)
