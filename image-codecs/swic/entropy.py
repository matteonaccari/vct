'''
Routines to perform entropy encoding and decoding in the Simple Wavelet-based Image Coding (SWIC)

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

from typing import Any, List

import numpy as np
from nptyping import NDArray

from bit_io import BitReader, BitWriter

code_block_size, code_block_buffer_size = 32, 8192


def egc0_encoding(value, bw: BitWriter) -> None:
    if not value:
        bw.write_bits(1, 1)
        return
    value += 1
    k = int(np.log2(value))
    r = value % (1 << k)
    bw.write_bits(1 << k, k + 1)
    bw.write_bits(r, k)


def egc0_decoding(br: BitReader) -> int:
    k = 0
    while not br.read(1):
        k += 1

    r = 0
    if k:
        r = br.read(k)
    value = (1 << k) + r - 1
    return value


def encode_pure_egc(coefficients: NDArray[(Any), np.int32], bw: BitWriter) -> None:
    # Take absolute value and sign of coefficients
    coefficients_abs = np.abs(coefficients)
    coefficients_s = np.sign(coefficients)

    # Calculate the total_number of coefficients
    idx = np.where(coefficients_abs)
    total_coefficients = idx[0][-1] + 1 if len(idx[0]) else 0

    # Write the total coefficients
    egc0_encoding(total_coefficients, bw)

    for idx in range(total_coefficients):
        egc0_encoding(coefficients_abs[idx], bw)
        if coefficients_abs[idx]:
            # Write also the sign for a significant coefficient
            sign_bit = 1 if coefficients_s[idx] < 0 else 0
            bw.write_bits(sign_bit, 1)


def encode_rle_egc(coefficients: NDArray[(Any), np.int32], bw: BitWriter) -> None:
    # Take absolute value and sign of coefficients
    coefficients_abs = np.abs(coefficients)
    coefficients_s = np.sign(coefficients)

    # Calculate the total_number of coefficients
    idx = np.where(coefficients_abs)
    total_coefficients = idx[0][-1] + 1 if len(idx[0]) else 0

    # Write the total coefficients
    egc0_encoding(total_coefficients, bw)

    idx = 0
    while idx < total_coefficients:
        current_length = 0
        while idx < total_coefficients and not coefficients_abs[idx]:
            current_length += 1
            idx += 1

        # Encode the run length
        egc0_encoding(current_length, bw)
        # Encode the value interrupting the run
        egc0_encoding(coefficients_abs[idx], bw)
        # Encode its sign
        sign_bit = 1 if coefficients_s[idx] < 0 else 0
        bw.write_bits(sign_bit, 1)
        idx += 1


def decode_pure_egc(br: BitReader, rows: int, cols: int) -> NDArray[(Any, Any), np.int32]:
    coefficients = np.zeros((rows * cols), np.int32)
    total_coefficients = egc0_decoding(br)

    for idx in range(total_coefficients):
        coefficients[idx] = egc0_decoding(br)
        if coefficients[idx]:
            # Decode also the sign
            sign_bit = br.read(1)
            if sign_bit:
                coefficients[idx] = -coefficients[idx]

    return np.reshape(coefficients, (rows, cols))


def decode_rle_egc(br: BitReader, rows: int, cols: int) -> NDArray[(Any, Any), np.int32]:
    coefficients = np.zeros((rows * cols), np.int32)
    total_coefficients = egc0_decoding(br)

    idx = 0
    while idx < total_coefficients:
        current_length = egc0_decoding(br)
        idx += current_length
        coefficients[idx] = egc0_decoding(br)
        s_bit = br.read(1)
        if s_bit:
            coefficients[idx] = -coefficients[idx]
        idx += 1

    return np.reshape(coefficients, (rows, cols))


def encode_subband(subband: NDArray[(Any, Any, 3), np.int32], is_ll: bool) -> List[NDArray[(Any), np.uint8]]:
    # Determine the number of code blocks for this subband
    rows, cols = subband.shape[0], subband.shape[1]
    components = 3 if len(subband.shape) == 3 else 1

    rows_cb, cols_cb = (rows + code_block_size - 1) // code_block_size, (cols + code_block_size - 1) // code_block_size
    code_block_payload = []

    for r in range(rows_cb):
        row_slice = slice(r * code_block_size, min(rows, (r + 1) * code_block_size))
        for c in range(cols_cb):
            col_slice = slice(c * code_block_size, min(cols, (c + 1) * code_block_size))
            cb = subband[row_slice, col_slice]

            # Initialise a new bit writer for this code block
            bw = BitWriter(np.zeros((code_block_buffer_size), np.uint8))

            for comp in range(components):
                if components > 1:
                    coefficients = cb[:, :, comp].flatten()
                else:
                    coefficients = cb.flatten()
                if is_ll:
                    encode_pure_egc(coefficients, bw)
                else:
                    encode_rle_egc(coefficients, bw)

            # Flush the bit writer and get the number of bytes written
            bw.flush()
            current_size = bw.bytes_written()
            size_array = np.array([np.uint8(b) for b in int(current_size).to_bytes(2, byteorder="little")], np.uint8)
            payload = np.concatenate((size_array, bw.buffer[:bw.current_ptr]))
            code_block_payload.append(payload)

    return code_block_payload


def decode_subband(payload_cbs: List[NDArray[(Any), np.uint8]], rows_sb: int, cols_sb: int, components: int, is_ll: bool) -> NDArray[(Any, Any, 3), np.int32]:
    if components == 1:
        subband = np.zeros((rows_sb, cols_sb), np.int32)
    else:
        subband = np.zeros((rows_sb, cols_sb, components), np.int32)
    rows_cb, cols_cb = (rows_sb + code_block_size - 1) // code_block_size, (cols_sb + code_block_size - 1) // code_block_size

    cb_idx = 0
    for r in range(rows_cb):
        row_slice = slice(r * code_block_size, min(rows_sb, (r + 1) * code_block_size))
        cb_height = row_slice.stop - row_slice.start
        for c in range(cols_cb):
            col_slice = slice(c * code_block_size, min(cols_sb, (c + 1) * code_block_size))
            cb_width = col_slice.stop - col_slice.start

            br = BitReader(payload_cbs[cb_idx])
            for comp in range(components):
                if is_ll:
                    coefficients = decode_pure_egc(br, cb_height, cb_width)
                else:
                    coefficients = decode_rle_egc(br, cb_height, cb_width)
                if components == 1:
                    subband[row_slice, col_slice] = coefficients
                else:
                    subband[row_slice, col_slice, comp] = coefficients
            cb_idx += 1

    return subband
