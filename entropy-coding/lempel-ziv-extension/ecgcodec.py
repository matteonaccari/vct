'''
Routines implementing the extension of the Lempel-Ziv algorithm to lossy compression
of ECG signals.

See R. N. Horspool and W. J. Windels, "An LZ Approach to ECG Compression", in Proceedings of the IEEE
Symposium on Computer-Based Medical Systems (CBMS), Winston-Salem, NC, USA, June 1994.

Copyright(c) 2021 Matteo Naccari
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

from typing import Any, List, Tuple

import numpy as np
from bitio import BitReader, BitWriter
from nptyping import NDArray


def ecgencoder(input_data: NDArray[(Any), np.int32],
               buffer: NDArray[(Any), np.int32],
               tolerance: int, min_match: int = 20, max_match: int = 255) -> Tuple[List[List[int]], NDArray[(Any), np.int32], int]:
    if min_match > buffer.size:
        raise Exception(f"No dictionary based compression can be performed given that the buffer size {buffer.size} is less than min match length {min_match}")

    # Support variables and buffers
    cb = buffer.size
    buffer[::] = -1  # Set to impossible initial value
    decoded = np.zeros(input_data.shape, np.uint8)
    ptr_input, ptr_decoded, ptr_buffer = 0, 0, 0
    input_bits = 8
    sym_bitstream = []
    total_raw, total_pairs, total_bits = 0, 0, 8
    bits_position_length = int(np.ceil(np.log2(cb)) + np.ceil(np.log2(max_match)))

    # Fill the buffer with as many samples as the min_match parameter
    for _ in range(min(min_match, input_data.size)):
        buffer[ptr_buffer] = input_data[ptr_input]
        decoded[ptr_decoded] = input_data[ptr_input]
        total_bits += 1 + input_bits
        sym_bitstream.append([0, [input_data[ptr_input]]])
        ptr_input += 1
        ptr_buffer += 1
        ptr_decoded += 1
        total_raw += 1

    while ptr_input < input_data.size:
        # Determine the extent of the current match
        look_ahead = min(max_match, ptr_buffer)

        # Extract the input segment
        segment_input = input_data[ptr_input:min(input_data.size, ptr_input + look_ahead)]

        if len(segment_input) >= min_match:
            position, length = match_input_in_buffer(segment_input, buffer, tolerance, min_match)

        if len(segment_input) >= min_match and length >= min_match:
            # Output the position-length pair
            sym_bitstream.append([1, [position, length]])
            total_pairs += 1
            total_bits += 1 + bits_position_length
            ptr_input += length
            for i in range(length):
                decoded[ptr_decoded] = buffer[ptr_buffer % cb] = buffer[(position + i) % cb]
                ptr_decoded += 1
                ptr_buffer += 1
        else:
            sym_bitstream.append([0, [input_data[ptr_input]]])
            total_bits += 1 + input_bits
            total_raw += 1
            decoded[ptr_decoded] = buffer[ptr_buffer % cb] = input_data[ptr_input]
            ptr_input += 1
            ptr_decoded += 1
            ptr_buffer += 1

    return sym_bitstream, decoded, total_bits


def match_input_in_buffer(input: NDArray[(Any), np.int32], buffer: NDArray[(Any), np.int32],
                          tolerance: int, min_match: int) -> Tuple[int, int]:
    # Default values for position and length
    match_position, match_length, cb = -1, 0, buffer.size

    # Find the starting point of the match, if any
    d = np.abs(buffer - input[0])

    match_start = np.where(d <= tolerance)[0]

    if len(match_start):
        for idx in match_start:
            for i in range(1, input.size):
                if buffer[(idx + i) % cb] < 0:
                    break
                if np.abs(buffer[(idx + i) % cb] - input[i]) > tolerance:
                    break
            if i > match_length:
                match_length = i
                match_position = idx

    return match_position, match_length


def ecgdecoder(compressed_buffer: NDArray[(Any), np.uint8]) -> NDArray[(Any), np.uint8]:
    match_length_bits = compressed_buffer[0] & 0x0F
    position_bits = (compressed_buffer[0] >> 4) & 0x0F
    cb = (1 << position_bits)
    br = BitReader(compressed_buffer[1:])
    buffer = np.zeros(cb, np.uint8)
    decoded = []
    ptr_buffer, input_bits = 0, 8

    while not br.eob():
        code_type = br.read(1)
        if not code_type:
            # Raw value
            buffer[ptr_buffer % cb] = br.read(input_bits)
            decoded.append(buffer[ptr_buffer % cb])
            ptr_buffer += 1
        else:
            # Position length pair
            position = br.read(position_bits)
            length = br.read(match_length_bits)
            j = position
            for _ in range(length):
                decoded.append(buffer[j % cb])
                buffer[ptr_buffer % cb] = buffer[j % cb]
                j += 1
                ptr_buffer += 1

    return np.array(decoded, np.uint8)


def write_compressed_egc(compressed_data: List[List[int]], bitstream_name: str, position_bits: int, match_length_bits: int) -> Tuple[int, int]:
    first_byte = ((position_bits & 0x0F) << 4) | (match_length_bits & 0x0F)
    input_bits = 8
    bw = BitWriter()

    for code_type in compressed_data:
        if not code_type[0]:
            # Raw value
            bw.write(0, 1)
            bw.write(code_type[1][0], input_bits)
        else:
            # Position, length pair
            bw.write(1, 1)
            bw.write(code_type[1][0], position_bits)
            bw.write(code_type[1][1], match_length_bits)

    with open(bitstream_name, 'wb') as fh:
        fh.write(first_byte.to_bytes(1, 'little'))
        bw.flush()
        data_buffer = bw.get_data_buffer()
        fh.write(data_buffer.tobytes())

    return bw.get_bits_written(), bw.get_bytes_written()
