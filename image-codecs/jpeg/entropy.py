'''
Helper functions to handle run length encoding with alphabet extension and category codes
as specified by the JPEG standard: https://www.itu.int/rec/T-REC-T.81-199209-I/en.

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

from typing import Any, Dict, List, Tuple

import numpy as np
from nptyping import NDArray

# Luma Huffman tables
luma_dc_bits = np.array([0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], np.int32)
luma_dc_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], np.int32)
luma_ac_bits = np.array([0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125], np.int32)
luma_ac_values = np.array([0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
                           0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
                           0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
                           0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
                           0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
                           0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
                           0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA], np.int32)

# Chroma Huffman table
chroma_dc_bits = np.array([0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], np.int32)
chroma_dc_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], np.int32)
chroma_ac_bits = np.array([0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119], np.int32)
chroma_ac_values = np.array([0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
                             0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33, 0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26,
                             0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
                             0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
                             0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
                             0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA,
                             0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA], np.int32)


def expand_huffman_table(bits: NDArray[(Any), np.int32], values: NDArray[(Any), np.int32]) -> NDArray[(256, 2), np.int32]:
    if len(values) != np.sum(bits):
        raise Exception("Huffman table provided in the wrong format")

    htable = -1 * np.ones((256, 2), np.int32)
    cw, idx_value = 0, 0

    for idx, b in enumerate(bits):
        if b:
            for _ in range(b):
                htable[values[idx_value], :] = [cw, idx + 1]
                cw += 1
                idx_value += 1
        cw <<= 1

    return htable


def get_zigzag_scan(block_size: int) -> Tuple[NDArray[(Any, Any), np.uint8], NDArray[(Any, Any), np.uint8]]:
    scan_idx = np.zeros((block_size, block_size), np.uint8)
    scan_idx_inv = np.zeros((block_size, block_size), np.uint8)

    going_down = True
    idx, r, c = 1, 0, 1
    bound = 2

    # Upper triangle
    while idx < (block_size**2) >> 1:
        if going_down:
            for _ in range(bound, 0, -1):
                scan_idx[r, c] = idx
                scan_idx_inv[idx // block_size, idx % block_size] = r * block_size + c
                idx += 1
                c = max(0, c - 1)
                r = min(block_size - 1, r + 1)
            going_down = False
        else:
            for _ in range(bound, 0, -1):
                scan_idx[r, c] = idx
                scan_idx_inv[idx // block_size, idx % block_size] = r * block_size + c
                idx += 1
                c = min(block_size - 1, c + 1)
                r = max(0, r - 1)
            going_down = True
        bound += 1 if idx < (block_size**2) >> 1 else -1

    # Lower triangle
    if block_size % 2:
        r, c = 1, block_size - 1
    else:
        r, c = block_size - 1, 1
    while idx < block_size**2:
        if going_down:
            for _ in range(bound, 0, -1):
                scan_idx[r, c] = idx
                scan_idx_inv[idx // block_size, idx % block_size] = r * block_size + c
                idx += 1
                c = max(0, c - 1)
                r = min(block_size - 1, r + 1)
            going_down = False
            c = min(block_size - 1, c + 2)
        else:
            for _ in range(bound, 0, -1):
                scan_idx[r, c] = idx
                scan_idx_inv[idx // block_size, idx % block_size] = r * block_size + c
                idx += 1
                c = min(block_size - 1, c + 1)
                r = max(0, r - 1)
            going_down = True
            r = min(r + 2, block_size - 1)
        bound -= 1

    return scan_idx, scan_idx_inv


def encode_block(block: NDArray[(Any, Any), np.int32], pred_dc, dch: NDArray[(256, 2), np.int32], ach: NDArray[(256, 2), np.int32]) -> Tuple[Dict, int]:
    res_dc = block[0] - pred_dc
    cw_dict = {}
    cw_list = []

    # Determine the codewords for the DC coefficient
    if not res_dc:
        cw_list.append([dch[0, 0], dch[0, 1]])
        rate = dch[0, 1]
    else:
        # Find the minimum number of bits required to represent the DC residual (i.e. the category)
        # VLC the category value with the Huffman table whilst use Fixed Length Coding (FLC) for the
        # remainder. The remainder also carries the sign bit according to the convention: v < 0 -> 0
        # v > 0 -> 1
        res_dc_category = int(np.ceil(np.log2(np.abs(res_dc) + 1)))
        cw_list.append([dch[res_dc_category, 0], dch[res_dc_category, 1]])
        rate = dch[res_dc_category, 1]
        flc = (1 << res_dc_category) - 1 + res_dc if res_dc < 0 else res_dc
        cw_list.append([flc, res_dc_category])
        rate += res_dc_category

    cw_dict["DC"] = cw_list
    cw_list = []

    # Find the position of the last significant coefficient
    idx = np.where(block[-1:0:-1] != 0)
    last_sig_coeff = 63 - idx[0][0] if len(idx[0]) else 0

    # Run length encoding with alphabet extension of the run length value pairs.
    # The length of the run is packed in the MSBs of a 8 bit codeword, hence it cannot exceed the value 15.
    # When this happens, the special codeword 240 = 15 << 4 is written in the bitstream and the run length is reset.
    # The LSBs of the codeword carry the category of the significant coefficient. Remainder bits are written as for the DC's case.
    # If the last significant coefficient is different from 63, the End Of Block (EOB) codeword is written, i.e. 0
    run_length = 0
    i = 1
    while i <= last_sig_coeff:
        while not block[i]:
            run_length += 1
            if run_length > 15:
                cw_list.append([ach[15 << 4, 0], ach[15 << 4, 1]])
                rate += ach[15 << 4, 1]
                run_length = 0
            i += 1
        ac_category = int(np.ceil(np.log2(np.abs(block[i]) + 1)))
        rlv_pair = (run_length << 4) | ac_category
        cw_list.append([ach[rlv_pair, 0], ach[rlv_pair, 1]])
        flc = (1 << ac_category) - 1 + block[i] if block[i] < 0 else block[i]
        cw_list.append([flc, ac_category])
        rate += ach[rlv_pair, 1] + ac_category
        run_length = 0
        i += 1
    if last_sig_coeff != 63:
        cw_list.append([ach[0, 0], ach[0, 1]])
        rate += ach[0, 1]

    cw_dict["AC"] = cw_list

    return cw_dict, rate


def get_block_symbols(block: NDArray[(Any, Any), np.int32], pred_dc) -> Tuple[int, List[int]]:
    res_dc = block[0] - pred_dc

    # Determine the category for the DC coefficient
    if not res_dc:
        dc_sym = 0
    else:
        dc_sym = int(np.ceil(np.log2(np.abs(res_dc) + 1)))

    # Find the position of the last significant coefficient
    idx = np.where(block[-1:0:-1] != 0)
    last_sig_coeff = 63 - idx[0][0] if len(idx[0]) else 0

    # Determine the run length value pairs using alphabet extension.
    # The length of the run is packed in the MSBs of a 8 bit number, hence it cannot exceed the value 15.
    # When this happens, the special symbol 240 = 15 << 4 is used.
    # The LSBs of the symbol carry the category of the significant coefficient.
    # If the last significant coefficient is different from 63, the End Of Block (EOB) symbol is also added (i.e. 0)
    ac_sym = []
    run_length = 0
    i = 1
    while i <= last_sig_coeff:
        while not block[i]:
            run_length += 1
            if run_length > 15:
                ac_sym.append(15 << 4)
                run_length = 0
            i += 1
        ac_category = int(np.ceil(np.log2(np.abs(block[i]) + 1)))
        rlv_pair = (run_length << 4) | ac_category
        ac_sym.append(rlv_pair)
        run_length = 0
        i += 1
    if last_sig_coeff != 63:
        ac_sym.append(0)

    return dc_sym, ac_sym


def limit_codewords_length(bits_array: NDArray[(32), np.int32]) -> NDArray[(32), np.int32]:
    i = 32
    while i > 16:
        while(bits_array[i] > 0):
            j = i - 2
            while (bits_array[j] == 0):
                j -= 1
            bits_array[i] -= 2
            bits_array[i - 1] += 1
            bits_array[j + 1] += 2
            bits_array[j] -= 1
        i -= 1

    while bits_array[i] == 0:
        i -= 1
    bits_array[i] -= 1
    return bits_array


def derive_huffman_table(freq: NDArray[(257), np.int32]) -> Tuple[NDArray[(33), np.int32], NDArray[(257), np.int32]]:
    code_size = np.zeros((257), np.int32)
    others = -1 * np.ones((257), np.int32)

    while True:
        # Find V1 for least value of freq(V1) > 0
        min_value = np.iinfo(np.int32).max
        v1 = -1
        for idx, entry in enumerate(freq):
            if entry and entry <= min_value:
                min_value = entry
                v1 = idx

        # Find V2 for next least value of freq(V2) > 0
        min_value = np.iinfo(np.int32).max
        v2 = -1
        for idx, entry in enumerate(freq):
            if entry and entry <= min_value and idx != v1:
                min_value = entry
                v2 = idx

        if v2 == -1:
            break

        freq[v1] += freq[v2]
        freq[v2] = 0

        code_size[v1] += 1

        while others[v1] != -1:
            v1 = others[v1]
            code_size[v1] += 1

        others[v1] = v2
        code_size[v2] += 1

        while others[v2] != -1:
            v2 = others[v2]
            code_size[v2] += 1

    # Find the bits array
    bits = np.zeros((33), np.int32)
    for entry in code_size:
        if entry:
            bits[entry] += 1

    return bits, code_size


def sort_input(code_size: NDArray[(257), np.int32]) -> NDArray[(Any), np.int32]:
    values = []
    for i in range(1, 33):
        for idx in range(256):
            if code_size[idx] == i:
                values.append(idx)

    return np.array(values, np.int32)


def design_huffman_table(symbols: List[int]) -> Tuple[NDArray[(32), np.int32], NDArray[(Any), np.int32]]:
    # Declaration of arrays freq, others and code_size
    frequency_table = np.zeros((257), np.int32)
    frequency_table[-1] = 1

    # Derive the frequency table
    for s in symbols:
        frequency_table[s] += 1

    # Derive the Huffman code as bits array
    bits, code_size = derive_huffman_table(frequency_table)

    # Limit the length of codewords to 16 bits (if needed)
    bits = limit_codewords_length(bits)

    # Sort the input
    values = sort_input(code_size)

    return bits[1:17], np.array(values, np.int32)
