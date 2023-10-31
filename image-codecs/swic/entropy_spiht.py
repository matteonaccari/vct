'''
Routines to perform entropy encoding and decoding according to the
Set Partitioning In Hierarchical Trees (SPIHT) method

Copyright(c) 2025 Matteo Naccari
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

from typing import List, Tuple

import numpy as np
from nptyping import NDArray, Shape

from bit_io import BitReaderLimited, BitWriterAppend, EndOfParsing

msb_bits, bytes_size = 4, 4


def is_setA_significant(abs_image_levels: List[NDArray[Shape["*, *"], np.int32]],
                        successor_map: NDArray[Shape["*, *, 2"], np.int32],
                        row: int, col: int, threshold: int) -> bool:
    # Check all the descendants of the four offsprings
    rs, cs = successor_map[row, col]
    descendants = [[rs, cs], [rs, cs + 1], [rs + 1, cs], [rs + 1, cs + 1]]
    while descendants:
        r, c = descendants.pop(0)
        if abs_image_levels[r, c] >= threshold:
            return True
        rs, cs = successor_map[r, c]
        if rs == -1 or cs == -1:
            continue
        descendants.insert(0, [rs + 1, cs + 1])
        descendants.insert(0, [rs + 1, cs])
        descendants.insert(0, [rs, cs + 1])
        descendants.insert(0, [rs, cs])

    return False


sb_delta = [[0, 0], [0, 1], [1, 0], [1, 1]]


def is_setB_significant(abs_image_levels: List[NDArray[Shape["*, *"], np.int32]],
                        successor_map: NDArray[Shape["*, *, 2"], np.int32],
                        row: int, col: int, threshold: int) -> bool:
    # Check all the descendants of the descendants, i.e. the elements in set G = D - O
    rs, cs = successor_map[row, col]
    descendants = []
    for delta in sb_delta:
        rg, cg = successor_map[rs + delta[0], cs + delta[1]]
        descendants.append([rg, cg])
        descendants.append([rg, cg + 1])
        descendants.append([rg + 1, cg])
        descendants.append([rg + 1, cg + 1])

    while descendants:
        r, c = descendants.pop(0)
        if abs_image_levels[r, c] >= threshold:
            return True
        rs, cs = successor_map[r, c]
        if rs == -1 or cs == -1:
            continue
        descendants.insert(0, [rs + 1, cs + 1])
        descendants.insert(0, [rs + 1, cs])
        descendants.insert(0, [rs, cs + 1])
        descendants.insert(0, [rs, cs])

    return False


def encode_image_spiht(image_levels: List[NDArray[Shape["*, *, 3"], np.int32]],
                       successor_map: NDArray[Shape["*, *, 2"], np.int32],
                       decomposition_levels, components: int) -> Tuple[NDArray[Shape["3"], np.int32], NDArray[Shape["*"], np.uint8]]:
    # Find the most significant bitplane for each colour component
    bp_max = np.zeros((components), np.int32)
    component_bytes, previous_size = np.zeros((components), np.int32), 0
    for c_idx in range(components):
        current_max = np.max(np.abs(image_levels[:, :, c_idx]))
        bp_max[c_idx] = int(np.log2(current_max)) if current_max else -1

    # Setup the bit writer
    bwa = BitWriterAppend()
    q, s = np.abs(image_levels), -((np.sign(image_levels) - 1) >> 1)

    for c_idx in range(components):
        ll_rows = image_levels[:, :, c_idx].shape[0] >> decomposition_levels
        ll_cols = image_levels[:, :, c_idx].shape[1] >> decomposition_levels

        # Initialise the three lists
        lic, lis, lsc = [], [], []

        for r in range(ll_rows):
            for c in range(ll_cols):
                lic.append([r, c])
                if (r & 1) or (c & 1):
                    lis.append([r, c, 'A'])

        # Write the maximum bitplane value
        bp_to_write = bp_max[c_idx] if bp_max[c_idx] != -1 else 0
        bwa.write_bits(bp_to_write, msb_bits)
        for bp in range(bp_max[c_idx], -1, -1):
            i = 0
            while i < len(lic):
                current_significance = q[lic[i][0], lic[i][1], c_idx] >= (1 << bp)
                bwa.write_bits(current_significance, 1)
                if current_significance:
                    lsc.append([lic[i][0], lic[i][1]])
                    current_sign = s[lic[i][0], lic[i][1], c_idx]
                    bwa.write_bits(current_sign, 1)
                    lic.pop(i)
                    i -= 1
                i += 1

            i = 0
            while i < len(lis):
                if lis[i][2] == 'A':
                    current_significance = is_setA_significant(q[:, :, c_idx], successor_map, lis[i][0], lis[i][1], 1 << bp)
                    bwa.write_bits(current_significance, 1)
                    if current_significance:
                        s_r, s_c = successor_map[lis[i][0], lis[i][1], :]
                        for delta in sb_delta:
                            curr_r, curr_c = s_r + delta[0], s_c + delta[1]
                            current_significance = q[curr_r, curr_c, c_idx] >= (1 << bp)
                            bwa.write_bits(current_significance, 1)
                            if current_significance:
                                lsc.append([curr_r, curr_c])
                                current_sign = s[curr_r, curr_c, c_idx]
                                bwa.write_bits(current_sign, 1)
                            else:
                                lic.append([curr_r, curr_c])
                        g = successor_map[s_r, s_c, :]
                        if g[0] != -1 and g[1] != -1:
                            lis.append([lis[i][0], lis[i][1], 'B'])
                        lis.pop(i)
                        i -= 1
                else:
                    current_significance = is_setB_significant(q[:, :, c_idx], successor_map, lis[i][0], lis[i][1], 1 << bp)
                    bwa.write_bits(current_significance, 1)
                    if current_significance:
                        s_r, s_c = successor_map[lis[i][0], lis[i][1], :]
                        for delta in sb_delta:
                            curr_r, curr_c = s_r + delta[0], s_c + delta[1]
                            lis.append([curr_r, curr_c, 'A'])
                        lis.pop(i)
                        i -= 1
                i += 1
            i = 0
            while i < len(lsc):
                current_significance = q[lsc[i][0], lsc[i][1], c_idx] >= (1 << (bp + 1))
                if current_significance:
                    bit = (q[lsc[i][0], lsc[i][1], c_idx] >> bp) & 1
                    bwa.write_bits(bit, 1)
                i += 1
        bwa.flush()
        current_bytes = len(bwa.buffer)
        component_bytes[c_idx] = current_bytes - previous_size
        previous_size = current_bytes
    return component_bytes, np.array(bwa.buffer, np.uint8)


def compute_successor_map(height: int, width: int, decomposition_levels: int) -> NDArray[Shape["*, *, 2"], np.int32]:
    successor_map = -np.ones((height, width, 2), np.int32)
    rows_ll = height // (1 << decomposition_levels)
    cols_ll = width // (1 << decomposition_levels)

    # High frequency quadrants
    successor_map[:height >> 1, :width >> 1, 0] = np.tile(np.arange(0, height, 2), (width >> 1, 1)).T
    successor_map[:height >> 1, :width >> 1, 1] = np.tile(np.arange(0, width, 2), (height >> 1, 1))

    # LL quadrant
    # Root nodes, no offsprings, i.e. [0, 0] position
    successor_map[:rows_ll:2, :cols_ll:2] = -1
    # Nodes at [0, 1] position
    successor_map[:rows_ll:2, 1:cols_ll:2, 0] = np.tile(np.arange(0, rows_ll, 2), (cols_ll >> 1, 1)).T
    successor_map[:rows_ll:2, 1:cols_ll:2, 1] = np.tile(np.arange(cols_ll, cols_ll * 2, 2), (rows_ll >> 1, 1))
    # Nodes at [1, 0] position
    successor_map[1:rows_ll:2, :cols_ll:2, 0] = np.tile(np.arange(rows_ll, rows_ll * 2, 2), (cols_ll >> 1, 1)).T
    successor_map[1:rows_ll:2, :cols_ll:2, 1] = np.tile(np.arange(0, cols_ll, 2), (rows_ll >> 1, 1))
    # Nodes at [1, 1] position
    successor_map[1:rows_ll:2, 1:cols_ll:2, 0] = np.tile(np.arange(rows_ll, rows_ll * 2, 2), (cols_ll >> 1, 1)).T
    successor_map[1:rows_ll:2, 1:cols_ll:2, 1] = np.tile(np.arange(cols_ll, cols_ll * 2, 2), (rows_ll >> 1, 1))

    return successor_map


def decode_image_spiht(bitstream: NDArray[Shape["*"], np.uint8],
                       successor_map: NDArray[Shape["*, *, 2"], np.int32],
                       decomposition_levels: int, components: int,
                       bits_to_process: int,
                       weighty: float) -> NDArray[Shape["*, *, *"], np.int32]:
    rows, cols = successor_map.shape[0], successor_map.shape[1]
    image_levels = np.zeros((rows, cols, components), np.int32)
    start = 0
    if bits_to_process:
        if not (0 <= weighty and weighty <= 1):
            raise Exception("Weight for luma must be in the range [0, 1] inclusive")
        weightc = (1 - weighty) / 2
        bits_to_decode = [int(bits_to_process * weighty + 0.5), int(bits_to_process * weightc + 0.5), int(bits_to_process * weightc + 0.5)]
    else:
        max_bits = rows * cols * 16 + msb_bits
        bits_to_decode = [max_bits] * components

    for c_idx in range(components):
        # Read the payload size for this component
        stop = int(bitstream[start]) | (int(bitstream[start + 1]) << 8) | (int(bitstream[start + 2]) << 16) | (int(bitstream[start + 3]) << 24)
        # Extract this component's payload
        br = BitReaderLimited(bitstream[start + bytes_size:start + bytes_size + stop])
        start += stop + bytes_size
        br.set_bit_limit(bits_to_decode[c_idx])
        try:
            ll_rows = rows >> decomposition_levels
            ll_cols = cols >> decomposition_levels
            current_plane = np.zeros((rows, cols), np.int32)
            current_sign = np.zeros((rows, cols), np.int32)

            # Initialise the three lists
            lic, lis, lsc = [], [], []

            for r in range(ll_rows):
                for c in range(ll_cols):
                    lic.append([r, c])
                    if (r & 1) or (c & 1):
                        lis.append([r, c, 'A'])

            # Read the maximum bitplane value for this colour component
            bp_max = br.read(msb_bits)
            for bp in range(bp_max, -1, -1):
                i = 0
                while i < len(lic):
                    current_significance = br.read(1)
                    if current_significance:
                        lsc.append([lic[i][0], lic[i][1]])
                        current_sign[lic[i][0], lic[i][1]] = br.read(1)
                        current_plane[lic[i][0], lic[i][1]] = 1 << bp
                        lic.pop(i)
                        i -= 1
                    i += 1

                i = 0
                while i < len(lis):
                    if lis[i][2] == 'A':
                        current_significance = br.read(1)
                        if current_significance:
                            s_r, s_c = successor_map[lis[i][0], lis[i][1], :]
                            for delta in sb_delta:
                                curr_r, curr_c = s_r + delta[0], s_c + delta[1]
                                current_significance = br.read(1)
                                if current_significance:
                                    lsc.append([curr_r, curr_c])
                                    current_sign[curr_r, curr_c] = br.read(1)
                                    current_plane[curr_r, curr_c] = 1 << bp
                                else:
                                    lic.append([curr_r, curr_c])
                            g = successor_map[s_r, s_c, :]
                            if g[0] != -1 and g[1] != -1:
                                lis.append([lis[i][0], lis[i][1], 'B'])
                            lis.pop(i)
                            i -= 1
                    else:
                        current_significance = br.read(1)
                        if current_significance:
                            s_r, s_c = successor_map[lis[i][0], lis[i][1], :]
                            for delta in sb_delta:
                                lis.append([s_r + delta[0], s_c + delta[1], 'A'])
                            lis.pop(i)
                            i -= 1
                    i += 1
                i = 0
                while i < len(lsc):
                    current_significance = np.abs(current_plane[lsc[i][0], lsc[i][1]]) >= (1 << (bp + 1))
                    if current_significance:
                        bit = br.read(1)
                        current_plane[lsc[i][0], lsc[i][1]] |= (bit << bp)
                    i += 1
        except EndOfParsing:
            current_sign = -(current_sign << 1) + 1
            current_plane *= current_sign
            if components == 1:
                image_levels = current_plane
            else:
                image_levels[:, :, c_idx] = current_plane
        else:
            current_sign = -(current_sign << 1) + 1
            current_plane *= current_sign
            if components == 1:
                image_levels = current_plane
            else:
                image_levels[:, :, c_idx] = current_plane

    return image_levels
