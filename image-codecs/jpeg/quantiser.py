'''
Helper functions to handle uniform quantisation of the DCT coefficients with the default tables
reported in the JPEG standard specification: https://www.itu.int/rec/T-REC-T.81-199209-I/en.

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

from typing import Tuple

import numpy as np
from nptyping import NDArray
from dataclasses import dataclass

# Quantisation tables for luma and chroma components, see Section K.1 of the spec.
luma_quantisation_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                     [12, 12, 14, 19, 26, 58, 60, 55],
                                     [14, 13, 16, 24, 40, 57, 69, 56],
                                     [14, 17, 22, 29, 51, 87, 80, 62],
                                     [18, 22, 37, 56, 68, 109, 103, 77],
                                     [24, 35, 55, 64, 81, 104, 113, 92],
                                     [49, 64, 78, 87, 103, 121, 120, 101],
                                     [72, 92, 95, 98, 112, 100, 103, 99]], np.int32)

chroma_quantisation_matrix = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                       [18, 21, 26, 66, 99, 99, 99, 99],
                                       [24, 26, 56, 99, 99, 99, 99, 99],
                                       [47, 66, 99, 99, 99, 99, 99, 99],
                                       [99, 99, 99, 99, 99, 99, 99, 99],
                                       [99, 99, 99, 99, 99, 99, 99, 99],
                                       [99, 99, 99, 99, 99, 99, 99, 99],
                                       [99, 99, 99, 99, 99, 99, 99, 99]], np.int32)


@dataclass
class RdoqInfo:
    category: int
    run_length: int
    start_idx: int
    hit15: int
    distortion: float
    rate: int


def compute_quantisation_matrices(quality: int) -> Tuple[NDArray[(8, 8), np.int32], NDArray[(8, 8), np.int32]]:
    # Adjust the quality value according to the relationship worked out by the Indipendent JPEG Group (IJG)
    quality = 5000 // quality if quality < 50 else 200 - quality * 2

    q_luma = np.clip((luma_quantisation_matrix * quality + 50) // 100, 1, 255)
    q_chroma = np.clip((chroma_quantisation_matrix * quality + 50) // 100, 1, 255)

    return q_luma, q_chroma


def rdoq_8x8_plane(coefficients: NDArray[(8, 8), np.float64], qm: NDArray[(8, 8), np.float64],
                   table_dc: NDArray[(256, 2), np.int32], table_ac: NDArray[(256, 2), np.int32],
                   _lambda: float, zz_idx: NDArray[(8, 8), np.uint8], r_zz_idx: NDArray[(8, 8), np.uint8],
                   pred_dc: int) -> Tuple[NDArray[(8, 8), np.int32], float]:
    # Regular quantisation
    levels = np.divide(coefficients + 0.5, qm).astype(np.int32)

    # Coefficient-based rate and distortion storage
    distortion0 = np.square(coefficients)
    distortion_coeff = distortion0.copy()
    rate_coeff = np.zeros((64), np.int32)

    # Apply DPCM for the DC coefficient
    res_dc = levels[0, 0] - pred_dc

    # Optimise its rate and distortion
    if not res_dc:
        # Zero value residual, no quantisation distortion, just add the rate
        rate = table_dc[0, 1]
        rate_coeff[0] = table_dc[0, 1]
        distortion = (coefficients[0, 0] - levels[0, 0] * qm[0, 0])**2
        distortion_coeff[0, 0] = distortion
    else:
        # Perform RDO: test the current DC value, zero and its absolute value minus one
        best_rd_cost = np.finfo(np.float64).max
        sign_dc = np.sign(levels[0, 0])
        abs_dc = np.abs(levels[0, 0])
        candidates = np.unique([0, abs_dc, abs_dc - 1])
        for current_lev in candidates:
            current_dist = (coefficients[0, 0] - sign_dc * current_lev * qm[0, 0])**2
            res_dc = sign_dc * current_lev - pred_dc
            if res_dc:
                res_dc_category = int(np.ceil(np.log2(np.abs(res_dc) + 1)))
                current_rate = table_dc[res_dc_category, 1] + res_dc_category
            else:
                current_rate = table_dc[0, 1]
            rd_cost_current = current_dist + _lambda * current_rate
            if rd_cost_current < best_rd_cost:
                best_rd_cost = rd_cost_current
                rate = current_rate
                rate_coeff[0] = current_rate
                distortion = current_dist
                levels[0, 0] = sign_dc * current_lev
                distortion_coeff[0, 0] = current_dist

    # Find the eob
    levels_zz = levels.flatten()[zz_idx]
    qm_zz = qm.flatten()[zz_idx]
    coefficients_zz = coefficients.flatten()[zz_idx]
    idx = np.where(levels_zz[-1:0:-1] != 0)
    last_sig_coeff = 63 - idx[0][0] if len(idx[0]) else 0

    # RDO for the AC coefficients
    distortion0 = distortion0.flatten()[zz_idx]
    distortion_coeff = distortion_coeff.flatten()[zz_idx]

    # Step 1: Find run length value pairs and their RD cost
    rate15 = table_ac[15 << 4, 1]
    rdoq_memory = []
    i = 1
    while i <= last_sig_coeff:
        current_info = RdoqInfo(0, 0, i, 0, 0.0, 0)
        while not levels_zz[i]:
            current_info.run_length += 1
            current_info.distortion += distortion0[i]
            i += 1
            if current_info.run_length > 15:
                current_info.run_length = 0
                current_info.hit15 += 1
                current_info.rate += rate15
                rate += rate15
        distortion_coeff[i] = (coefficients_zz[i] - levels_zz[i] * qm_zz[i])**2
        current_info.distortion += distortion_coeff[i]
        distortion += current_info.distortion
        current_info.category = int(np.ceil(np.log2(np.abs(levels_zz[i]) + 1)))
        rlv_pair = (current_info.run_length << 4) | current_info.category
        rate_coeff[i] = table_ac[rlv_pair, 1] + current_info.category
        current_info.rate += rate_coeff[i]
        rate += rate_coeff[i]
        rdoq_memory.append(current_info)
        i += 1

    # Step 2: Scan through all run length value pairs and check if adjacent ones can be merged
    i = 0
    while i < len(rdoq_memory) - 1:
        e = rdoq_memory[i]
        e_next = rdoq_memory[i + 1]
        distortion_singletons = e.distortion + e_next.distortion
        rate_singletons = e.rate + e_next.rate
        rd_cost_singletons = distortion_singletons + _lambda * rate_singletons

        full_length = e.run_length + e.hit15 * 15
        full_length_next = e_next.run_length + e_next.hit15 * 15
        end_run = e.start_idx + full_length
        distortion_merge = e.distortion - distortion_coeff[end_run] + distortion0[end_run] + e_next.distortion
        new_run_length = (full_length + full_length_next + 1) % 15
        new_hit15 = (full_length + full_length_next + 1) // 15
        merge_rlv_pair = (new_run_length << 4) | e_next.category
        rate_merge = table_ac[merge_rlv_pair, 1] + e_next.category + (e.hit15 + e_next.hit15 + new_hit15) * rate15
        rd_cost_merge = distortion_merge + _lambda * rate_merge

        if rd_cost_merge <= rd_cost_singletons:
            levels_zz[end_run] = 0
            distortion_coeff[end_run] = distortion0[end_run]
            rate_coeff[end_run] = 0
            rate_coeff[end_run + full_length_next + 1] = rate_merge
            distortion += distortion_merge - distortion_singletons
            rate += rate_merge - rate_singletons
            e_next.distortion = distortion_merge
            e_next.rate = rate_merge
            e_next.start_idx = e.start_idx
            e_next.hit15 = ((e.hit15 + e_next.hit15 + new_hit15) * 15 + 1) // 15
            e_next.run_length = new_run_length
            rdoq_memory[i + 1] = e_next

        i += 1

    assert rate == np.sum(rate_coeff)
    assert np.isclose(distortion, np.sum(distortion_coeff[:last_sig_coeff + 1]))

    # Step 3: Move the EOB towards the top left corner
    i = last_sig_coeff
    best_eob = last_sig_coeff + 1
    rate_eob = table_ac[0, 1]
    distortion_now, rate_now = distortion, rate
    cost_best = distortion + _lambda * rate
    while i > 0:
        # Adjust the cost
        cost_now = (distortion_now - distortion_coeff[i] + distortion0[i]) + _lambda * (rate_now - rate_coeff[i] + rate_eob)
        if cost_now <= cost_best:
            best_eob = i + 1
            cost_best = cost_now
            distortion, rate = distortion_now, rate_now
        distortion_now += distortion0[i] - distortion_coeff[i]
        rate_now -= rate_coeff[i]
        i -= 1

    levels_zz[best_eob:] = 0

    if best_eob != 64:
        rate += rate_eob

    # Finish off the distortion calculation for all coefficients beyond the EOB (if any)
    for i in range(last_sig_coeff + 1, 64, 1):
        distortion += coefficients_zz[i]**2

    rd_cost = distortion + _lambda * rate
    levels_rdoq = np.reshape(levels_zz[r_zz_idx], (8, 8))
    return levels_rdoq, rd_cost
