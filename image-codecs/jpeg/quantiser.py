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
from entropy import encode_block

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
class RunLenghtValueInfo:
    category: int
    run_length: int
    start_idx: int
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
                   pred_dc: int) -> Tuple[NDArray[(8, 8), np.int32], float, int]:
    # Regular quantisation
    s = np.sign(coefficients).astype(np.int32)
    levels = np.divide(coefficients + 0.5, qm).astype(np.int32)
    levels_m1 = np.multiply(s, np.abs(levels) - 1)
    distortion_coeff = np.square(coefficients - levels.astype(np.float64) * qm)
    distortion_coeff_m1 = np.square(coefficients - levels_m1.astype(np.float64) * qm)
    distortion_no_opt = np.sum(distortion_coeff)
    _, rate_no_opt = encode_block(levels.flatten()[zz_idx], pred_dc, table_dc, table_ac)
    rd_cost_no_opt = distortion_no_opt + _lambda * rate_no_opt

    # Coefficient-based rate and distortion storage
    distortion0 = np.square(coefficients)
    rate_coeff = np.zeros((64), np.int32)
    rate_coeff_m1 = np.zeros((64), np.int32)

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
    idx = np.where(levels_zz[-1:0:-1] != 0)
    last_sig_coeff = 63 - idx[0][0] if len(idx[0]) else 0

    # RDO for the AC coefficients
    distortion0 = distortion0.flatten()[zz_idx]
    distortion_coeff = distortion_coeff.flatten()[zz_idx]
    distortion_coeff_m1 = distortion_coeff_m1.flatten()[zz_idx]
    levels_zz_m1 = levels_m1.flatten()[zz_idx]

    # Step 1: Find run length value pairs and their RD cost
    rate15_0 = table_ac[15 << 4, 1]
    rate_eob = table_ac[0, 1]
    rdoq_memory = []
    i = 1
    while i <= last_sig_coeff:
        current_info = RunLenghtValueInfo(0, 0, i, 0.0, 0)
        while not levels_zz[i]:
            current_info.run_length += 1
            current_info.distortion += distortion0[i]
            i += 1
        current_info.distortion += distortion_coeff[i]
        distortion += current_info.distortion
        current_info.category = int(np.ceil(np.log2(np.abs(levels_zz[i]) + 1)))
        rlv_pair = ((current_info.run_length & 15) << 4) | current_info.category
        rate_coeff[i] = table_ac[rlv_pair, 1] + current_info.category + (current_info.run_length >> 4) * rate15_0
        current_info.rate += rate_coeff[i]
        rate += rate_coeff[i]
        rdoq_memory.append(current_info)
        if np.abs(levels_zz[i]) > 1:
            category_m1 = int(np.ceil(np.log2(np.abs(levels_zz_m1[i]) + 1)))
            rlv_pair = ((current_info.run_length & 15) << 4) | category_m1
            rate_coeff_m1[i] = table_ac[rlv_pair, 1] + category_m1 + (current_info.run_length >> 4) * rate15_0
        i += 1

    # Step 2: Scan through all run length value pairs and check if adjacent ones can be merged
    i = 0
    while i < len(rdoq_memory) - 1:
        e_curr = rdoq_memory[i]
        end_run = e_curr.start_idx + e_curr.run_length
        if np.abs(levels_zz[end_run]) > 1:
            cost_q = distortion_coeff[end_run] + _lambda * rate_coeff[end_run]
            cost_q_m1 = distortion_coeff_m1[end_run] + _lambda * rate_coeff_m1[end_run]
            if cost_q_m1 < cost_q:
                e_curr.distortion += distortion_coeff_m1[end_run] - distortion_coeff[end_run]
                e_curr.rate += rate_coeff_m1[end_run] - rate_coeff[end_run]
                e_curr.category = int(np.ceil(np.log2(np.abs(levels_zz_m1[end_run]) + 1)))
                levels_zz[end_run] = levels_zz_m1[end_run]
                distortion += distortion_coeff_m1[end_run] - distortion_coeff[end_run]
                rate += rate_coeff_m1[end_run] - rate_coeff[end_run]
                distortion_coeff[end_run] = distortion_coeff_m1[end_run]
                rate_coeff[end_run] = rate_coeff_m1[end_run]

        e_next = rdoq_memory[i + 1]
        distortion_singletons = e_curr.distortion + e_next.distortion
        rate_singletons = e_curr.rate + e_next.rate
        rd_cost_singletons = distortion_singletons + _lambda * rate_singletons

        distortion_merge = e_curr.distortion - distortion_coeff[end_run] + distortion0[end_run] + e_next.distortion
        new_run_length = e_curr.run_length + e_next.run_length + 1
        merge_rlv_pair = ((new_run_length & 15) << 4) | e_next.category
        rate_merge = table_ac[merge_rlv_pair, 1] + e_next.category + (new_run_length >> 4) * rate15_0
        rd_cost_merge = distortion_merge + _lambda * rate_merge

        if rd_cost_merge < rd_cost_singletons:
            levels_zz[end_run] = 0
            distortion_coeff[end_run] = distortion0[end_run]
            rate_coeff[end_run] = 0
            rate_coeff[end_run + e_next.run_length + 1] = rate_merge
            distortion += distortion_merge - distortion_singletons
            rate += rate_merge - rate_singletons
            if np.abs(levels_zz[end_run + e_next.run_length + 1]) > 1:
                category_m1 = int(np.ceil(np.log2(np.abs(levels_zz_m1[end_run + e_next.run_length + 1]) + 1)))
                merge_rlv_pair = ((new_run_length & 15) << 4) | category_m1
                rate_merge_m1 = table_ac[merge_rlv_pair, 1] + category_m1 + (new_run_length >> 4) * rate15_0
                rate_coeff_m1[end_run + e_next.run_length + 1] = rate_merge_m1

            e_next.distortion = distortion_merge
            e_next.rate = rate_merge
            e_next.start_idx = e_curr.start_idx
            e_next.run_length = new_run_length
            rdoq_memory[i + 1] = e_next

        i += 1

    # Add the distortion for those pixels beyond the last significant coefficient
    if last_sig_coeff != 63:
        distortion += np.sum(distortion_coeff[last_sig_coeff + 1:])

    # Re-check rate and distortion calculations
    _, rate_now = encode_block(levels_zz, pred_dc, table_dc, table_ac)
    levels_now = np.reshape(levels_zz[r_zz_idx], (8, 8))
    distortion_now = np.sum(np.square(coefficients - levels_now * qm))

    if last_sig_coeff != 63:
        assert rate_now == rate + rate_eob
    else:
        assert rate_now == rate
    assert rate == np.sum(rate_coeff)

    assert np.isclose(distortion_now, distortion)

    # Step 3: Move the EOB towards the top left corner
    i = last_sig_coeff
    best_eob = last_sig_coeff + 1
    rate += rate_eob
    distortion_now, rate_now = distortion, rate
    cost_best = distortion + _lambda * rate
    if last_sig_coeff == 63:
        cost_best -= _lambda * rate_eob

    while i > 0:
        if levels_zz[i]:
            # Adjust the rate and distortion
            distortion_now += distortion0[i] - distortion_coeff[i]
            rate_now -= rate_coeff[i]
            cost_now = distortion_now + _lambda * rate_now
            if cost_now < cost_best:
                best_eob = i
                cost_best = cost_now
        i -= 1

    levels_zz[best_eob:] = 0

    levels_rdoq = np.reshape(levels_zz[r_zz_idx], (8, 8))
    _, rate = encode_block(levels_zz, pred_dc, table_dc, table_ac)
    distortion = np.sum(np.square(coefficients - levels_rdoq.astype(np.float64) * qm))
    rd_cost = distortion + _lambda * rate
    if not rd_cost < rd_cost_no_opt:
        assert np.isclose(rd_cost, rd_cost_no_opt)

    return levels_rdoq, distortion, rate
