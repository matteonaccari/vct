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
        # Perform RDO
        best_rd_cost = np.finfo(np.float64).max
        sign_dc = np.sign(levels[0, 0])
        for step in range(0, 2, 1):
            current_lev = np.abs(levels[0, 0]) - step
            current_dist = (coefficients[0, 0] - sign_dc * current_lev * qm[0, 0])**2
            res_dc = sign_dc * current_lev - pred_dc
            if res_dc:
                res_dc_category = int(np.ceil(np.log2(np.abs(res_dc) + 1)))
                current_rate = table_dc[res_dc_category, 1] + res_dc_category
            else:
                current_rate = table_dc[0, 1]
            current_rd_cost = current_dist + _lambda * current_rate
            if current_rd_cost < best_rd_cost:
                best_rd_cost = current_rd_cost
                rate = current_rate
                rate_coeff[0] = current_rate
                distortion = current_dist
                levels[0, 0] = sign_dc * current_lev
                distortion_coeff[0, 0] = current_dist

    distortion0 = distortion0.flatten()[zz_idx]
    distortion_coeff = distortion_coeff.flatten()[zz_idx]

    # Find the eob and perform AC coefficients rate calculation
    levels_zz = levels.flatten()[zz_idx]
    qm_zz = qm.flatten()[zz_idx]
    coefficients_zz = coefficients.flatten()[zz_idx]
    idx = np.where(levels_zz[-1:0:-1] != 0)
    last_sig_coeff = 63 - idx[0][0] if len(idx[0]) else 0

    run_length = 0
    i = 1
    while i <= last_sig_coeff:
        if not levels_zz[i]:
            run_length += 1
            distortion += coefficients_zz[i]**2
            if run_length > 15:
                rate += table_ac[15 << 4, 1]
                rate_coeff[i] = table_ac[15 << 4, 1]
                run_length = 0
        else:
            ac_category = int(np.ceil(np.log2(np.abs(levels_zz[i]) + 1)))
            rlv_pair = (run_length << 4) | ac_category
            # RD cost for the current value
            distortion_non_zero = (coefficients_zz[i] - levels_zz[i] * qm_zz[i])**2
            rate_non_zero = table_ac[rlv_pair, 1] + ac_category

            # RD cost if the value is set to zero
            distortion_zero = coefficients_zz[i]**2
            run_length_current, j = run_length + 1, i + 1
            rate_zero, hit_15 = 0, 0
            # Find how much the run length can be expanded
            while j < last_sig_coeff and not levels_zz[j]:
                run_length_current += 1
                j += 1
                if run_length_current > 15:
                    rate_zero += table_ac[15 << 4, 1]
                    hit_15 += 1
                    run_length_current = 0
            next_ac_category = int(np.ceil(np.log2(np.abs(levels_zz[j]) + 1)))
            next_rlv_pair = (run_length_current << 4) | next_ac_category
            rate_zero += table_ac[next_rlv_pair, 1] + next_ac_category
            rate_zero /= (run_length_current + hit_15 * 15)

            # RD costs comparison
            rd_cost_non_zero = distortion_non_zero + _lambda * rate_non_zero
            rd_cost_zero = distortion_zero + _lambda * rate_zero
            if rd_cost_non_zero < rd_cost_zero:
                distortion += distortion_non_zero
                distortion_coeff[i] = distortion_non_zero
                rate += rate_non_zero
                rate_coeff[i] = rate_non_zero
                run_length = 0
            else:
                distortion += distortion_zero
                rate += rate_zero
                rate_coeff[i] = rate_zero
                run_length = 0 if run_length_current > 15 else run_length + 1
                levels_zz[i] = 0
        i += 1

    if last_sig_coeff != 63:
        rate += table_ac[0, 1]

    i = last_sig_coeff
    best_eob = last_sig_coeff
    distortion_now, rate_now = distortion, rate
    cost_best = distortion + _lambda * rate
    while i > 0:
        # Adjust the cost
        cost_now = (distortion_now - distortion_coeff[i] + distortion0[i]) + _lambda * (rate_now - rate_coeff[i])
        if cost_now <= cost_best:
            best_eob = i
            cost_best = cost_now
            distortion, rate = distortion_now, rate_now
        distortion_now += distortion0[i] - distortion_coeff[i]
        rate_now -= rate_coeff[i]
        i -= 1

    levels_zz[best_eob:] = 0

    # Finish off the distortion calculation for all coefficients beyond the EOB (if any)
    for i in range(last_sig_coeff + 1, 64, 1):
        distortion += coefficients_zz[i]**2

    rd_cost = distortion + _lambda * rate
    levels_rdoq = np.reshape(levels_zz[r_zz_idx], (8, 8))
    return levels_rdoq, rd_cost
