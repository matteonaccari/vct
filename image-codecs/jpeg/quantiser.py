'''
Helper functions to handle uniform quantisation of the DCT coefficients with the default tables
reported in the JPEG standard specification: https://www.itu.int/rec/T-REC-T.81-199209-I/en.

Copyright(c) 2022 - 2023 Matteo Naccari
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
from nptyping import NDArray, Shape

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


def compute_quantisation_matrices(quality: int) -> Tuple[NDArray[Shape["8, 8"], np.int32], NDArray[Shape["8, 8"], np.int32]]:
    # Adjust the quality value according to the relationship worked out by the Indipendent JPEG Group (IJG)
    quality = 5000 // quality if quality < 50 else 200 - quality * 2

    q_luma = np.clip((luma_quantisation_matrix * quality + 50) // 100, 1, 255)
    q_chroma = np.clip((chroma_quantisation_matrix * quality + 50) // 100, 1, 255)

    return q_luma, q_chroma


def rd_rl_pair(start: int, coefficients_zz: NDArray[Shape["64"], np.float64], levels_zz: NDArray[Shape["64"], np.int32],
               qm_zz: NDArray[Shape["8, 8"], np.float64], ht: NDArray[Shape["256, 2"], np.int32]) -> Tuple[int, float, float, int]:
    i, run_length = start, 0
    distortion = 0
    while levels_zz[i] == 0:
        run_length += 1
        distortion += coefficients_zz[i]**2
        i += 1

    distortion_value = (coefficients_zz[i] - levels_zz[i] * qm_zz[i])**2
    distortion += distortion_value
    category = int(np.ceil(np.log2(np.abs(levels_zz[i]) + 1)))
    rlv_pair = ((run_length & 15) << 4) | category
    rate = ht[rlv_pair, 1] + category + (run_length >> 4) * ht[15 << 4, 1]

    return rate, distortion, distortion_value, run_length


def rdoq_8x8_plane(coefficients: NDArray[Shape["8, 8"], np.float64], qm: NDArray[Shape["8, 8"], np.float64],
                   table_dc: NDArray[Shape["256, 2"], np.int32], table_ac: NDArray[Shape["256, 2"], np.int32],
                   _lambda: float, zz_idx: NDArray[Shape["8, 8"], np.uint8], r_zz_idx: NDArray[Shape["8, 8"], np.uint8],
                   pred_dc: int) -> Tuple[NDArray[Shape["8, 8"], np.int32], float, int]:
    # Regular quantisation
    s = np.sign(coefficients).astype(np.int32)
    levels = np.divide(coefficients + 0.5, qm).astype(np.int32)

    # Coefficient-based rate and distortion storage
    distortion0 = np.square(coefficients).flatten()[zz_idx]
    distortion_coeff = np.square(coefficients).flatten()[zz_idx]
    rate_coeff = np.zeros((64), np.int32)

    # Apply DPCM for the DC coefficient
    res_dc = levels[0, 0] - pred_dc

    # Phase 1: Optimise the RD cost for the DC coefficient
    if not res_dc:
        # Zero value residual, no quantisation distortion, just add the rate
        rate = table_dc[0, 1]
        rate_coeff[0] = table_dc[0, 1]
        distortion_coeff[0] = (coefficients[0, 0] - levels[0, 0] * qm[0, 0])**2
        distortion = distortion_coeff[0]
    else:
        # Perform RDO: test the current DC value, zero and its absolute value minus one
        best_rd_cost = np.finfo(np.float64).max
        sign_dc = s[0, 0]
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
                distortion_coeff[0] = current_dist

    # Find the eob
    levels_zz = levels.flatten()[zz_idx]
    idx = np.where(levels_zz[-1:0:-1] != 0)
    last_sig_coeff = 63 - idx[0][0] if len(idx[0]) else 0

    # RDO for the AC coefficients
    coefficients_zz = coefficients.flatten()[zz_idx]
    qm_zz = qm.flatten()[zz_idx]
    s_zz = s.flatten()[zz_idx]

    # Phase 2: Try different level values for each AC coefficient and pick the one which
    # minimimise its RD cost
    rate_eob = table_ac[0, 1]
    i = 1
    while i <= last_sig_coeff:
        # RD cost for the current quantised level
        rq, dq, dsq, rlq = rd_rl_pair(i, coefficients_zz, levels_zz, qm_zz, table_ac)
        rd_costq = dq + _lambda * rq
        distortion_coeff[i + rlq] = dsq
        rate_coeff[i + rlq] = rq
        rd_cost = rd_costq

        # RD cost for |level| - 1
        lq = levels_zz[i + rlq]
        if np.abs(lq) > 1:
            lq_m1 = s_zz[i + rlq] * (np.abs(lq) - 1)
            levels_zz[i + rlq] = lq_m1
            rq_m1, dq_m1, dsq_m1, _ = rd_rl_pair(i, coefficients_zz, levels_zz, qm_zz, table_ac)
            rd_costq_m1 = dq_m1 + _lambda * rq_m1
            if rd_costq_m1 < rd_costq:
                distortion_coeff[i + rlq] = dsq_m1
                rate_coeff[i + rlq] = rq_m1
                rd_cost = rd_costq_m1
            else:
                levels_zz[i + rlq] = lq

        # RD cost for level zero
        if i + rlq != last_sig_coeff:
            rqn, dqn, _, _ = rd_rl_pair(i + rlq + 1, coefficients_zz, levels_zz, qm_zz, table_ac)
            rd_costn = dqn + _lambda * rqn
            lq = levels_zz[i + rlq]
            levels_zz[i + rlq] = 0
            rq0, dq0, dsq0, rlq0 = rd_rl_pair(i, coefficients_zz, levels_zz, qm_zz, table_ac)
            rd_cost0 = dq0 + _lambda * rq0

            if rd_cost0 < rd_cost + rd_costn:
                rate_coeff[i + rlq0] = rq0
                distortion_coeff[i + rlq0] = dsq0
                distortion_coeff[i + rlq] = distortion0[i + rlq]
                rate_coeff[i + rlq] = 0
                if i + rlq0 == last_sig_coeff:
                    break
            else:
                levels_zz[i + rlq] = lq
                i += rlq + 1
        else:
            break

    distortion = np.sum(distortion_coeff)
    rate = np.sum(rate_coeff)

    # Phase 3: Move the EOB towards the block's top left corner
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
                distortion, rate = distortion_now, rate_now
        i -= 1

    levels_zz[best_eob:] = 0

    levels_rdoq = np.reshape(levels_zz[r_zz_idx], (8, 8))

    return levels_rdoq, distortion, rate
