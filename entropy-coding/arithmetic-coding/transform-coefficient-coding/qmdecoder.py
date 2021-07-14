'''
Class implementing decoding as per the QM coder described in Annex D of the
JPEG standard: https://www.itu.int/rec/T-REC-T.81-199209-I/en

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

from typing import List

import numpy as np

from .qmutilities import (Context, endOfArithmeticCodingMarker,
                          probability_state_machine)


class QMDecoder:
    def __init__(self):
        self.m_A: int = 0x10000
        self.m_C: int = 0
        self.m_CT: int = 0
        self.m_Qe: int = 0
        self.m_Cx: int = 0
        self.m_stop_marker_found: bool = False

    def initDec(self, buffer: List[np.uint8]) -> None:
        self.m_Qe = probability_state_machine[0][0]
        self.m_A = 0x10000
        self.m_C = 0
        self._byte_in(buffer)
        self.m_C <<= 8
        self._byte_in(buffer)
        self.m_C <<= 8
        self.m_CT = 0
        self.m_Cx = (self.m_C & 0xFFFF0000) >> 16
        self.m_stop_marker_found = False

    def decode(self, s: Context, buffer: List[np.uint8]) -> int:
        self.m_Qe = probability_state_machine[s.idx][0]
        self.m_A -= self.m_Qe

        if self.m_Cx < self.m_A:
            if self.m_A < 0x8000:
                D = self._conditional_mps_exchange(s)
                self._renormalise_d(buffer)
            else:
                D = s.mps
        else:
            D = self._conditional_lps_exchange(s)
            self._renormalise_d(buffer)
        return D

    def _byte_in(self, buffer: List[np.uint8]) -> None:
        B = 0 if self.m_stop_marker_found else buffer.pop(0)
        if B == 0xFF:
            self._unstuff_0(buffer)
        else:
            self.m_C += (B << 8)

    def _unstuff_0(self, buffer: List[np.uint8]) -> None:
        B = buffer.pop(0)
        if B == 0:
            self.m_C |= 0xFF00
        elif B == endOfArithmeticCodingMarker:
            # A termination marker is reached, from now on, virtual zero bytes
            # are shoved up in the decoder's proverbial place
            self.m_stop_marker_found = True

    def _renormalise_d(self, buffer: List[np.uint8]) -> None:
        while True:
            if self.m_CT == 0:
                self._byte_in(buffer)
                self.m_CT = 8
            self.m_A <<= 1
            self.m_C <<= 1
            self.m_CT -= 1
            if not self.m_A < 0x8000:
                break
        self.m_Cx = (self.m_C & 0xFFFF0000) >> 16

    def _conditional_lps_exchange(self, s: Context) -> int:
        if self.m_A < self.m_Qe:
            D = s.mps
            self.m_Cx -= self.m_A
            c_low = self.m_C & 0x0000FFFF
            self.m_C = (self.m_Cx << 16) + c_low
            self.m_A = self.m_Qe
            self._estimate_Qe_after_mps(s)
        else:
            D = 1 - s.mps
            self.m_Cx -= self.m_A
            c_low = self.m_C & 0x0000FFFF
            self.m_C = (self.m_Cx << 16) + c_low
            self.m_A = self.m_Qe
            self._estimate_Qe_after_lps(s)

        return D

    def _conditional_mps_exchange(self, s: Context) -> int:
        if self.m_A < self.m_Qe:
            D = 1 - s.mps
            self._estimate_Qe_after_lps(s)
        else:
            D = s.mps
            self._estimate_Qe_after_mps(s)

        return D

    def _estimate_Qe_after_lps(self, s: Context) -> None:
        if probability_state_machine[s.idx][3]:
            s.mps = 1 - s.mps

        s.idx = probability_state_machine[s.idx][1]
        self.m_Qe = probability_state_machine[s.idx][0]

    def _estimate_Qe_after_mps(self, s: Context) -> None:
        s.idx = probability_state_machine[s.idx][2]
        self.m_Qe = probability_state_machine[s.idx][0]
