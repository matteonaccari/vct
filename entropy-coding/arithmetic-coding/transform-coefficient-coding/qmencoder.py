'''
Class implementing encoding as per the QM coder described in Annex D of the
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

from .qmutilities import Context, probability_state_machine


class QMEncoder:
    def __init__(self):
        self.m_A: int = 0x10000
        self.m_C: int = 0
        self.m_CT: int = 11
        self.m_ST: int = 0
        self.m_Qe: int = probability_state_machine[0][0]
        self.m_buffer_size: int = 0
        self.m_byte_buffer: List[np.uint8] = []

    def flush(self) -> None:
        self._clear_final_bits()
        self.m_C <<= self.m_CT
        self._byte_out()
        self.m_C <<= 8
        self._byte_out()
        self._discard_final_zeros()

    def init_enc(self) -> None:
        self.m_A = 0x10000
        self.m_C = 0
        self.m_CT = 11
        self.m_ST = 0
        self.m_byte_buffer = []
        self.m_buffer_size = 0
        self.m_Qe = probability_state_machine[0][0]

    def code0(self, s: Context) -> None:
        self.m_Qe = probability_state_machine[s.idx][0]
        if (s.mps == 0):
            self._code_MPS(s)
        else:
            self._code_LPS(s)

    def code1(self, s: Context) -> None:
        self.m_Qe = probability_state_machine[s.idx][0]
        if (s.mps):
            self._code_MPS(s)
        else:
            self._code_LPS(s)

    def insert_marker(self, marker: np.uint8) -> None:
        self.m_byte_buffer.append(0xFF)
        self.m_byte_buffer.append(marker)
        self.m_buffer_size += 2

    def get_byte_buffer(self) -> List[np.uint8]:
        return self.m_byte_buffer

    def _renormalise_e(self) -> None:
        while True:
            self.m_A <<= 1
            self.m_C <<= 1
            self.m_CT -= 1
            if self.m_CT == 0:
                self._byte_out()
                self.m_CT = 8
            if not self.m_A < 32768:
                break

    def _code_MPS(self, s: Context) -> None:
        self.m_A -= self.m_Qe
        if self.m_A < 32768:
            if self.m_A < self.m_Qe:
                self.m_C += self.m_A
                self.m_A = self.m_Qe
            self._estimate_Qe_after_MPS(s)
            self._renormalise_e()

    def _code_LPS(self, s: Context) -> None:
        self.m_A -= self.m_Qe
        if self.m_A >= self.m_Qe:
            self.m_C += self.m_A
            self.m_A = self.m_Qe
        self._estimate_Qe_after_LPS(s)
        self._renormalise_e()

    def _byte_out(self) -> None:
        t = self.m_C >> 19

        if t > 0xFF:
            self.m_byte_buffer[self.m_buffer_size - 1] += 1
            self._stuff0()
            self._output_stacked_zeros()
            self.m_byte_buffer.append(np.uint8(t))
            self.m_buffer_size += 1
        else:
            if t == 0xFF:
                self.m_ST += 1
            else:
                self._output_stacked_FFs()
                self.m_byte_buffer.append(t)
                self.m_buffer_size += 1
        self.m_C &= 0x7FFFF

    def _stuff0(self) -> None:
        if self.m_byte_buffer[self.m_buffer_size - 1] == 0xFF:
            self.m_byte_buffer.append(0)
            self.m_buffer_size += 1

    def _output_stacked_zeros(self) -> None:
        while self.m_ST > 0:
            self.m_byte_buffer.append(0)
            self.m_buffer_size += 1
            self.m_ST -= 1

    def _output_stacked_FFs(self) -> None:
        while self.m_ST > 0:
            self.m_byte_buffer.append(255)
            self.m_byte_buffer.append(0)
            self.m_buffer_size += 2
            self.m_ST -= 1

    def _clear_final_bits(self) -> None:
        t = self.m_C + self.m_A - 1
        t &= 0xffff0000
        if t < self.m_C:
            t += 0x8000
        self.m_C = t

    def _discard_final_zeros(self) -> None:
        if self.m_buffer_size:
            # Scan the byte buffer backwards and discard the last run of zeros
            start_of_final_run = self.m_buffer_size
            bytes_removed = 0
            while True:
                if self.m_byte_buffer[start_of_final_run - 1] == 0:
                    start_of_final_run -= 1
                    bytes_removed += 1
                else:
                    break
        if start_of_final_run != self.m_buffer_size:
            del self.m_byte_buffer[start_of_final_run:]
            self.m_buffer_size -= bytes_removed

    def _estimate_Qe_after_LPS(self, s: Context) -> None:
        if probability_state_machine[s.idx][3]:
            s.mps = 1 - s.mps
        s.idx = probability_state_machine[s.idx][1]
        self.m_Qe = probability_state_machine[s.idx][0]

    def _estimate_Qe_after_MPS(self, s: Context) -> None:
        s.idx = probability_state_machine[s.idx][2]
        self.m_Qe = probability_state_machine[s.idx][0]
