'''
Routines implementing the reading and writing of bits to a file.

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

from ctypes import c_uint32
from typing import Any, List

import numpy as np
from nptyping import NDArray


class BitReader:
    def __init__(self, buffer: NDArray[(Any), np.uint8]) -> None:
        self.buffer: NDArray[(Any), np.uint8] = buffer
        self.current_ptr: int = 0
        self.capacity: int = buffer.size
        self.accumulator: np.uint32 = 0
        self.bits_held: int = 0
        self.total_bits_read: int = 0

    def eob(self) -> bool:
        return self.current_ptr >= self.capacity

    def read(self, bits: int) -> int:
        if not (0 < bits and bits <= 32):
            raise Exception("Bits to read must be in the range [1, 32] inclusive")
        self.total_bits_read += bits
        value, bits_remaining = 0, bits
        while True:
            if self.bits_held >= bits_remaining:
                value |= (self.accumulator & ((1 << bits_remaining) - 1)) << (bits - bits_remaining)
                self.accumulator >>= bits_remaining
                self.bits_held -= bits_remaining
                break
            value |= self.accumulator << (bits - bits_remaining)
            bits_remaining -= self.bits_held
            self.accumulator = self.buffer[self.current_ptr]
            self.current_ptr += 1
            self.bits_held = 8

        return value


class BitWriter:
    def __init__(self) -> None:
        self.buffer: List[np.uint8] = []
        self.accumulator: np.int32 = 0
        self.bits_accumulated: int = 0
        self.total_bits_written: int = 0

    def write(self, value: int, bits: int) -> None:
        if not 0 < bits and bits <= 32:
            raise Exception("Bits to write must be in the range [1, 32] inclusive")

        self.accumulator |= c_uint32(value << self.bits_accumulated).value

        if self.bits_accumulated + bits >= 32:
            for b in int(self.accumulator).to_bytes(4, byteorder='little'):
                self.buffer.append(np.uint8(b))
            self.accumulator = value >> (32 - self.bits_accumulated)
            self.bits_accumulated += bits - 32
        else:
            self.bits_accumulated += bits

        self.total_bits_written += bits

    def flush(self) -> None:
        bytes_2_write = (self.bits_accumulated + 7) >> 3
        for b in int(self.accumulator).to_bytes(bytes_2_write, byteorder='little'):
            self.buffer.append(np.uint8(b))
        self.accumulator = self.bits_accumulated = 0

    def get_bits_written(self) -> int:
        return self.total_bits_written

    def get_bytes_written(self) -> int:
        return (self.total_bits_written + 7) >> 3

    def get_data_buffer(self) -> NDArray[(Any), np.uint8]:
        return np.array(self.buffer, np.uint8)
