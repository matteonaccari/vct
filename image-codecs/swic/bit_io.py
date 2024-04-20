'''
Helper functions to write bits to a given memory buffer.

Copyright(c) 2023 Matteo Naccari
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

import numpy as np
from nptyping import NDArray, Shape


class BitWriter:
    def __init__(self, buffer: NDArray[Shape["*"], np.uint8]) -> None:
        self.buffer: NDArray[Shape["*"], np.uint8] = buffer
        self.accumulator: int = 0
        self.bit_counter: int = 0
        self.bits_accumulated: int = 0
        self.current_ptr: int = 0
        self.capacity: int = buffer.size

    def write_bits(self, value: int, bits: int) -> None:
        if not (0 < bits and bits <= 32):
            raise Exception(f"Wrong value of bits ({bits}) provided")

        self.accumulator |= c_uint32(value << self.bits_accumulated).value

        if self.bits_accumulated + bits >= 32:
            for current_byte in int(self.accumulator).to_bytes(4, byteorder="little"):
                self.buffer[self.current_ptr] = np.uint8(current_byte)
                self.current_ptr += 1
                if self.current_ptr >= self.capacity:
                    raise Exception("Memory buffer capacity exceeded")
            self.accumulator = c_uint32(value >> (32 - self.bits_accumulated)).value
            self.bits_accumulated += bits - 32
        else:
            self.bits_accumulated += bits
        self.bit_counter += bits

    def flush(self) -> None:
        if self.current_ptr >= self.capacity:
            return
        bytes_2_write = (self.bits_accumulated + 7) >> 3
        for current_bytes in int(self.accumulator).to_bytes(bytes_2_write, byteorder="little"):
            self.buffer[self.current_ptr] = np.uint8(current_bytes)
            self.current_ptr += 1
            self.bit_counter += 8
            if self.current_ptr >= self.capacity:
                raise Exception("Memory buffer capacity exceeded")

    def bits_written(self) -> int:
        return self.bit_counter

    def bytes_written(self) -> int:
        return self.current_ptr


class BitReader:
    def __init__(self, buffer: NDArray[Shape["*"], np.uint8]) -> None:
        self.buffer: NDArray[Shape["*"], np.uint8] = buffer
        self.current_ptr: int = 0
        self.capacity: int = buffer.size
        self.accumulator: int = 0
        self.bits_held: int = 0
        self.bit_counter: int = 0

    def read(self, bits: int) -> int:
        if not (0 < bits and bits <= 32):
            raise Exception(f"Wrong number of bits ({bits}) to be read")

        if self.bit_counter + bits > self.capacity << 3:
            raise Exception("Trying to read out of the buffer memory")

        self.bit_counter += bits
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
