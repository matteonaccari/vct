'''
Helper functions to write bits and bytes out to a file.

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

from typing import BinaryIO, List

import numpy as np


class BitWriter:
    def __init__(self, file_name: str) -> None:
        self.accumulator: np.int32 = 0
        self.bit_counter: np.int32 = 0
        self.vlc_bits: np.int32 = 0
        self.fh: BinaryIO = open(file_name, "wb")

        if not self.fh:
            raise Exception(f"Failed to open file: {file_name}")

    def write_bytes(self, values: List[np.uint8]) -> None:
        self.fh.write(bytearray(values))

    def submit_bits(self, value: int, length: int) -> None:
        if length < 0:
            raise Exception(f"Negative length value ({length}) detected")
        self.vlc_bits += length
        self.bit_counter += length
        self.accumulator <<= length
        self.accumulator |= value

        while self.bit_counter >= 8:
            self.bit_counter -= 8
            current_byte = np.uint8(self.accumulator >> self.bit_counter)
            self.fh.write(current_byte)
            self.accumulator &= (1 << self.bit_counter) - 1
            if current_byte == 255:
                # Marker emulation prevention
                self.fh.write(np.uint8(0))

    def flush(self) -> None:
        self.submit_bits(127, 7)

    def get_vlc_bits(self) -> int:
        return self.vlc_bits

    def terminate(self) -> None:
        self.fh.close()
