'''
Routines to write and parse high level syntax metadata as specified
in the Simple Wavelet-based Image Coding (SWIC)

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

from dataclasses import dataclass
from typing import BinaryIO

symbolic_header = "SWIC-V1"


@dataclass
class ImageParameterSet:
    rows: int = 0
    cols: int = 0
    components: int = 0
    bitdepth: int = 0
    levels: int = 0
    transform: int = 0
    qp: int = 0


def write_ips(fh: BinaryIO, ips: ImageParameterSet) -> None:
    # Symbolic header
    fh.write(symbolic_header.encode("ascii"))
    # Image width and height
    fh.write(int(ips.cols).to_bytes(2, byteorder="little"))
    fh.write(int(ips.rows).to_bytes(2, byteorder="little"))
    # Pixel bit depth and number of components
    components_depth_byte = ((ips.components & 0x03) << 4) | ((ips.bitdepth - 8) & 0x0F)
    fh.write(int(components_depth_byte).to_bytes(1, byteorder="little"))
    # Decomposition levels and transform type
    level_transform_byte = ((ips.levels & 0x0F) << 4) | (ips.transform & 0x03)
    fh.write(int(level_transform_byte).to_bytes(1, byteorder="little"))
    # QP
    fh.write(int(ips.qp).to_bytes(1, byteorder="little"))


def read_ips(fh: BinaryIO) -> ImageParameterSet:
    ips = ImageParameterSet()
    # Symbolic header
    header = fh.read(len(symbolic_header)).decode("ascii")
    if header != symbolic_header:
        raise Exception(f"Symbolic header '{header}' difference from: {symbolic_header}")
    # Image width and height
    ips.cols = int().from_bytes(fh.read(2), byteorder="little")
    ips.rows = int().from_bytes(fh.read(2), byteorder="little")
    # Pixel bit depth and number of components
    depth_components_bytes = int().from_bytes(fh.read(1), byteorder="little")
    ips.bitdepth = (depth_components_bytes & 0x0F) + 8
    ips.components = (depth_components_bytes >> 4) & 0x03
    # Decomposition levels and transform type
    levels_transform_type = int().from_bytes(fh.read(1), byteorder="little")
    ips.levels = (levels_transform_type >> 4) & 0x0F
    ips.transform = levels_transform_type & 0x03

    # Quantisation parameter
    ips.qp = int().from_bytes(fh.read(1), byteorder="little")
    return ips
