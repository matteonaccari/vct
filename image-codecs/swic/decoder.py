'''
Image decoding using different discrete wavelet transforms as frequency decomposition:
 * Haar Wavelet (with and w/o dynamic range expansion)
 * LeGall 5/3 as used in the JPEG 2000 (ITU-T T.800) standard for lossless encoding
 * Cohen Dabeuchies Feauveau (CDF) 9/7 as used in the JPEG 2000 standard for lossy
   encoding

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

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import numpy as np
from typing import Any
from nptyping import NDArray
import time
from dwt import inverse_haar_dwt
from quantiser import reconstruct_plane
from entropy import code_block_size, decode_subband
from ct import ycbcr_to_rgb_bt709
import cv2

symbolic_header = "SWIC-V1"


def swic_decoder(bitstream_file: str, levels: int) -> NDArray[(Any, Any, 3), np.int32]:
    # High level syntax parsing
    with open(bitstream_file, "rb") as fh:
        # Symbolic header
        header = fh.read(len(symbolic_header)).decode("ascii")
        if header != symbolic_header:
            raise Exception(f"Symbolic header '{header}' difference from: {symbolic_header}")
        # Image width and height
        cols = int().from_bytes(fh.read(2), byteorder="little")
        rows = int().from_bytes(fh.read(2), byteorder="little")
        # Pixel bit depth and number of components
        depth_components_bytes = int().from_bytes(fh.read(1), byteorder="little")
        bitdepth = (depth_components_bytes & 0x0F) + 8
        components = (depth_components_bytes >> 4) & 0x03
        # Decomposition levels and transform type
        levels_transform_type = int().from_bytes(fh.read(1), byteorder="little")
        levels_encoded = (levels_transform_type >> 4) & 0x0F
        transform = levels_transform_type & 0x03
        if levels_encoded < levels:
            print(f"Warning: levels required to be decoded ({levels}) are more than the ones actually encoded ({levels_encoded})")
            levels = levels_encoded
        elif not levels:
            levels = levels_encoded
        if transform:
            raise Exception("Only Haar supported so far")
        # Quantisation parameter
        qp = int().from_bytes(fh.read(1), byteorder="little")

        # Load into memory all code block payloads
        payload_levels = [None] * levels
        for level in range(levels):
            total_subbands = 4 if not level else 3
            rows_sb, cols_sb = rows >> (levels_encoded - level), cols >> (levels_encoded - level)
            rows_cb, cols_cb = (rows_sb + code_block_size - 1) // code_block_size, (cols_sb + code_block_size - 1) // code_block_size
            total_cbs = rows_cb * cols_cb

            payload_sbs = [None] * total_subbands
            for sb_idx in range(total_subbands):
                payload_sb = [None] * total_cbs
                for cb_idx in range(total_cbs):
                    cb_payload_size = int().from_bytes(fh.read(2), byteorder="little")
                    payload_sb[cb_idx] = np.frombuffer(fh.read(cb_payload_size), dtype=np.uint8)
                payload_sbs[sb_idx] = payload_sb
            payload_levels[level] = payload_sbs

    # Print out high level syntax information
    print(f"Encoded image size: {cols}x{rows}")
    print(f"Encoded image bit depth: {bitdepth} [bpp]")
    print(f"Encoded image number of components: {components}")
    print(f"Output image size: {cols >> (levels_encoded - levels)}x{rows >> (levels_encoded - levels)}")
    print(f"Quantisation parameter: {qp}")

    # Entropy decoding
    coefficients = []
    for level in range(levels):
        current_level = payload_levels[level]
        rows_sb, cols_sb = rows >> (levels_encoded - level), cols >> (levels_encoded - level)
        coefficients_sbs = []
        for sb_idx, sb in enumerate(current_level):
            is_ll = not level and not sb_idx
            coefficients_sb = decode_subband(sb, rows_sb, cols_sb, components, is_ll)
            coefficients_sbs.append(coefficients_sb)
        coefficients.append(coefficients_sbs)

    # Inverse quantisation
    coefficients_r = []
    for level in range(levels):
        shift = levels_encoded - level - 1
        offset = 1 << (shift - 1) if shift else 0
        current_level = coefficients[level]
        coefficients_sbs_r = []
        for sb_idx, sb in enumerate(current_level):
            coefficients_sb_r = np.zeros(sb.shape, np.int32)
            if components == 1:
                coefficients_sb_r = (reconstruct_plane(sb, qp) + offset) >> shift
            else:
                for comp in range(components):
                    coefficients_sb_r[:, :, comp] = (reconstruct_plane(sb[:, :, comp], qp) + offset) >> shift
            coefficients_sbs_r.append(coefficients_sb_r)
        coefficients_r.append(coefficients_sbs_r)

    max_value = (1 << bitdepth) - 1
    midrange_value = 1 << (bitdepth - 1)

    # Inverse transform
    for level in range(levels):
        subbands = coefficients_r[level]
        if not level:
            current_ll = inverse_haar_dwt(subbands[0], subbands[1], subbands[2], subbands[3])
        else:
            current_ll = inverse_haar_dwt(current_ll, subbands[0], subbands[1], subbands[2])

    decoded_image = current_ll + midrange_value
    decoded_image = np.clip(decoded_image, 0, max_value)

    return decoded_image


if __name__ == "__main__":
    cl_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    cl_parser.add_argument("-i", "--input", type=str, required=True, help="Input bitstream compliant with SWIC compression format")
    cl_parser.add_argument("-o", "--output", type=str, required=True, help="Decoded file")
    cl_parser.add_argument("-l", "--levels", type=int, default=0, help="Number of decomposition levels to be decoded, default all which were compressed")
    supported_output = (".yuv", ".png", ".bmp")

    if len(sys.argv) < 3:
        cl_parser.print_help()
        sys.exit(0)

    args = cl_parser.parse_args(sys.argv[1:])

    # Do some sanity check on the input parameters
    if args.levels < 0:
        raise Exception(f"Decomposition levels {args.levels} cannot be negative")

    output_ext = Path(args.output).suffix
    if output_ext not in supported_output:
        raise Exception(f"Decoded file extension {output_ext} not supported. Image formats allowed are: {supported_output}")

    # Print encoding parameters
    header_str = "Video coding tutorial - Simple Wavelet-based Image Coding (SWIC) [decoder]"
    print("-" * len(header_str))
    print(header_str)
    print(f"Input bitstream: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Decomposition levels: {'all encoded' if not args.levels else args.levels}")

    # Call the encoder
    start = time.time()
    decoded_image = swic_decoder(args.input, args.levels)
    stop = time.time()

    # Write out the decoded image
    components = 3 if len(decoded_image.shape) == 3 else 2
    if output_ext == ".yuv":
        with open(args.output, "wb") as fh:
            for comp in range(components):
                plane = decoded_image[:, :, comp].astype(np.uint8)
                fh.write(plane.tobytes())
    else:
        if components == 3:
            rgb_image = ycbcr_to_rgb_bt709(decoded_image[:, :, 0], decoded_image[:, :, 1], decoded_image[:, :, 2])
            bgr_image = np.zeros_like(rgb_image)
            bgr_image[:, :, 0] = rgb_image[:, :, 2]
            bgr_image[:, :, 1] = rgb_image[:, :, 1]
            bgr_image[:, :, 2] = rgb_image[:, :, 0]
            cv2.imwrite(args.output, bgr_image.astype(np.uint8))
        else:
            cv2.imwrite(args.output, decoded_image.astype(np.uint8))

    # Print final stats
    print(f"Total decoding time (s): {stop - start:.2f}")
    print("-" * len(header_str))
