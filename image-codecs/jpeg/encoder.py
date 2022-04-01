'''
Methods implementing image compression as specified in the
JPEG standard: https://www.itu.int/rec/T-REC-T.81-199209-I/en

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

import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ct import rgb_to_ycbcr_bt601
from dct import compute_dct, compute_dct_matrix
from nptyping import NDArray
from quantiser import compute_quantisation_matrices


def jpeg_encoding(input_image: NDArray[(Any, Any, 3), np.uint8], bitstream_name: str, quality: int) -> None:
    qy, qc = compute_quantisation_matrices(quality)
    qm = np.dstack((qy, qc, qc)).astype(np.float64)

    # Pad the input image to make its dimensions a multiple of 8
    rows, cols = input_image.shape[0], input_image.shape[1]
    rows8, cols8 = ((rows + 7) >> 3) << 3, ((cols + 7) >> 3) << 3
    input_image8 = np.pad(input_image, ((0, rows8 - rows), (0, cols8 - cols), (0, 0)), "edge")

    # Compute the DCT matrix
    T = compute_dct_matrix(8)

    # Loop over all 8x8 blocks, compute the 2D DCT and quantise the transform coefficients
    image_dct_q = np.zeros(input_image8.shape, np.int32)
    block_t = np.zeros((8, 8, 3), np.float64)
    for r in range(0, rows8, 8):
        row_slice = slice(r, r + 8)
        for c in range(0, cols8, 8):
            col_slice = slice(c, c + 8)
            block = input_image8[row_slice, col_slice].astype(np.float64) - 128
            block_t[:, :, 0] = compute_dct(block[:, :, 0], T)
            block_t[:, :, 1] = compute_dct(block[:, :, 1], T)
            block_t[:, :, 2] = compute_dct(block[:, :, 2], T)
            image_dct_q[row_slice, col_slice] = np.divide(block_t, qm).astype(np.int32)

    # Loop again all 8x8 blocks to perform entropy coding


if __name__ == "__main__":
    cl_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    cl_parser.add_argument("-i", "--input", type=str, required=True, help="Input image to be compressed. Supported types are: png, bmp, ppm, yuv")
    cl_parser.add_argument("-o", "--output", type=str, required=True, help="Output bitstream in jpg format")
    cl_parser.add_argument("-r", "--rows", type=int, default=0, help="Image rows for yuv inputs")
    cl_parser.add_argument("-c", "--cols", type=int, default=0, help="Image columns for yuv inputs")
    cl_parser.add_argument("-q", "--quality", type=int, default=90, help="Compression quality in the range [1, 100] inclusive")

    if len(sys.argv) < 3:
        cl_parser.print_help()
        sys.exit(0)

    args = cl_parser.parse_args(sys.argv[1:])

    # Do some sanity check of the command line input
    is_yuv = Path(args.input).suffix == '.yuv'
    if not (1 <= args.quality and args.quality <= 100):
        raise Exception(f"Input quality {args.quality} out of legal range")

    # Read the input image into memory
    rows, cols = args.rows, args.cols
    if is_yuv:
        if args.rows <= 0 or args.cols <= 0:
            raise Exception(f"Frame size not correct, heigh: {args.rows}, width: {args.cols}")

        is_rgb = False
        input_image = np.zeros((rows, cols, 3), dtype=np.uint8)
        with open(args.input, "rb") as fh:
            input_image[:, :, 0] = np.reshape(np.frombuffer(fh.read(rows * cols), dtype=np.uint8), (rows, cols))
            input_image[:, :, 1] = np.reshape(np.frombuffer(fh.read(rows * cols), dtype=np.uint8), (rows, cols))
            input_image[:, :, 2] = np.reshape(np.frombuffer(fh.read(rows * cols), dtype=np.uint8), (rows, cols))
    else:
        is_rgb = True
        input_image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        red, green, blue = input_image[:, :, 2], input_image[:, :, 1], input_image[:, :, 0]
        input_image = rgb_to_ycbcr_bt601(red, green, blue)

    # Call the encoder
    jpeg_encoding(input_image, args.output, args.quality)
