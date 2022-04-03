'''
Methods implementing image compression as specified in the
JPEG standard: https://www.itu.int/rec/T-REC-T.81-199209-I/en.

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
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
from bit_io import BitWriter
from ct import rgb_to_ycbcr_bt601
from dct import compute_dct, compute_dct_matrix
from entropy import (chroma_ac_bits, chroma_ac_values, chroma_dc_bits,
                     chroma_dc_values, encode_block, expand_huffman_table,
                     get_zigzag_scan, luma_ac_bits, luma_ac_values,
                     luma_dc_bits, luma_dc_values)
from nptyping import NDArray
from quantiser import compute_quantisation_matrices
from syntax import (write_comment, write_huffman_table, write_jfif_header,
                    write_quantisation_tables, write_segment_marker,
                    write_start_of_frame, write_start_of_scan)


def jpeg_encoding(input_image: NDArray[(Any, Any, 3), np.uint8], bitstream_name: str, quality: int) -> Tuple[int, int]:
    qy, qc = compute_quantisation_matrices(quality)
    qm = np.dstack((qy, qc, qc)).astype(np.float64)
    _, zigzag_scan = get_zigzag_scan(8)

    # Luma Huffman table generation
    luma_dc_table = expand_huffman_table(luma_dc_bits, luma_dc_values)
    luma_ac_table = expand_huffman_table(luma_ac_bits, luma_ac_values)

    # Chroma Huffman table generation
    chroma_dc_table = expand_huffman_table(chroma_dc_bits, chroma_dc_values)
    chroma_ac_table = expand_huffman_table(chroma_ac_bits, chroma_ac_values)

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
            block = input_image8[row_slice, col_slice].astype(np.float64)
            block_t[:, :, 0] = compute_dct(block[:, :, 0] - 128, T)
            block_t[:, :, 1] = compute_dct(block[:, :, 1], T)
            block_t[:, :, 2] = compute_dct(block[:, :, 2], T)
            image_dct_q[row_slice, col_slice] = np.divide(block_t + 0.5, qm).astype(np.int32)

    # Loop again over all 8x8 blocks to perform entropy coding
    total_blocks = rows8 * cols8 // 64
    dcp_y, dcp_cb, dcp_cr = 0, 0, 0
    block_idx = 0
    y_cw, cb_cw, cr_cw = [None] * total_blocks, [None] * total_blocks, [None] * total_blocks
    zigzag_idx = zigzag_scan.flatten()
    for r in range(0, rows8, 8):
        row_slice = slice(r, r + 8)
        for c in range(0, cols8, 8):
            col_slice = slice(c, c + 8)
            block_y = image_dct_q[row_slice, col_slice, 0].flatten()
            block_cb = image_dct_q[row_slice, col_slice, 1].flatten()
            block_cr = image_dct_q[row_slice, col_slice, 2].flatten()

            y_cw[block_idx] = encode_block(block_y[zigzag_idx], dcp_y, luma_dc_table, luma_ac_table)
            cb_cw[block_idx] = encode_block(block_cb[zigzag_idx], dcp_cb, chroma_dc_table, chroma_ac_table)
            cr_cw[block_idx] = encode_block(block_cr[zigzag_idx], dcp_cr, chroma_dc_table, chroma_ac_table)

            dcp_y, dcp_cb, dcp_cr = block_y[0], block_cb[0], block_cr[0]
            block_idx += 1

    # Compose all data to form the final bitstream
    # Start with the high level syntax metadata
    bw = BitWriter(bitstream_name)
    write_jfif_header(bw)
    write_comment(bw, "VCT JPEG encoder in Python")
    write_quantisation_tables(bw, qy.flatten()[zigzag_idx], qc.flatten()[zigzag_idx])
    write_start_of_frame(bw, rows, cols)
    write_huffman_table(bw, luma_dc_bits, luma_dc_values, luma_ac_bits, luma_ac_values,
                        chroma_dc_bits, chroma_dc_values, chroma_ac_bits, chroma_ac_values)
    write_start_of_scan(bw)

    # Write the VLC payload
    for payload_y, payload_cb, payload_cr in zip(y_cw, cb_cw, cr_cw):
        for code in payload_y:
            bw.submit_bits(code[0], code[1])
        for code in payload_cb:
            bw.submit_bits(code[0], code[1])
        for code in payload_cr:
            bw.submit_bits(code[0], code[1])

    bw.flush()

    # Write the end of image segment marker
    write_segment_marker(bw, 0xD9, 0)

    # Get coding stats
    bytes_total = bw.fh.tell()
    vlc_bits = bw.get_vlc_bits()
    bw.terminate()

    return bytes_total, vlc_bits


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

        input_image = np.zeros((rows, cols, 3), dtype=np.uint8)
        with open(args.input, "rb") as fh:
            input_image[:, :, 0] = np.reshape(np.frombuffer(fh.read(rows * cols), dtype=np.uint8), (rows, cols))
            input_image[:, :, 1] = np.reshape(np.frombuffer(fh.read(rows * cols), dtype=np.uint8), (rows, cols))
            input_image[:, :, 2] = np.reshape(np.frombuffer(fh.read(rows * cols), dtype=np.uint8), (rows, cols))
    else:
        input_image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        rows, cols = input_image.shape[0], input_image.shape[1]
        red, green, blue = input_image[:, :, 2], input_image[:, :, 1], input_image[:, :, 0]
        input_image = rgb_to_ycbcr_bt601(red, green, blue)

    # Print encoding parameters
    header_str = "Video coding tutorial - Python implementation of a JPEG encoder"
    print("-" * len(header_str))
    print(header_str)
    print(f"Input image: {args.input} - {cols}x{rows}")
    print(f"Bitstream: {args.output}")
    print(f"Quality factor: {args.quality}\n")

    # Call the encoder
    start = time.time()
    total_bytes, vlc_bits = jpeg_encoding(input_image, args.output, args.quality)
    stop = time.time()

    # Print final stats
    print(f"Total bytes written: {total_bytes}, corresponding to {total_bytes *8 / rows / cols:.2f} bpp")
    print(f"VLC bits: {vlc_bits}, ({vlc_bits / total_bytes / 8 * 100:.2f})% share")
    print(f"Total encoding time (s): {stop - start:.2f}")
    print("-" * len(header_str))
