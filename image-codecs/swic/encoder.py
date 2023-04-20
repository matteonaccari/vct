'''
Image encoding using different discrete wavelet transforms as frequency decomposition:
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
from ct import rgb_to_ycbcr_bt709, ycbcr_to_rgb_bt709
from pathlib import Path
import numpy as np
import cv2
from typing import Any, Tuple
from nptyping import NDArray
import time
from dwt import forward_haar_dwt, inverse_haar_dwt, DwtType, forward_legall_5_3_dwt, inverse_legall_5_3_dwt
from quantiser import quantise_plane, reconstruct_plane
from entropy import encode_subband
from hls import ImageParameterSet, write_ips


def swic_encoder(input_image: NDArray[(Any, Any, 3), np.int32], bitstream_name: str, qp: int, bitdepth: int, levels: int, transform_type: DwtType, reconstruction_needed: bool) -> Tuple[int, NDArray[(Any, Any, Any), np.int32]]:
    # Remove mid range value from input data
    midrange_value = 1 << (bitdepth - 1)
    max_value = (1 << bitdepth) - 1
    input_image -= midrange_value
    components = 3 if len(input_image.shape) == 3 else 1
    transform = transform_type.value
    forward_dwt = {DwtType.Haar: forward_haar_dwt, DwtType.LeGall5_3: forward_legall_5_3_dwt}
    inverse_dwt = {DwtType.Haar: inverse_haar_dwt, DwtType.LeGall5_3: inverse_legall_5_3_dwt}

    # Perform forward DWT over the number of given levels
    subbands = []
    current_ll = input_image
    for level in range(levels):
        ll, hl, lh, hh = forward_dwt[transform_type](current_ll)
        if level == levels - 1:
            current_sbs = [ll, hl, lh, hh]
        else:
            current_sbs = [hl, lh, hh]

        subbands.append(current_sbs)
        current_ll = ll.copy()

    # Perform uniform quantisation
    subbands_q = []
    for level in range(levels):
        current_levq = []
        shift = (level)
        for sb in subbands[level]:
            sbq = np.zeros(sb.shape, np.int32)
            if components > 1:
                for comp in range(components):
                    sbq[:, :, comp] = quantise_plane(sb[:, :, comp] << shift, qp)
            else:
                sbq = quantise_plane(sb << shift, qp)
            current_levq.append(sbq)
        subbands_q.append(current_levq)

    # Entropy coding
    payload_levels = []
    for level in range(levels - 1, -1, -1):
        current_lev_sbs = subbands_q[level]
        payload_level = []
        for idx, sb in enumerate(current_lev_sbs):
            is_ll = level == levels - 1 and not idx
            payload_cbs = encode_subband(sb, is_ll)
            payload_level.append(payload_cbs)
        payload_levels.append(payload_level)

    # Write out the bitstream
    with open(bitstream_name, "wb") as fh:
        ips = ImageParameterSet(rows=rows, cols=cols, components=components, bitdepth=bitdepth, levels=levels, transform=transform, qp=qp)
        write_ips(fh, ips)

        # Decomposition levels, subbands and levels
        for current_level in payload_levels:
            for sb in current_level:
                for cb_payload in sb:
                    fh.write(cb_payload.tobytes())

        total_bytes = fh.tell()

    # Reconstruction path: inverse quantisation (aka reconstruction), inverse DWT and compute PSNR
    reconstructed_image = None

    if reconstruction_needed:
        subbands_r = []
        for level in range(levels):
            current_levr = []
            shift = (level)
            offset = 1 << (shift - 1) if shift else 0
            for sb in subbands_q[level]:
                sbr = np.zeros(sb.shape, np.int32)
                if components > 1:
                    for comp in range(components):
                        sbr[:, :, comp] = reconstruct_plane(sb[:, :, comp], qp)
                else:
                    sbr = reconstruct_plane(sb, qp)
                current_levr.append((sbr + offset) >> shift)
            subbands_r.append(current_levr)

        # Inverse DWT
        for level in range(levels - 1, -1, -1):
            current_sbs = subbands_r[level]
            if level == levels - 1:
                current_ll = inverse_dwt[transform_type](current_sbs[0], current_sbs[1], current_sbs[2], current_sbs[3])
            else:
                current_ll = inverse_dwt[transform_type](current_ll, current_sbs[0], current_sbs[1], current_sbs[2])

        reconstructed_image = current_ll + midrange_value
        reconstructed_image = np.clip(reconstructed_image, 0, max_value)

        psnr = np.zeros((components), np.int32)

        if components > 1:
            for comp in range(components):
                plane_o = input_image[:, :, comp] + midrange_value
                plane_r = reconstructed_image[:, :, comp]
                mse = np.mean(np.square(plane_o - plane_r))
                psnr[comp] = 10 * np.log10(max_value ** 2 / mse) if mse else np.inf
            print(f"PSNR-Y: {psnr[0]:.2f}, PSNR-Cb: {psnr[1]:.2f}, PSNR-Cr: {psnr[2]:.2f} [dB]")
        else:
            mse = np.mean(np.square(input_image + midrange_value - reconstructed_image))
            psnr[0] = 10 * np.log10(max_value ** 2 / mse) if mse else np.inf
            print(f"PSNR-Y: {psnr[0]:.2f}")

    return total_bytes, reconstructed_image


if __name__ == "__main__":
    cl_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    cl_parser.add_argument("-i", "--input", type=str, required=True, help="Input image to be compressed. Supported types are: png, bmp, ppm, yuv")
    cl_parser.add_argument("-o", "--output", type=str, required=True, help="Output bitstream in the SWIC format")
    cl_parser.add_argument("-r", "--rows", type=int, default=0, help="Image rows for yuv inputs")
    cl_parser.add_argument("-c", "--cols", type=int, default=0, help="Image columns for yuv inputs")
    cl_parser.add_argument("-q", "--quantisation", type=int, default=22, help="Quantisation parameter in the range [0, 51] inclusive")
    cl_parser.add_argument("-b", "--bitdepth", type=int, default=8, help="Input bitdepth in the range [8, 12] inclusive")
    cl_parser.add_argument("-l", "--levels", type=int, default=5, help="Number of decomposition levels in the range [1, 5]. Mallat (aka packed) decomposition is assumed")
    cl_parser.add_argument("-t", "--transform", type=DwtType.from_string, choices=list(DwtType), default=DwtType.Haar, help="Transform type")
    cl_parser.add_argument("--reconstructed", type=str, default="", help="Dump the reconstructed file in yuv, png or bmp format for comparison purposes")
    supported_recon = (".yuv", ".png", ".bmp")

    if len(sys.argv) < 3:
        cl_parser.print_help()
        sys.exit(0)

    args = cl_parser.parse_args(sys.argv[1:])

    # Do some sanity check on the input parameters
    is_yuv = Path(args.input).suffix == ".yuv"
    recon_extension = Path(args.reconstructed).suffix

    if not (0 <= args.quantisation and args.quantisation <= 51):
        raise Exception(f"Quantisation parameter {args.quantisation} out of its legal range")
    if not (8 <= args.bitdepth and args.bitdepth <= 12):
        raise Exception(f"Input bitdepth {args.bitdepth} out of its legal range")
    if not (1 <= args.levels and args.levels <= 5):
        raise Exception(f"Decomposition levels {args.levels} out of its legal range")
    if args.reconstructed and (recon_extension not in supported_recon):
        raise Exception(f"Reconstructed file format {recon_extension} not supported, only {supported_recon} are")

    rows, cols = args.rows, args.cols

    if is_yuv:
        if args.rows <= 0 or args.cols <= 0:
            raise Exception(f"Frame size not correct, heigh: {args.rows}, width: {args.cols}")

        curr_dtype = np.uint8 if args.bitdepth == 8 else np.int32
        bytes_per_pixels = 1 if args.bitdepth == 8 else 2
        comp_size = rows * cols * bytes_per_pixels
        input_image = np.zeros((rows, cols, 3), dtype=np.int32)
        with open(args.input, "rb") as fh:
            input_image[:, :, 0] = np.reshape(np.frombuffer(fh.read(comp_size), dtype=curr_dtype), (rows, cols)).astype(np.int32)
            input_image[:, :, 1] = np.reshape(np.frombuffer(fh.read(comp_size), dtype=curr_dtype), (rows, cols)).astype(np.int32)
            input_image[:, :, 2] = np.reshape(np.frombuffer(fh.read(comp_size), dtype=curr_dtype), (rows, cols)).astype(np.int32)
    else:
        input_image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED).astype(np.int32)
        rows, cols = input_image.shape[0], input_image.shape[1]
        components = 3 if len(input_image.shape) == 3 else 1
        if components == 3:
            red, green, blue = input_image[:, :, 2], input_image[:, :, 1], input_image[:, :, 0]
            input_image = rgb_to_ycbcr_bt709(red, green, blue, args.bitdepth)

    # Print encoding parameters
    header_str = "Video coding tutorial - Simple Wavelet-based Image Coding (SWIC) [encoder]"
    print("-" * len(header_str))
    print(header_str)
    print(f"Input image: {args.input} - {cols}x{rows}")
    print(f"Input bit depth: {args.bitdepth} [bpp]")
    print(f"Input components: {3 if len(input_image.shape) == 3 else 1}")
    print(f"Bitstream: {args.output}")
    print(f"Quantisation parameter: {args.quantisation}")
    print(f"Decomposition levels: {args.levels}")
    print(f"DWT: {args.transform}")
    if args.reconstructed:
        print(f"Reconstructed file: {args.reconstructed}")

    # Call the encoder
    start = time.time()
    total_bytes, reconstructed_image = swic_encoder(input_image, args.output, args.quantisation, args.bitdepth, args.levels, args.transform, args.reconstructed != "")
    stop = time.time()

    # Dump reconstructed file if needed
    if args.reconstructed:
        components = 3 if len(reconstructed_image.shape) == 3 else 1
        if recon_extension == ".yuv":
            with open(args.reconstructed, "wb") as fh:
                for comp in range(components):
                    plane = reconstructed_image[:, :, comp].astype(np.uint8)
                    fh.write(plane.tobytes())
        else:
            if components == 3:
                rgb_image = ycbcr_to_rgb_bt709(reconstructed_image[:, :, 0], reconstructed_image[:, :, 1], reconstructed_image[:, :, 2])
                bgr_image = np.zeros_like(rgb_image)
                bgr_image[:, :, 0] = rgb_image[:, :, 2]
                bgr_image[:, :, 1] = rgb_image[:, :, 1]
                bgr_image[:, :, 2] = rgb_image[:, :, 0]
                cv2.imwrite(args.reconstructed, bgr_image.astype(np.uint8))
            else:
                cv2.imwrite(args.reconstructed, reconstructed_image.astype(np.uint8))

    # Print final stats
    print(f"Total bytes written: {total_bytes}, corresponding to {total_bytes * 8 / rows / cols:.2f} [bpp]")
    print(f"Total encoding time (s): {stop - start:.2f}")
    print("-" * len(header_str))
