import os
from hashlib import md5

import cv2
import numpy as np
from ct import rgb_to_ycbcr_bt709
from encoder_rdoq import jpeg_encoding_rdoq
from entropy import (expand_huffman_table, get_zigzag_scan, luma_ac_bits,
                     luma_ac_values, luma_dc_bits, luma_dc_values)
from quantiser import rdoq_8x8_plane

table_dc = expand_huffman_table(luma_dc_bits, luma_dc_values)
table_ac = expand_huffman_table(luma_ac_bits, luma_ac_values)
r_zz_idx, zz_idx = get_zigzag_scan(8)
zz_idx = zz_idx.flatten()


def compute_md5(file_name: str) -> str:
    with open(file_name, "rb") as fh:
        data = fh.read()

    return md5(data).hexdigest()


def test_jpeg_rdoq_regression() -> None:
    test_image = "input-data/peppers.tiff"
    qualities = [5, 20, 40, 86]
    bitstream_name = 'bitstream.jpg'
    md5_expected = ["377e5ab80643ee5b67db7842d57cb80b", "e42b14051e5a8b0c1e6dd596de8cedb5", "2f3644a43bc570ad6eaadce5b003a2e8", "4153a031d9621d15aaf95cb20b0d3590"]

    im = cv2.imread(test_image, cv2.IMREAD_UNCHANGED)
    im_ycbcr = rgb_to_ycbcr_bt709(im[:, :, 2], im[:, :, 1], im[:, :, 0])

    for q, md5e in zip(qualities, md5_expected):
        _, _, _, _, _ = jpeg_encoding_rdoq(im_ycbcr, bitstream_name, q)
        md5c = compute_md5(bitstream_name)

        assert md5e == md5c

    os.remove(bitstream_name)


def test_zero_input_produces_zero_output() -> None:
    # Arrange
    coefficients = np.zeros((8, 8))
    qm = np.ones((8, 8))
    _lambda = 0.1
    pred_dc = 0
    expected_rate = table_dc[0, 1] + table_ac[0, 1]

    # Act
    levels, distortion, rate = rdoq_8x8_plane(coefficients, qm, table_dc, table_ac, _lambda, zz_idx, r_zz_idx, pred_dc)

    # Assert
    assert np.array_equal(np.zeros((8, 8), np.int32), levels)
    assert not distortion
    assert rate == expected_rate


def test_single_dc_input_produces_single_dc_output() -> None:
    # Arrange
    dc_value = 6
    coefficients = np.zeros((8, 8))
    coefficients[0, 0] = dc_value
    qm = np.ones((8, 8))
    _lambda = 0.1
    pred_dc = 0
    expected_rate = table_dc[3, 1] + 3 + table_ac[0, 1]
    expected_levels = np.zeros((8, 8), np.int32)
    expected_levels[0, 0] = dc_value

    # Act
    levels, distortion, rate = rdoq_8x8_plane(coefficients, qm, table_dc, table_ac, _lambda, zz_idx, r_zz_idx, pred_dc)

    # Assert
    assert np.array_equal(expected_levels, levels)
    assert not distortion
    assert rate == expected_rate


def test_single_dc_input_produces_dc_minus1_output() -> None:
    # Arrange
    dc_value = 0.625
    coefficients = np.zeros((8, 8))
    coefficients[0, 0] = dc_value
    qm = 14 * np.ones((8, 8))
    _lambda = 269.588
    pred_dc = -2
    expected_rate = table_dc[1, 1] + 1 + table_ac[0, 1]
    expected_levels = np.zeros((8, 8), np.int32)
    expected_levels[0, 0] = int(dc_value / qm[0, 0]) - 1

    # Act
    levels, distortion, rate = rdoq_8x8_plane(coefficients, qm, table_dc, table_ac, _lambda, zz_idx, r_zz_idx, pred_dc)

    # Assert
    assert np.array_equal(expected_levels, levels)
    assert distortion == (dc_value - expected_levels[0, 0] * qm[0, 0])**2
    assert rate == expected_rate

# Run is expanded
# EOB is reduced
# Rate and distortion are correct
# RD cost is minimised
# Expected output
