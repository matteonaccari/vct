import os
from hashlib import md5

import cv2
import numpy as np
from ct import rgb_to_ycbcr_bt709
from encoder_rdoq import jpeg_encoding_rdoq
from entropy import (encode_block, expand_huffman_table, get_zigzag_scan,
                     luma_ac_bits, luma_ac_values, luma_dc_bits,
                     luma_dc_values)
from quantiser import rd_rl_pair, rdoq_8x8_plane

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


def test_run_length_pair_expands() -> None:
    # Arrange
    coefficients = np.zeros((64))
    coefficients[0], coefficients[2], coefficients[4] = 16, 1, -4
    coefficients = np.reshape(coefficients[r_zz_idx], (8, 8))
    qm = np.ones((8, 8))
    _lambda = 1.2
    pred_dc = 0
    expected_levels = np.zeros((64), np.int32)
    expected_levels[0], expected_levels[4] = 15, -3
    expected_levels = np.reshape(expected_levels[r_zz_idx], (8, 8))
    expected_distortion = 1 + 1 + 1
    expected_rate = table_dc[4, 1] + 4 + 11 + table_ac[0, 1]

    # Act
    levels, distortion, rate = rdoq_8x8_plane(coefficients, qm, table_dc, table_ac, _lambda, zz_idx, r_zz_idx, pred_dc)

    # Assert
    assert np.array_equal(expected_levels, levels)
    assert np.isclose(distortion, expected_distortion)
    assert rate == expected_rate


def test_eob_reduces() -> None:
    # Arrange
    coefficients = np.zeros((64))
    coefficients[0], coefficients[2], coefficients[4] = 16, 1, -4
    coefficients = np.reshape(coefficients[r_zz_idx], (8, 8))
    qm = np.ones((8, 8))
    _lambda = 1.8
    pred_dc = 0
    eob_no_rdoq = 4

    # Act
    levels, _, _ = rdoq_8x8_plane(coefficients, qm, table_dc, table_ac, _lambda, zz_idx, r_zz_idx, pred_dc)
    levels_zz = levels.flatten()[zz_idx]
    idx = np.where(levels_zz[-1:0:-1] != 0)
    eob_rdoq = 63 - idx[0][0] if len(idx[0]) else 0

    # Assert
    assert eob_no_rdoq > eob_rdoq


def test_distortion_is_computed_correctly() -> None:
    # Arrange
    coefficients = np.array([[-320.62, -85.82, -100.15, -84.02, -83.62, -56.78, -33.07, -16.53],
                             [-70.46, 17.94, 7.28, 11.05, 14.25, 3.55, 0.07, 2.68],
                             [-81.56, 9.43, 14.78, 18.86, 11.11, 10.14, -2.49, 1.23],
                             [-62.03, 8.17, 11.23, 3.48, 15.48, 7.01, 8.00, -0.56],
                             [-55.88, 10.99, 3.27, 6.23, 10.62, 5.98, 14.75, 6.25],
                             [-52.31, 3.16, 11.31, 4.98, 13.00, 6.84, 5.04, -0.09],
                             [-31.30, 8.82, 7.26, 6.05, -0.95, 1.43, -3.78, 13.38],
                             [-11.69, 3.65, 5.07, 0.21, -0.69, 4.46, -1.18, -7.26]])
    qm = np.array([[14, 10, 9, 14, 22, 36, 46, 55],
                   [11, 11, 13, 17, 23, 52, 54, 50],
                   [13, 12, 14, 22, 36, 51, 62, 50],
                   [13, 15, 20, 26, 46, 78, 72, 56],
                   [16, 20, 33, 50, 61, 98, 93, 69],
                   [22, 32, 50, 58, 73, 94, 102, 83],
                   [44, 58, 70, 78, 93, 109, 108, 91],
                   [65, 83, 86, 88, 101, 90, 93, 89]], np.int32)
    _lambda = 0.1 * np.mean(qm)**2
    pred_dc = 0

    # Act
    levels, distortion, _ = rdoq_8x8_plane(coefficients, qm, table_dc, table_ac, _lambda, zz_idx, r_zz_idx, pred_dc)
    expected_distortion = np.sum(np.square(coefficients - levels * qm))

    # Assert
    assert np.isclose(distortion, expected_distortion)


def test_rate_is_computed_correctly() -> None:
    # Arrange
    coefficients = np.array([[-320.62, -85.82, -100.15, -84.02, -83.62, -56.78, -33.07, -16.53],
                             [-70.46, 17.94, 7.28, 11.05, 14.25, 3.55, 0.07, 2.68],
                             [-81.56, 9.43, 14.78, 18.86, 11.11, 10.14, -2.49, 1.23],
                             [-62.03, 8.17, 11.23, 3.48, 15.48, 7.01, 8.00, -0.56],
                             [-55.88, 10.99, 3.27, 6.23, 10.62, 5.98, 14.75, 6.25],
                             [-52.31, 3.16, 11.31, 4.98, 13.00, 6.84, 5.04, -0.09],
                             [-31.30, 8.82, 7.26, 6.05, -0.95, 1.43, -3.78, 13.38],
                             [-11.69, 3.65, 5.07, 0.21, -0.69, 4.46, -1.18, -7.26]])
    qm = np.array([[14, 10, 9, 14, 22, 36, 46, 55],
                   [11, 11, 13, 17, 23, 52, 54, 50],
                   [13, 12, 14, 22, 36, 51, 62, 50],
                   [13, 15, 20, 26, 46, 78, 72, 56],
                   [16, 20, 33, 50, 61, 98, 93, 69],
                   [22, 32, 50, 58, 73, 94, 102, 83],
                   [44, 58, 70, 78, 93, 109, 108, 91],
                   [65, 83, 86, 88, 101, 90, 93, 89]], np.int32)
    _lambda = 0.1 * np.mean(qm)**2
    pred_dc = 0

    # Act
    levels, _, rate = rdoq_8x8_plane(coefficients, qm, table_dc, table_ac, _lambda, zz_idx, r_zz_idx, pred_dc)
    _, expected_rate = encode_block(levels.flatten()[zz_idx], pred_dc, table_dc, table_ac)

    # Assert
    assert rate == expected_rate


def test_rd_cost_is_minimised() -> None:
    # Arrange
    coefficients = np.array([[-320.62, -85.82, -100.15, -84.02, -83.62, -56.78, -33.07, -16.53],
                             [-70.46, 17.94, 7.28, 11.05, 14.25, 3.55, 0.07, 2.68],
                             [-81.56, 9.43, 14.78, 18.86, 11.11, 10.14, -2.49, 1.23],
                             [-62.03, 8.17, 11.23, 3.48, 15.48, 7.01, 8.00, -0.56],
                             [-55.88, 10.99, 3.27, 6.23, 10.62, 5.98, 14.75, 6.25],
                             [-52.31, 3.16, 11.31, 4.98, 13.00, 6.84, 5.04, -0.09],
                             [-31.30, 8.82, 7.26, 6.05, -0.95, 1.43, -3.78, 13.38],
                             [-11.69, 3.65, 5.07, 0.21, -0.69, 4.46, -1.18, -7.26]])
    qm = np.array([[14, 10, 9, 14, 22, 36, 46, 55],
                   [11, 11, 13, 17, 23, 52, 54, 50],
                   [13, 12, 14, 22, 36, 51, 62, 50],
                   [13, 15, 20, 26, 46, 78, 72, 56],
                   [16, 20, 33, 50, 61, 98, 93, 69],
                   [22, 32, 50, 58, 73, 94, 102, 83],
                   [44, 58, 70, 78, 93, 109, 108, 91],
                   [65, 83, 86, 88, 101, 90, 93, 89]], np.int32)
    _lambda = 0.1 * np.mean(qm)**2
    pred_dc = 0
    levels_no_rdoq = np.divide(coefficients + 0.5, qm).astype(np.int32)
    distortion_no_rdoq = np.sum(np.square(coefficients - levels_no_rdoq * qm))
    _, rate_no_rdoq = encode_block(levels_no_rdoq.flatten()[zz_idx], pred_dc, table_dc, table_ac)
    cost_no_rdoq = distortion_no_rdoq + _lambda * rate_no_rdoq

    # Act
    levels_rdoq, distortion_rdoq, rate_rdoq = rdoq_8x8_plane(coefficients, qm, table_dc, table_ac, _lambda, zz_idx, r_zz_idx, pred_dc)
    cost_rdoq = distortion_rdoq + _lambda * rate_rdoq

    # Assert
    assert cost_rdoq < cost_no_rdoq


def test_run_length_data_are_correct() -> None:
    # Arrange
    coefficients = np.zeros((64))
    coefficients[-1] = 1
    levels = np.zeros((64), np.int32)
    levels[-1] = 1
    qm = np.ones((64))
    expected_rate = (62 // 16) * table_ac[15 << 4, 1] + table_ac[225, 1] + 1

    # Act
    rate, distortion, distortion_value, run_length = rd_rl_pair(1, coefficients, levels, qm, table_ac)

    # Assert
    assert rate == expected_rate
    assert not distortion
    assert not distortion_value
    assert run_length == 62


def test_run_length_data_are_correct_over_multiple_pairs() -> None:
    # Arrange
    coefficients = np.zeros((64))
    coefficients[9], coefficients[21], coefficients[22], coefficients[40], coefficients[43] = 5, 2, 5, 1, 5
    qm = 3 * np.ones((64))
    levels = np.divide(coefficients, qm).astype(np.int32)
    expected_rate = [table_ac[(128 + 1), 1] + 1, table_ac[(192 + 1), 1] + 1, table_ac[(64 + 1), 1] + 1 + table_ac[15 << 4, 1]]
    expected_distortion = [4, 4 + 4, 1 + 4]
    expected_run_length = [8, 12, 20]
    run_start = [1, 10, 23]
    expected_distortion_single = 4

    # Act
    for start_idx, er, ed, erl in zip(run_start, expected_rate, expected_distortion, expected_run_length):
        rate, distortion, distortion_value, run_length = rd_rl_pair(start_idx, coefficients, levels, qm, table_ac)

        # Assert
        assert rate == er
        assert np.isclose(distortion, ed)
        assert np.isclose(distortion_value, expected_distortion_single)
        assert run_length == erl
