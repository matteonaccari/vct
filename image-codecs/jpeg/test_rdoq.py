import os
from hashlib import md5

import cv2

from ct import rgb_to_ycbcr_bt709
from encoder_rdoq import jpeg_encoding_rdoq


def compute_md5(file_name: str) -> str:
    with open(file_name, "rb") as fh:
        data = fh.read()

    return md5(data).hexdigest()


def test_rdoq_regression() -> None:
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
