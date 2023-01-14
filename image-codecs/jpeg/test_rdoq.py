from hashlib import md5
import os
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
    md5_expected = ["2ed31c0f570c7ca4ea7482e9ff0737bf", "4b4d66a4ba82206e18764e328b65bcab", "e7950cb711a629b9affcb4a1c9730d82", "e33b06b070c426b2c1e28e7d9d215968"]

    im = cv2.imread(test_image, cv2.IMREAD_UNCHANGED)
    im_ycbcr = rgb_to_ycbcr_bt709(im[:, :, 0], im[:, :, 1], im[:, :, 2])

    for q, md5e in zip(qualities, md5_expected):
        _, _, _, _, _ = jpeg_encoding_rdoq(im_ycbcr, bitstream_name, q)
        md5c = compute_md5(bitstream_name)

        assert md5e == md5c

    os.remove(bitstream_name)
