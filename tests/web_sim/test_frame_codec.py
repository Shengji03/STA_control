import io

import numpy as np
from PIL import Image
import pytest

from src.web_sim.frame_codec import encode_jpeg


def test_encode_jpeg_returns_decodable_jpeg_bytes():
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    rgb[:, :, 0] = 180

    encoded = encode_jpeg(rgb, quality=82)

    assert encoded[:2] == b"\xff\xd8"
    image = Image.open(io.BytesIO(encoded))
    assert image.format == "JPEG"
    assert image.size == (16, 12)


def test_encode_jpeg_rejects_non_rgb_arrays():
    grayscale = np.zeros((12, 16), dtype=np.uint8)

    with pytest.raises(ValueError, match="RGB"):
        encode_jpeg(grayscale)


def test_encode_jpeg_rejects_non_uint8_arrays():
    rgb = np.zeros((12, 16, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="uint8"):
        encode_jpeg(rgb)
