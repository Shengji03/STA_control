import io

import numpy as np
from PIL import Image


def encode_jpeg(rgb: np.ndarray, quality: int = 80) -> bytes:
    """Encode an RGB uint8 image array as JPEG bytes."""
    if rgb.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB image data")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3)")

    safe_quality = max(1, min(int(quality), 95))
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(
        buffer,
        format="JPEG",
        quality=safe_quality,
        optimize=False,
        progressive=False,
    )
    return buffer.getvalue()

