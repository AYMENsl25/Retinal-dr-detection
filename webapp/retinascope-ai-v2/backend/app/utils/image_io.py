from base64 import b64encode
from io import BytesIO

import numpy as np
from PIL import Image


def decode_image_bytes(data: bytes, size: int | None = 512) -> Image.Image:
    image = Image.open(BytesIO(data)).convert("RGB")
    if size is None:
        return image
    return image.resize((size, size), Image.Resampling.LANCZOS)


def pil_to_data_url(image: Image.Image, mime: str = "image/png") -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def array_to_pil_rgb(array: np.ndarray) -> Image.Image:
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB")
    return Image.fromarray(array.astype(np.uint8), mode="RGB")
