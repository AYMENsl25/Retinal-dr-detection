from io import BytesIO

from PIL import Image, ImageOps

TARGET_SIZE = (512, 512)


def load_fundus_image(contents: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(contents))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except Exception as exc:
        raise ValueError("Could not decode the uploaded image.") from exc


def resize_for_model(image: Image.Image, size: tuple[int, int] = TARGET_SIZE) -> Image.Image:
    return image.resize(size, Image.Resampling.BILINEAR)

