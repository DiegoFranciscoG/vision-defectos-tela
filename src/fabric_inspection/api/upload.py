"""Uploaded image validation: size, real format (by content) and pixel count (rule R12, API4)."""

import hashlib
import io
import warnings

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from fabric_inspection.exceptions import ImageTooLargeError, InvalidImageError

ALLOWED_FORMATS = frozenset({"PNG", "JPEG"})
_CHUNK = 64 * 1024


async def read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Read at most max_bytes; the client-supplied filename and Content-Type are ignored."""
    data = bytearray()
    while chunk := await upload.read(_CHUNK):
        data.extend(chunk)
        if len(data) > max_bytes:
            raise ImageTooLargeError(f"The file exceeds {max_bytes // (1024 * 1024)} MB")
    if not data:
        raise InvalidImageError("The file is empty")
    return bytes(data)


def decode_image(data: bytes, max_side: int) -> tuple[Image.Image, str]:
    """Decode PNG/JPEG safely and return the RGB image with the SHA-256 of the raw bytes."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                if probe.format not in ALLOWED_FORMATS:
                    raise InvalidImageError("Only PNG and JPEG images are accepted")
                width, height = probe.size
                if width > max_side or height > max_side:
                    raise ImageTooLargeError(f"Image sides must be at most {max_side} px")
                if width < 32 or height < 32:
                    raise InvalidImageError("Image sides must be at least 32 px")
                probe.verify()
            # verify() leaves the file unusable, so decode again (Pillow docs, #31).
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                rgb = image.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ImageTooLargeError("The image has too many pixels") from error
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as error:
        raise InvalidImageError("The file is not a valid PNG or JPEG image") from error
    return rgb, hashlib.sha256(data).hexdigest()
