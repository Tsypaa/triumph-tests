"""Image processing and model-independent background-removal pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from .backends import MaskBackend

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageProcessingError(ValueError):
    """Base class for safe, user-caused image processing failures."""


class UnsupportedImageFormatError(ImageProcessingError):
    pass


class ImageTooLargeError(ImageProcessingError):
    pass


class InvalidImageError(ImageProcessingError):
    pass


@dataclass(frozen=True)
class PipelineConfig:
    """Safe image/output post-processing settings."""

    max_pixels: int = 100_000_000
    edge_blur_radius: float = 0.35
    snap_transparent_below: int = 1
    snap_opaque_above: int = 254

    def __post_init__(self) -> None:
        if self.max_pixels <= 0:
            raise ValueError("max_pixels must be positive")
        if self.edge_blur_radius < 0:
            raise ValueError("edge_blur_radius cannot be negative")
        if not 0 <= self.snap_transparent_below <= 255:
            raise ValueError("snap_transparent_below must be in [0, 255]")
        if not 0 <= self.snap_opaque_above <= 255:
            raise ValueError("snap_opaque_above must be in [0, 255]")
        if self.snap_transparent_below >= self.snap_opaque_above:
            raise ValueError("transparent threshold must be below opaque threshold")


def load_image(source: str | Path | BinaryIO, *, max_pixels: int = 100_000_000) -> Image.Image:
    """Decode a supported image, apply EXIF orientation, and detach from input IO."""
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    try:
        with Image.open(source) as opened:
            if opened.format not in SUPPORTED_FORMATS:
                raise UnsupportedImageFormatError(
                    f"Unsupported image format {opened.format!r}; expected JPEG, PNG, or WebP"
                )
            if opened.width * opened.height > max_pixels:
                raise ImageTooLargeError(
                    f"image has {opened.width * opened.height:,} pixels; limit is {max_pixels:,}"
                )
            oriented = ImageOps.exif_transpose(opened)
            oriented.load()
            result = oriented.copy()
            result.format = opened.format
            return result
    except (UnsupportedImageFormatError, ImageTooLargeError):
        raise
    except Image.DecompressionBombError as exc:
        raise ImageTooLargeError("Image exceeds Pillow's safe pixel limit") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError("Uploaded file is not a valid image") from exc


def _as_rgb(image: Image.Image) -> Image.Image:
    """Create the RGB source used both for inference and final RGBA pixels."""
    if image.mode == "RGB":
        return image.copy()
    if image.mode == "RGBA":
        # Existing alpha is intentionally not used: the output alpha represents
        # the requested new foreground mask, while RGB remains unpremultiplied.
        return image.convert("RGB")
    return image.convert("RGB")


def _refine_alpha(mask: Image.Image, size: tuple[int, int], config: PipelineConfig) -> Image.Image:
    if mask.mode != "L":
        mask = mask.convert("L")
    if mask.size != size:
        # Lanczos retains fine hair better than nearest/bilinear and provides
        # anti-aliased boundaries when returning to the source resolution.
        mask = mask.resize(size, Image.Resampling.LANCZOS)
    if config.edge_blur_radius:
        mask = mask.filter(ImageFilter.GaussianBlur(config.edge_blur_radius))

    values = np.asarray(mask, dtype=np.uint8).copy()
    values[values <= config.snap_transparent_below] = 0
    values[values >= config.snap_opaque_above] = 255
    return Image.fromarray(values, mode="L")


class BackgroundRemover:
    """Reusable inference service independent from HTTP and persistence.

    The backend is constructed by the caller and retained, so model weights are
    loaded exactly once per ``BackgroundRemover`` instance.
    """

    def __init__(self, backend: MaskBackend, *, config: PipelineConfig | None = None) -> None:
        if not isinstance(backend, MaskBackend):
            raise TypeError("backend must implement MaskBackend")
        self.backend = backend
        self.config = config or PipelineConfig()

    def remove(self, image: Image.Image) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        oriented = ImageOps.exif_transpose(image)
        width, height = oriented.size
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        if width * height > self.config.max_pixels:
            raise ValueError(
                f"image has {width * height:,} pixels; limit is {self.config.max_pixels:,}"
            )

        rgb = _as_rgb(oriented)
        alpha_at_model_size = self.backend.predict_alpha(rgb)
        alpha = _refine_alpha(alpha_at_model_size, rgb.size, self.config)
        output = rgb.convert("RGBA")
        output.putalpha(alpha)
        return output
