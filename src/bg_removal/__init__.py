"""Public API for the background-removal inference layer."""

from .backends import BiRefNetBackend, MaskBackend
from .core import (
    BackgroundRemover,
    ImageTooLargeError,
    InvalidImageError,
    PipelineConfig,
    UnsupportedImageFormatError,
    load_image,
)

__all__ = [
    "BackgroundRemover",
    "BiRefNetBackend",
    "ImageTooLargeError",
    "InvalidImageError",
    "MaskBackend",
    "PipelineConfig",
    "UnsupportedImageFormatError",
    "load_image",
]
