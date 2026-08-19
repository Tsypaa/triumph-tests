"""Environment-backed API configuration without import-time side effects."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _origins(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    model: str = "ZhengPeng7/BiRefNet"
    model_revision: str | None = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
    device: str = "auto"
    max_upload_size_bytes: int = 15 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    inference_resolution: int = 1024
    cors_origins: tuple[str, ...] = ()
    log_level: str = "INFO"
    max_concurrent_inferences: int = 1
    inference_queue_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("DEVICE must be one of: auto, cpu, cuda")
        for name in (
            "max_upload_size_bytes",
            "max_image_pixels",
            "inference_resolution",
            "max_concurrent_inferences",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.inference_queue_timeout_seconds <= 0:
            raise ValueError("inference_queue_timeout_seconds must be positive")
        if self.log_level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("LOG_LEVEL must be CRITICAL, ERROR, WARNING, INFO, or DEBUG")

    @classmethod
    def from_env(cls) -> Settings:
        revision = os.getenv("MODEL_REVISION", "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4").strip()
        return cls(
            model=os.getenv("MODEL", "ZhengPeng7/BiRefNet").strip(),
            model_revision=revision or None,
            device=os.getenv("DEVICE", "auto").strip().lower(),
            max_upload_size_bytes=_positive_int("MAX_UPLOAD_SIZE_BYTES", 15 * 1024 * 1024),
            max_image_pixels=_positive_int("MAX_IMAGE_PIXELS", 40_000_000),
            inference_resolution=_positive_int("INFERENCE_RESOLUTION", 1024),
            cors_origins=_origins(os.getenv("CORS_ORIGINS", "")),
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            max_concurrent_inferences=_positive_int("MAX_CONCURRENT_INFERENCES", 1),
            inference_queue_timeout_seconds=_positive_float(
                "INFERENCE_QUEUE_TIMEOUT_SECONDS", 30.0
            ),
        )
