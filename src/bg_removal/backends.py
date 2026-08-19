"""Model backends. This module has no HTTP or file-upload concerns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image


@runtime_checkable
class MaskBackend(Protocol):
    """Backend contract: return a continuous alpha mask at model resolution."""

    @property
    def device(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    @property
    def parameter_count(self) -> int | None: ...

    @property
    def model_size_bytes(self) -> int | None: ...

    def predict_alpha(self, image: Image.Image) -> Image.Image: ...


def _select_output_tensor(output: Any) -> Any:
    """Extract the final prediction from common HF/custom model outputs."""
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict):
        for key in ("logits", "pred", "out"):
            if key in output:
                return output[key]
    if isinstance(output, (list, tuple)):
        if not output:
            raise RuntimeError("Model returned an empty output sequence")
        return _select_output_tensor(output[-1])
    return output


@dataclass(frozen=True)
class BiRefNetConfig:
    model_id: str = "ZhengPeng7/BiRefNet"
    revision: str | None = None
    input_size: int = 1024
    device: str | None = None


class BiRefNetBackend:
    """One-time-loaded Hugging Face BiRefNet backend.

    The official repository requires remote model code. For production, pass a
    reviewed immutable commit through ``revision`` and pre-populate the HF cache.
    """

    def __init__(
        self,
        model_id: str = "ZhengPeng7/BiRefNet",
        *,
        revision: str | None = None,
        input_size: int = 1024,
        device: str | None = None,
    ) -> None:
        if input_size < 64:
            raise ValueError("input_size must be at least 64 pixels")
        try:
            import torch
            from transformers import AutoModelForImageSegmentation
        except ImportError as exc:
            raise RuntimeError(
                "BiRefNet dependencies are missing. Install the project with "
                "`pip install -e .[birefnet]`."
            ) from exc

        self._torch = torch
        self._model_id = model_id
        self._input_size = input_size
        self._device = self._resolve_device(device)
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": self._dtype,
        }
        if revision is not None:
            load_kwargs["revision"] = revision

        self._model = AutoModelForImageSegmentation.from_pretrained(model_id, **load_kwargs)
        self._model.eval().to(self._device)

        params = list(self._model.parameters())
        self._parameter_count = sum(p.numel() for p in params)
        self._model_size_bytes = sum(p.numel() * p.element_size() for p in params)

    def _resolve_device(self, requested: str | None) -> str:
        torch = self._torch
        if requested is None or requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        requested = requested.lower()
        if requested not in {"cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return requested

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    @property
    def model_size_bytes(self) -> int:
        return self._model_size_bytes

    @property
    def input_size(self) -> int:
        return self._input_size

    def _to_tensor(self, image: Image.Image) -> Any:
        resized = image.resize((self._input_size, self._input_size), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        # copy() makes the transposed view contiguous and writable for torch.
        tensor = self._torch.from_numpy(array.transpose(2, 0, 1).copy())
        return tensor.unsqueeze(0).to(device=self._device, dtype=self._dtype)

    def predict_alpha(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            raise ValueError("BiRefNetBackend expects an RGB PIL image")
        torch = self._torch
        tensor = self._to_tensor(image)
        try:
            # Autocast is useful on CUDA; CPU stays in stable FP32.
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=self._device,
                    dtype=self._dtype,
                    enabled=self._device == "cuda",
                ),
            ):
                raw = self._model(tensor)
                prediction = _select_output_tensor(raw)
                if prediction.ndim == 4:
                    prediction = prediction[0, 0]
                elif prediction.ndim == 3:
                    prediction = prediction[0]
                if prediction.ndim != 2:
                    raise RuntimeError(
                        f"Unexpected BiRefNet output shape: {tuple(prediction.shape)}"
                    )
                # Official checkpoints return logits, but tolerate an already
                # sigmoid-normalized third-party export.
                min_value = float(prediction.detach().min())
                max_value = float(prediction.detach().max())
                if min_value < 0.0 or max_value > 1.0:
                    prediction = prediction.sigmoid()
                mask = prediction.float().clamp_(0, 1).mul_(255).byte().cpu().numpy()
        finally:
            del tensor

        return Image.fromarray(mask, mode="L")
