from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from bg_removal import BackgroundRemover, PipelineConfig, load_image


class FakeBackend:
    def __init__(self, mask: Image.Image | None = None) -> None:
        self.calls = 0
        self.mask = mask or Image.linear_gradient("L").resize((8, 8))

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def model_id(self) -> str:
        return "fake/test"

    @property
    def parameter_count(self) -> int:
        return 0

    @property
    def model_size_bytes(self) -> int:
        return 0

    def predict_alpha(self, image: Image.Image) -> Image.Image:
        self.calls += 1
        assert image.mode == "RGB"
        return self.mask.copy()


def test_remove_preserves_source_resolution_and_returns_rgba() -> None:
    backend = FakeBackend(Image.fromarray(np.array([[0, 0], [255, 255]], dtype=np.uint8), mode="L"))
    remover = BackgroundRemover(backend, config=PipelineConfig(edge_blur_radius=0))
    source = Image.new("RGB", (37, 19), (10, 20, 30))
    output = remover.remove(source)
    assert output.mode == "RGBA"
    assert output.size == source.size
    assert output.getpixel((5, 5))[:3] == (10, 20, 30)
    assert output.getchannel("A").getextrema() == (0, 255)


def test_backend_is_reused_across_calls() -> None:
    backend = FakeBackend(Image.new("L", (4, 4), 128))
    remover = BackgroundRemover(backend)
    remover.remove(Image.new("RGB", (10, 10))).close()
    remover.remove(Image.new("RGB", (12, 8))).close()
    assert backend.calls == 2


def test_existing_alpha_does_not_premultiply_rgb() -> None:
    source = Image.new("RGBA", (3, 2), (100, 80, 60, 0))
    remover = BackgroundRemover(
        FakeBackend(Image.new("L", (3, 2), 255)),
        config=PipelineConfig(edge_blur_radius=0),
    )
    assert remover.remove(source).getpixel((0, 0)) == (100, 80, 60, 255)


def test_pixel_limit_is_enforced_before_backend_call() -> None:
    backend = FakeBackend()
    remover = BackgroundRemover(backend, config=PipelineConfig(max_pixels=99))
    with pytest.raises(ValueError, match="limit"):
        remover.remove(Image.new("RGB", (10, 10)))
    assert backend.calls == 0


@pytest.mark.parametrize("file_format", ["JPEG", "PNG", "WEBP"])
def test_load_image_supported_formats(file_format: str) -> None:
    buffer = BytesIO()
    Image.new("RGB", (7, 5), "red").save(buffer, format=file_format)
    buffer.seek(0)
    loaded = load_image(buffer)
    assert loaded.size == (7, 5)
    assert loaded.mode == "RGB"


def test_exif_orientation_is_applied() -> None:
    pixels = np.zeros((3, 2, 3), dtype=np.uint8)
    pixels[0, 0] = (255, 0, 0)
    source = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[274] = 6
    buffer = BytesIO()
    source.save(buffer, format="JPEG", exif=exif, quality=100, subsampling=0)
    buffer.seek(0)
    loaded = load_image(buffer)
    assert loaded.size == (3, 2)
    rotated_pixel = loaded.getpixel((2, 0))
    assert rotated_pixel[0] > rotated_pixel[1] and rotated_pixel[0] > rotated_pixel[2]


def test_load_image_rejects_oversized_image_before_full_decode() -> None:
    buffer = BytesIO()
    Image.new("RGB", (11, 10)).save(buffer, format="PNG")
    buffer.seek(0)
    with pytest.raises(ValueError, match="limit"):
        load_image(buffer, max_pixels=100)


def test_alpha_thresholds_snap_only_extremes() -> None:
    values = np.array([[0, 1, 2, 127, 253, 254, 255]], dtype=np.uint8)
    remover = BackgroundRemover(
        FakeBackend(Image.fromarray(values, mode="L")),
        config=PipelineConfig(edge_blur_radius=0),
    )
    alpha = np.asarray(remover.remove(Image.new("RGB", (7, 1))).getchannel("A"))
    assert alpha.tolist() == [[0, 0, 2, 127, 253, 255, 255]]
