from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bg_removal import BackgroundRemover, PipelineConfig
from bg_removal.api import Settings, create_app


class FakeBackend:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def model_id(self) -> str:
        return "fake/background-remover"

    @property
    def parameter_count(self) -> int:
        return 123

    @property
    def model_size_bytes(self) -> int:
        return 492

    def predict_alpha(self, image: Image.Image) -> Image.Image:
        if self.fail:
            raise RuntimeError("private backend details")
        return Image.new("L", (8, 8), 160)


def image_bytes(file_format: str, size: tuple[int, int] = (13, 7)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (20, 40, 60)).save(buffer, format=file_format)
    return buffer.getvalue()


def client_for(
    *,
    fail: bool = False,
    max_upload_size_bytes: int = 1024 * 1024,
    max_image_pixels: int = 1_000_000,
) -> TestClient:
    settings = Settings(
        model="fake/background-remover",
        model_revision="test-revision",
        device="cpu",
        max_upload_size_bytes=max_upload_size_bytes,
        max_image_pixels=max_image_pixels,
        inference_resolution=512,
    )

    def factory(_: Settings) -> BackgroundRemover:
        return BackgroundRemover(
            FakeBackend(fail=fail),
            config=PipelineConfig(max_pixels=max_image_pixels),
        )

    return TestClient(create_app(settings=settings, remover_factory=factory))


def error_code(response: object) -> str:
    return response.json()["error"]["code"]  # type: ignore[attr-defined,no-any-return]


def test_health_and_model_info() -> None:
    with client_for() as client:
        health = client.get("/api/health")
        info = client.get("/api/model-info")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "model_ready": True,
        "device": "cpu",
        "model": "fake/background-remover",
    }
    assert info.status_code == 200
    assert info.json()["backend"] == "FakeBackend"
    assert info.json()["inference_resolution"] == 512
    assert info.json()["parameter_count"] == 123
    assert info.json()["model_size_bytes"] == 492


@pytest.mark.parametrize(
    ("file_format", "mime"),
    [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")],
)
def test_supported_image_returns_source_sized_rgba_png(file_format: str, mime: str) -> None:
    source = image_bytes(file_format, (17, 9))
    with client_for() as client:
        response = client.post(
            "/api/remove-background",
            files={"file": ("untrusted.exe", source, mime)},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    with Image.open(BytesIO(response.content)) as output:
        assert output.format == "PNG"
        assert output.mode == "RGBA"
        assert output.size == (17, 9)
        assert output.getchannel("A").getextrema() == (160, 160)


def test_corrupted_image() -> None:
    with client_for() as client:
        response = client.post(
            "/api/remove-background",
            files={"file": ("image.png", b"not a png", "image/png")},
        )
    assert response.status_code == 422
    assert error_code(response) == "INVALID_IMAGE"


def test_unsupported_format() -> None:
    with client_for() as client:
        response = client.post(
            "/api/remove-background",
            files={"file": ("image.gif", image_bytes("GIF"), "image/gif")},
        )
    assert response.status_code == 415
    assert error_code(response) == "INVALID_FILE_TYPE"


def test_declared_mime_must_match_decoded_format() -> None:
    with client_for() as client:
        response = client.post(
            "/api/remove-background",
            files={"file": ("wrong.jpg", image_bytes("PNG"), "image/jpeg")},
        )
    assert response.status_code == 415
    assert error_code(response) == "INVALID_FILE_TYPE"


def test_file_too_large() -> None:
    with client_for(max_upload_size_bytes=8) as client:
        response = client.post(
            "/api/remove-background",
            files={"file": ("large.png", b"123456789", "image/png")},
        )
    assert response.status_code == 413
    assert error_code(response) == "FILE_TOO_LARGE"


def test_request_body_guard_rejects_before_multipart_processing() -> None:
    with client_for(max_upload_size_bytes=8) as client:
        response = client.post(
            "/api/remove-background",
            files={"file": ("huge.png", b"x" * 70_000, "image/png")},
        )
    assert response.status_code == 413
    assert error_code(response) == "FILE_TOO_LARGE"


def test_missing_file_uses_structured_error() -> None:
    with client_for() as client:
        response = client.post("/api/remove-background", files={})
    assert response.status_code == 422
    assert error_code(response) == "INVALID_IMAGE"


def test_image_resolution_too_large() -> None:
    with client_for(max_image_pixels=20) as client:
        response = client.post(
            "/api/remove-background",
            files={"file": ("large.png", image_bytes("PNG", (5, 5)), "image/png")},
        )
    assert response.status_code == 413
    assert error_code(response) == "IMAGE_TOO_LARGE"


def test_model_failure_has_safe_structured_error() -> None:
    with client_for(fail=True) as client:
        response = client.post(
            "/api/remove-background",
            files={"file": ("image.png", image_bytes("PNG"), "image/png")},
        )
    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "MODEL_ERROR", "message": "Model inference failed."}
    }
    assert "private backend details" not in response.text


def test_failed_model_initialization_keeps_health_available() -> None:
    settings = Settings(model="broken/model", device="cpu")

    def broken_factory(_: Settings) -> BackgroundRemover:
        raise RuntimeError("load failure")

    app = create_app(settings=settings, remover_factory=broken_factory)
    with TestClient(app) as client:
        health = client.get("/api/health")
        response = client.post(
            "/api/remove-background",
            files={"file": ("image.png", image_bytes("PNG"), "image/png")},
        )

    assert health.status_code == 200
    assert health.json()["status"] == "degraded"
    assert health.json()["model_ready"] is False
    assert response.status_code == 503
    assert error_code(response) == "MODEL_ERROR"
