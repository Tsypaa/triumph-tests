"""Production FastAPI application for local background removal."""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from bg_removal import (
    BackgroundRemover,
    BiRefNetBackend,
    ImageTooLargeError,
    InvalidImageError,
    PipelineConfig,
    UnsupportedImageFormatError,
    load_image,
)

from .config import Settings
from .errors import ApiError

LOGGER = logging.getLogger("bg_removal.api")
MIME_TO_FORMAT = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
READ_CHUNK_SIZE = 1024 * 1024
RemoverFactory = Callable[[Settings], BackgroundRemover]


class _RequestBodyTooLarge(Exception):
    pass


class UploadSizeLimitMiddleware:
    """Reject oversized request bodies before multipart can spool them to disk."""

    def __init__(self, app: Any, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not (
            scope["method"] == "POST" and scope["path"] == "/api/remove-background"
        ):
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                if int(raw_length) > self.max_body_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                pass

        received = 0

        async def limited_receive() -> Any:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope: Any, receive: Any, send: Any) -> None:
        error = ApiError(413, "FILE_TOO_LARGE", "Uploaded file is too large.")
        await JSONResponse(status_code=413, content=error.body())(scope, receive, send)


def _default_remover_factory(settings: Settings) -> BackgroundRemover:
    backend = BiRefNetBackend(
        settings.model,
        revision=settings.model_revision,
        input_size=settings.inference_resolution,
        device=settings.device,
    )
    return BackgroundRemover(
        backend,
        config=PipelineConfig(max_pixels=settings.max_image_pixels),
    )


async def _read_upload(upload: UploadFile, limit: int) -> bytes:
    data = io.BytesIO()
    total = 0
    try:
        while True:
            chunk = await upload.read(min(READ_CHUNK_SIZE, limit - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ApiError(
                    413,
                    "FILE_TOO_LARGE",
                    f"Uploaded file exceeds the {limit}-byte limit.",
                )
            data.write(chunk)
    finally:
        await upload.close()
    return data.getvalue()


def _backend_info(app: FastAPI) -> dict[str, Any]:
    remover: BackgroundRemover | None = app.state.remover
    settings: Settings = app.state.settings
    if remover is None:
        return {
            "model": settings.model,
            "backend": "BiRefNetBackend",
            "device": settings.device,
            "inference_resolution": settings.inference_resolution,
            "parameter_count": None,
            "model_size_bytes": None,
            "ready": False,
        }
    backend = remover.backend
    return {
        "model": backend.model_id,
        "backend": type(backend).__name__,
        "device": backend.device,
        "inference_resolution": settings.inference_resolution,
        "parameter_count": backend.parameter_count,
        "model_size_bytes": backend.model_size_bytes,
        "ready": True,
    }


def create_app(
    *,
    settings: Settings | None = None,
    remover_factory: RemoverFactory | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    factory = remover_factory or _default_remover_factory
    logging.getLogger().setLevel(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.settings = resolved_settings
        application.state.remover = None
        application.state.model_error = None
        application.state.inference_semaphore = asyncio.Semaphore(
            resolved_settings.max_concurrent_inferences
        )
        try:
            application.state.remover = await asyncio.to_thread(factory, resolved_settings)
            LOGGER.info(
                "Background-removal model ready: model=%s device=%s",
                application.state.remover.backend.model_id,
                application.state.remover.backend.device,
            )
        except Exception as exc:
            application.state.model_error = exc
            LOGGER.exception("Background-removal model failed to initialize")
        yield
        application.state.remover = None

    application = FastAPI(
        title="Background Removal API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Multipart framing is bounded separately from the exact decoded file size.
    application.add_middleware(
        UploadSizeLimitMiddleware,
        max_body_bytes=resolved_settings.max_upload_size_bytes + 64 * 1024,
    )

    if resolved_settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @application.exception_handler(ApiError)
    async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.body())

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, __: RequestValidationError) -> JSONResponse:
        error = ApiError(422, "INVALID_IMAGE", "A multipart image file is required.")
        return JSONResponse(status_code=error.status_code, content=error.body())

    @application.exception_handler(Exception)
    async def internal_error_handler(_: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled API error", exc_info=exc)
        error = ApiError(500, "INTERNAL_ERROR", "An internal error occurred.")
        return JSONResponse(status_code=error.status_code, content=error.body())

    @application.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        info = _backend_info(request.app)
        return {
            "status": "ok" if info["ready"] else "degraded",
            "model_ready": info["ready"],
            "device": info["device"],
            "model": info["model"],
        }

    @application.get("/api/model-info")
    async def model_info(request: Request) -> dict[str, Any]:
        info = _backend_info(request.app)
        info.update(
            {
                "model_revision": resolved_settings.model_revision,
                "max_upload_size_bytes": resolved_settings.max_upload_size_bytes,
                "max_image_pixels": resolved_settings.max_image_pixels,
                "max_concurrent_inferences": resolved_settings.max_concurrent_inferences,
            }
        )
        return info

    @application.post("/api/remove-background")
    async def remove_background(request: Request, file: Annotated[UploadFile, File()]) -> Response:
        declared_mime = (file.content_type or "").split(";", 1)[0].lower().strip()
        if declared_mime not in MIME_TO_FORMAT:
            await file.close()
            raise ApiError(
                415,
                "INVALID_FILE_TYPE",
                "Only JPEG, PNG, and WebP images are supported.",
            )

        payload = await _read_upload(file, resolved_settings.max_upload_size_bytes)
        try:
            image = load_image(io.BytesIO(payload), max_pixels=resolved_settings.max_image_pixels)
        except ImageTooLargeError as exc:
            raise ApiError(
                413,
                "IMAGE_TOO_LARGE",
                f"Image exceeds the {resolved_settings.max_image_pixels}-pixel limit.",
            ) from exc
        except UnsupportedImageFormatError as exc:
            raise ApiError(
                415,
                "INVALID_FILE_TYPE",
                "Only JPEG, PNG, and WebP images are supported.",
            ) from exc
        except InvalidImageError as exc:
            raise ApiError(422, "INVALID_IMAGE", "Uploaded file is not a valid image.") from exc

        if image.format != MIME_TO_FORMAT[declared_mime]:
            image.close()
            raise ApiError(
                415,
                "INVALID_FILE_TYPE",
                "Declared Content-Type does not match the uploaded image.",
            )

        remover: BackgroundRemover | None = request.app.state.remover
        if remover is None:
            image.close()
            raise ApiError(503, "MODEL_ERROR", "The model is not ready.")

        semaphore: asyncio.Semaphore = request.app.state.inference_semaphore
        acquired = False
        try:
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=resolved_settings.inference_queue_timeout_seconds,
            )
            acquired = True
        except TimeoutError as exc:
            image.close()
            raise ApiError(503, "MODEL_ERROR", "The model is currently busy.") from exc

        try:
            try:
                result = await asyncio.to_thread(remover.remove, image)
            except Exception as exc:
                LOGGER.exception("Model inference failed")
                raise ApiError(500, "MODEL_ERROR", "Model inference failed.") from exc
        finally:
            image.close()
            if acquired:
                semaphore.release()

        output = io.BytesIO()
        try:
            result.save(output, format="PNG", compress_level=6)
        finally:
            result.close()

        return Response(
            content=output.getvalue(),
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": 'attachment; filename="removed-background.png"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    return application


app = create_app()
