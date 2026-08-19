# Архитектура API удаления фона

## Обзор

```text
React / API client
  ↓  multipart/form-data
FastAPI
  ↓
Upload and MIME validation
  ↓
Pillow decode + format check + EXIF transpose + pixel limit
  ↓
RGB preprocessing at configured inference resolution
  ↓
BiRefNet (one shared model instance, controlled concurrency)
  ↓
Continuous alpha mask
  ↓
Lanczos source-size restore + edge cleanup
  ↓
RGBA PNG response (no-store, nosniff)
```

Production frontend собирается Vite и раздаётся nginx. nginx обслуживает SPA и
проксирует относительный `/api/*` к единственному backend service, поэтому в
браузерную сборку не вшивается адрес API. UI реализует последовательность
`Upload → Preview → Processing → Before/After → Download`, drag-and-drop и
отменяет активный `fetch` через `AbortController` при замене изображения.

```text
Browser :8080
  ├─ /             → nginx → static React assets
  └─ /api/*        → nginx → FastAPI :8000
                               ↓
                         shared BiRefNet
                               ↓
                    Hugging Face cache volume
```

HTTP-слой находится в `bg_removal.api`, а ML-логика остаётся в независимых
`BackgroundRemover` и `BiRefNetBackend`. API не вызывает внешние сервисы и не
создаёт постоянных файлов с пользовательскими изображениями.

## Lifecycle модели

FastAPI lifespan создаёт `BiRefNetBackend` ровно один раз. Используется pinned
Hugging Face revision из `MODEL_REVISION`; `DEVICE=auto` выбирает CUDA при её
наличии и CPU иначе. После startup все запросы используют общий
`BackgroundRemover`.

Если загрузка весов завершается ошибкой, API остаётся доступным для диагностики:
`/api/health` возвращает `status=degraded` и `model_ready=false`, а inference —
`503 MODEL_ERROR`. Детали исключения пишутся только в server log.

## Endpoint-ы

### `GET /api/health`

Возвращает статус, readiness, фактический device и model id. Это liveness и
diagnostic endpoint: HTTP 200 сохраняется при `degraded`, readiness задаёт поле
`model_ready`.

### `GET /api/model-info`

Возвращает model/revision, backend, device, model input resolution, число
параметров, размер parameter storage и эксплуатационные лимиты.

### `POST /api/remove-background`

Принимает поле `file` в `multipart/form-data`. Разрешены только MIME
`image/jpeg`, `image/png`, `image/webp`; MIME сверяется с форматом Pillow.
Filename игнорируется. Ответ — `image/png` с RGBA исходных display dimensions
после EXIF transpose.

## Валидация и память

- ASGI middleware ограничивает request body до `MAX_UPLOAD_SIZE_BYTES` плюс
  64 KiB на multipart framing до обработки чрезмерного body parser-ом.
- Содержимое `UploadFile` читается chunks и имеет точный независимый
  `MAX_UPLOAD_SIZE_BYTES` limit.
- Starlette может использовать автоматически удаляемый `SpooledTemporaryFile`
  для multipart. Приложение не задаёт путь, не сохраняет файл постоянно и
  всегда закрывает upload; малые файлы остаются в памяти.
- Pillow проверяет формат и dimensions до полного decode. Максимум задаётся
  числом пикселей `MAX_IMAGE_PIXELS`, что безопаснее отдельного ограничения
  ширины или высоты.
- После inference входной и выходной Pillow objects закрываются. Модель не
  копируется между запросами.

## Concurrency

Inference выполняется через `asyncio.to_thread`, поэтому event loop обслуживает
health-запросы. `asyncio.Semaphore` допускает не более
`MAX_CONCURRENT_INFERENCES` одновременных forward pass. Default 1 защищает full
BiRefNet от RAM/VRAM spikes. Ожидание ограничено
`INFERENCE_QUEUE_TIMEOUT_SECONDS`; timeout возвращает `503 MODEL_ERROR`.

Один uvicorn process означает одну копию модели. Несколько workers создадут
модель в каждом процессе и умножат VRAM/RAM. Для GPU рекомендуется один worker
и горизонтальное масштабирование отдельными replicas.

## Ошибки

```json
{
  "error": {
    "code": "INVALID_IMAGE",
    "message": "Uploaded file is not a valid image."
  }
}
```

Коды: `INVALID_FILE_TYPE` (415), `FILE_TOO_LARGE` (413), `IMAGE_TOO_LARGE`
(413), `INVALID_IMAGE` (422), `MODEL_ERROR` (500/503), `INTERNAL_ERROR` (500).
Stack traces не включаются в response.

## Конфигурация и запуск

Переменные перечислены в `.env.example`:

```powershell
uvicorn bg_removal.api:app --host 0.0.0.0 --port 8000 --workers 1
```

Локальный файл можно загрузить явно через `--env-file .env`. В production
настройки следует передавать средствами оркестратора.

## Проверка этапа

API unit/integration suite с fake backend: **24 passed, 1 skipped**. Skip —
намеренный тест тяжёлого backend, который не должен скачивать weights во время
обычного pytest.

Отдельно выполнен end-to-end smoke с настоящим cached full BiRefNet revision
`e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`, PyTorch CPU и model input 512²:

- FastAPI lifespan/model startup: 15.76 с;
- `/api/health`: HTTP 200, `model_ready=true`, `device=cpu`;
- multipart inference: 8.85 с;
- response: HTTP 200, `Content-Type: image/png`, RGBA 320×240;
- alpha mask: диапазон 0–255.

Это технический single-image smoke, а не latency benchmark. Он подтверждает
полную связку API и модели, но также сохраняет ограничение предыдущего этапа:
full BiRefNet на CPU не подходит для low-latency SLA.

Для финальной production-проверки собраны оба Docker image, backend запущен с
пустым named volume, скачал pinned weights и перешёл в `healthy`. Через nginx
проверены SPA, proxy health и реальный multipart inference при default input
1024²: HTTP 200, `image/png`, RGBA 320×240, alpha 0–255, 43.84 с на доступном
CPU-only Docker Desktop. Оба Compose service после smoke имели status `healthy`.
