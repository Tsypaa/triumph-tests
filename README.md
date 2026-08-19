# Clearcut — удаление фона

Ориентированный на production веб-сервис для локального удаления фона. React-клиент отправляет изображение в FastAPI, общая модель BiRefNet формирует непрерывную альфа-маску, а API возвращает RGBA PNG исходного размера. Изображения не передаются сторонним API и не сохраняются на постоянной основе.

```text
Браузер -> валидация FastAPI -> обработка EXIF -> BiRefNet
        <- прозрачный RGBA PNG <- обработка alpha <- inference
```

## Модель и вывод исследования

В production используется полная модель `ZhengPeng7/BiRefNet`, зафиксированная на ревизии `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`.

- Research SOTA среди рассмотренных публикаций: PDFNet-L + Depth Anything V2. Результаты, полученные на разных наборах данных и оборудовании, нельзя сравнивать напрямую.
- Practical SOTA: BiRefNet благодаря качественной обработке границ, открытым весам, CPU fallback и простой интеграции с Transformers/PyTorch.
- Выбор для сервиса: полная универсальная BiRefNet. Она медленнее облегчённых вариантов, особенно на CPU, но является более надёжным quality-first решением для волос, шерсти и сложных контуров.

Полное сравнение моделей, ограничения benchmark и анализ лицензий приведены в [docs/research.md](docs/research.md). Репозиторий и model card выбранной модели указывают лицензию MIT; коммерческое использование разрешено при сохранении уведомления о лицензии. Загрузка удалённого кода модели остаётся supply-chain риском, поэтому приложение использует проверенную неизменяемую ревизию.

## Архитектура

Код разделён на независимые слои:

- `BiRefNetBackend` загружает модель один раз и выполняет inference на CUDA при её наличии или на CPU;
- `BackgroundRemover` выполняет обработку EXIF, восстановление alpha в исходном разрешении, очистку границ и формирование RGBA;
- FastAPI отвечает за валидацию, структурированные ошибки, lifecycle и ограничение параллельных inference;
- React/Vite реализует загрузку, preview, отмену, сравнение до/после и скачивание PNG;
- nginx раздаёт production frontend и проксирует `/api` в FastAPI.

Подробный жизненный цикл запроса и контракт ошибок описаны в [docs/architecture.md](docs/architecture.md).

## Требования

- Python 3.11 или новее;
- Node.js 20.19+ или 22.12+ и npm для локальной разработки frontend;
- Docker с Compose для воспроизводимого production-окружения;
- достаточно места для зависимостей PyTorch и примерно 0,9 ГБ FP32-параметров модели.

CUDA необязательна. При запуске без Docker сначала установите сборку PyTorch, совместимую с драйвером компьютера, затем установите Python-проект.

## Локальная разработка

Установите все зависимости:

```bash
python -m pip install -e ".[birefnet,api,benchmark,test,dev]"
cd frontend && npm ci && cd ..
```

Если стандартные настройки требуется изменить, скопируйте `.env.example` в `.env`. Запустите оба dev-сервера через GNU Make:

```bash
make dev
```

Или запустите их отдельно:

```bash
python -m uvicorn bg_removal.api:app --host 0.0.0.0 --port 8000 --reload
cd frontend
npm run dev
```

Vite будет доступен по адресу `http://localhost:5173` и станет проксировать `/api` на `http://localhost:8000`.

## Docker

Соберите и запустите production-окружение с поддержкой CPU:

```bash
docker compose up --build
```

Frontend будет доступен по адресу `http://localhost:8080`, а API также напрямую по адресу `http://localhost:8000`. При первом запуске backend скачивает зафиксированные веса — это может занять несколько минут.

Для компьютера с NVIDIA Container Toolkit выберите официальный индекс PyTorch с CUDA, совместимый с вашим драйвером, и примените GPU-конфигурацию:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Не копируйте индекс CUDA из примера без проверки совместимости с установленным драйвером.

### Веса модели и кеш

- источник: `https://huggingface.co/ZhengPeng7/BiRefNet`;
- зафиксированная ревизия: `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`;
- объём параметров загруженной модели: 880 705 992 байта, или 220 176 498 параметров;
- локальный кеш: стандартный кеш Hugging Face, путь можно изменить через `HF_HOME`;
- кеш контейнера: `/models/huggingface`, сохраняемый в Compose volume `model-cache`.

Чтобы заранее загрузить веса в Docker volume до начала обработки запросов:

```bash
docker compose build backend
docker compose run --rm backend python -c "from bg_removal import BiRefNetBackend; BiRefNetBackend()"
```

## API

### Состояние и информация о модели

```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/model-info
```

### Удаление фона

```bash
curl -F "file=@photo.webp;type=image/webp" \
  http://localhost:8000/api/remove-background \
  --output result.png
```

Поддерживаются JPEG, PNG и WebP. Ответ — RGBA-изображение `image/png` с display dimensions после применения EXIF orientation. Ошибки API имеют следующую структуру:

```json
{
  "error": {
    "code": "INVALID_IMAGE",
    "message": "Uploaded file is not a valid image."
  }
}
```

## Конфигурация

Все runtime-настройки задаются переменными окружения; значения по умолчанию приведены в [.env.example](.env.example).

| Переменная | Назначение |
| --- | --- |
| `MODEL` / `MODEL_REVISION` | идентификатор модели Hugging Face и неизменяемая ревизия |
| `DEVICE` | `auto`, `cpu` или `cuda` |
| `INFERENCE_RESOLUTION` | размер квадратного входа модели; по умолчанию `1024` |
| `MAX_UPLOAD_SIZE_BYTES` | ограничение размера загружаемого файла |
| `MAX_IMAGE_PIXELS` | ограничение количества пикселей декодированного изображения |
| `MAX_CONCURRENT_INFERENCES` | число одновременных forward pass; по умолчанию `1` |
| `INFERENCE_QUEUE_TIMEOUT_SECONDS` | максимальное время ожидания в очереди |
| `CORS_ORIGINS` | разделённый запятыми список разрешённых origin |
| `LOG_LEVEL` | уровень логирования backend |

Используйте один worker uvicorn на одну реплику модели. Несколько workers создадут отдельную копию модели в RAM/VRAM для каждого процесса.

## Тесты и проверка качества кода

```bash
make test
make lint
```

Эквивалентные команды:

```bash
python -m pytest
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
cd frontend && npm test && npm run lint && npm run build
```

Большинство тестов API используют fake inference backend, поэтому стандартный набор тестов не скачивает и не запускает тяжёлую модель.

## Benchmark

Добавьте JPEG, PNG или WebP с подходящей лицензией согласно инструкции [benchmark/images/README.md](benchmark/images/README.md), затем выполните:

```bash
make benchmark
# Необязательное сравнение моделей:
python scripts/benchmark.py --compare --device auto --repeats 3
```

Результаты сохраняются в [benchmark/results.json](benchmark/results.json). В текущем окружении не было репрезентативного набора фотографий, поэтому формальный benchmark отмечен как `skipped`. Сохранённый технический тест на синтетических изображениях не является сравнительным benchmark качества или latency разных моделей.

На доступной системе только с CPU полная зафиксированная модель при размере входа 512 px загрузилась из кеша за 40,88 с; отдельные inference заняли 8,99 и 6,90 с. Отдельный локальный smoke test API занял 15,76 с на запуск и 8,85 с на запрос с изображением 320×240. Финальный smoke test Docker/nginx со стандартным production-входом 1024 px занял 43,84 с при тех же выходных размерах. Результаты зависят от устройства и входных данных.

## Сокращённые команды

```bash
make install
make dev
make test
make lint
make benchmark
make docker-up
make docker-down
```

## Ограничения

- Полная BiRefNet требовательна к вычислительным ресурсам: CPU fallback работает, но не подходит для low-latency SLA.
- Первый запуск требует доступа к Hugging Face, если веса не были загружены заранее.
- Удаление salient object может ошибаться в неоднозначных сценах с несколькими одинаково важными объектами, прозрачными предметами или большим количеством отражений.
- Результат представляет собой непрерывную segmentation alpha, а не portrait matting с trimap; на очень тонких полупрозрачных прядях возможны артефакты.
- В комплект benchmark намеренно не входят сторонние фотографии. Перед использованием результатов для capacity planning добавьте лицензированный репрезентативный набор.
- Запуск GPU-контейнера требует совместимых драйвера NVIDIA, Container Toolkit и CUDA-сборки PyTorch.
