#!/usr/bin/env python
"""Benchmark one or more BiRefNet checkpoints on local representative images."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bg_removal import BackgroundRemover, BiRefNetBackend, load_image  # noqa: E402

DEFAULT_MODELS = ["ZhengPeng7/BiRefNet"]
COMPARE_MODELS = ["ZhengPeng7/BiRefNet", "ZhengPeng7/BiRefNet_lite"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class MemorySampler:
    def __init__(self, interval: float = 0.02) -> None:
        self.interval = interval
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: Any = None

    def __enter__(self) -> MemorySampler:
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("Benchmark requires psutil: pip install -e .[benchmark]") from exc
        self._process = psutil.Process(os.getpid())
        self.peak_rss = self._process.memory_info().rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak_rss = max(self.peak_rss, self._process.memory_info().rss)

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        self.peak_rss = max(self.peak_rss, self._process.memory_info().rss)


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, int((len(ordered) - 1) * 0.95 + 0.999999))
    return ordered[index]


def discover_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in EXTENSIONS)


def runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": False, "device": "cpu"}
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
        )
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    return info


def cuda_metrics(reset: bool = False) -> dict[str, int | None]:
    try:
        import torch
    except ImportError:
        return {"allocated_bytes": None, "reserved_bytes": None}
    if not torch.cuda.is_available():
        return {"allocated_bytes": None, "reserved_bytes": None}
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return {
        "allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def synchronize_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_model(
    model_id: str,
    images: list[tuple[Path, Any]],
    *,
    repeats: int,
    device: str,
    input_size: int,
    revision: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"model_id": model_id, "status": "failed"}
    baseline_rss = 0
    try:
        import psutil

        baseline_rss = psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        pass

    try:
        cuda_metrics(reset=True)
        with MemorySampler() as memory:
            start = time.perf_counter()
            backend = BiRefNetBackend(
                model_id,
                revision=revision,
                input_size=input_size,
                device=device,
            )
            synchronize_cuda()
            cold_load = time.perf_counter() - start

            start = time.perf_counter()
            first_output = BackgroundRemover(backend).remove(images[0][1])
            synchronize_cuda()
            cold_first_inference = time.perf_counter() - start
            first_output.close()

            remover = BackgroundRemover(backend)
            latencies: list[float] = []
            for _ in range(repeats):
                for _, image in images:
                    start = time.perf_counter()
                    output = remover.remove(image)
                    synchronize_cuda()
                    latencies.append(time.perf_counter() - start)
                    output.close()

        result.update(
            {
                "status": "ok",
                "device": backend.device,
                "input_size": input_size,
                "image_count": len(images),
                "measured_inferences": len(latencies),
                "repeats": repeats,
                "cold_start_seconds": cold_load,
                "cold_first_inference_seconds": cold_first_inference,
                "mean_latency_seconds": statistics.fmean(latencies),
                "median_latency_seconds": statistics.median(latencies),
                "p95_latency_seconds": percentile95(latencies),
                "peak_ram_bytes": memory.peak_rss,
                "incremental_peak_ram_bytes": max(0, memory.peak_rss - baseline_rss),
                "peak_vram": cuda_metrics(),
                "parameter_count": backend.parameter_count,
                "model_size_bytes": backend.model_size_bytes,
            }
        )
    except Exception as exc:  # Results file must survive dependency/OOM failures.
        result.update({"error_type": type(exc).__name__, "error": str(exc)})
    finally:
        if "backend" in locals():
            del backend
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "benchmark" / "images")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark" / "results.json")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--compare", action="store_true", help="Benchmark full and lite BiRefNet")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--revision", help="Reviewed immutable HF commit for all models")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = discover_images(args.images)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "runtime": runtime_info(),
        },
        "image_directory": str(args.images.resolve()),
        "images": [str(path.relative_to(args.images)) for path in paths],
        "models": [],
    }

    if not paths:
        report.update(
            {
                "status": "skipped",
                "reason": "No JPEG, PNG, or WebP benchmark images were found.",
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report["reason"], file=sys.stderr)
        return 2

    decoded: list[tuple[Path, Any]] = []
    try:
        decoded = [(path, load_image(path)) for path in paths]
        models = args.models or (COMPARE_MODELS if args.compare else DEFAULT_MODELS)
        for model_id in models:
            report["models"].append(
                benchmark_model(
                    model_id,
                    decoded,
                    repeats=args.repeats,
                    device=args.device,
                    input_size=args.input_size,
                    revision=args.revision,
                )
            )
        report["status"] = (
            "ok" if all(item["status"] == "ok" for item in report["models"]) else "failed"
        )
    except Exception as exc:
        report.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        for _, image in decoded:
            image.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
