"""Stable public API error envelope."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ApiError(Exception):
    status_code: int
    code: str
    message: str

    def body(self) -> dict[str, dict[str, str]]:
        return {"error": {"code": self.code, "message": self.message}}
