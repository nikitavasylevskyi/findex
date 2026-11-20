"""Configuration comes only from the environment (INDEX_PATH, PORT, LOG_LEVEL, ...)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(case_sensitive=False, extra="ignore")

    index_path: Path = Path("index.bin")
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)  # Render and most platforms set PORT
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    page_size: int = Field(10, ge=1, le=50)
    workers: int = Field(1, ge=1, le=64)  # uvicorn worker processes for `findex serve`
