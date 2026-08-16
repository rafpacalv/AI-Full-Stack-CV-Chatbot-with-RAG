"""Central configuration, loaded from .env with sane defaults for this machine."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    ollama_host: str = "http://localhost:11434"

    # Both default to gemma2:9b for one measured reason: this machine has an
    # 8 GB RTX 4070 Laptop. gemma4:12b (9.9 GB) only gets ~4.4 GB onto the GPU
    # and runs the rest on CPU -> 7.3 tok/s. gemma2:9b fits and hits 18.3 tok/s,
    # a 3.7x speedup at comparable quality once the prompts were tightened.
    # Set GEN_MODEL=gemma4:12b in .env to trade ~4x wall time for richer prose.
    gen_model: str = "gemma2:9b"
    chat_model: str = "gemma2:9b"
    embed_model: str = "bge-m3"

    top_k: int = 5
    rrf_k: int = 60

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    # Only used to scope the API's CORS policy to the UI's origin.
    ui_port: int = 8501

    photo_enabled: bool = True
    pollinations_url: str = "https://image.pollinations.ai/prompt"

    # --- paths ---
    @property
    def data_dir(self) -> Path:
        return ROOT / "data"

    @property
    def profiles_dir(self) -> Path:
        return self.data_dir / "profiles"

    @property
    def photos_dir(self) -> Path:
        return self.data_dir / "photos"

    @property
    def cvs_dir(self) -> Path:
        return self.data_dir / "cvs"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def api_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"

    def ensure_dirs(self) -> None:
        for d in (
            self.profiles_dir,
            self.photos_dir,
            self.cvs_dir,
            self.index_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
