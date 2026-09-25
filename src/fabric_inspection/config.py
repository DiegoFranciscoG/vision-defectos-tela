"""Runtime configuration read from environment variables (never from committed files)."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

MIN_API_KEY_LENGTH = 32


class QualitySettings(BaseSettings):
    """Parameters of the quality decision. Defaults are documented assumptions (S1, S3)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mm_per_pixel: float = Field(
        default=0.1, gt=0, description="Camera scale at native resolution (S1)"
    )
    max_points_per_100_sq_yd: float = Field(
        default=40.0, gt=0, description="Roll acceptance limit (S3)"
    )
    warning_ratio: float = Field(default=0.8, gt=0, lt=1)


class Settings(QualitySettings):
    """Settings shared by the pipeline, the API and the seed commands."""

    database_url: str = "sqlite:///data/app.db"
    model_dir: Path = Path("models")
    model_cache_dir: Path = Path(".cache/models")
    model_registry: Path = Path("models/registry.json")
    model_base_url: str | None = Field(
        default=None, description="Overrides the release URL used to download ONNX files"
    )
    log_level: str = "INFO"


class ApiSettings(Settings):
    """API settings. API_KEYS has no default: the service refuses to start without it."""

    api_keys: Annotated[list[SecretStr], NoDecode]
    cors_allowed_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    enable_docs: bool = True
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, gt=0)
    max_image_side: int = Field(default=4096, gt=0)
    rate_limit_per_minute: int = Field(default=30, gt=0)
    max_concurrent_inferences: int = Field(default=2, gt=0)
    inference_queue_timeout_s: float = Field(default=10.0, gt=0)

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_keys(cls, value: object) -> object:
        if isinstance(value, str):
            return [key.strip() for key in value.split(",") if key.strip()]
        return value

    @field_validator("api_keys")
    @classmethod
    def _require_strong_keys(cls, keys: list[SecretStr]) -> list[SecretStr]:
        if not keys:
            raise ValueError("API_KEYS must contain at least one key")
        for key in keys:
            if len(key.get_secret_value()) < MIN_API_KEY_LENGTH:
                raise ValueError(f"Each API key must have at least {MIN_API_KEY_LENGTH} characters")
        return keys

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            origins = [origin.strip() for origin in value.split(",") if origin.strip()]
            if "*" in origins:
                raise ValueError("CORS_ALLOWED_ORIGINS must list explicit origins, not '*'")
            return origins
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
