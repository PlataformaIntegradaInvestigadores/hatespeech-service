from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración del servicio, leída desde variables de entorno o `.env`.

    Los umbrales por defecto son los congelados en validación durante la TIC
    (nunca se reeligieron mirando el test). Ver README, sección "Umbrales".
    """

    PROJECT_NAME: str = "Centinela - Moderación de discurso de odio"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    ALLOWED_HOSTS: list[str] = ["*"]
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:8082", "http://127.0.0.1:8082"]

    # --- Modelo ---------------------------------------------------------------
    MODEL_BACKEND: Literal["qwen", "mock"] = "mock"
    BASE_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"
    ADAPTER_PATH: str = "/models/qlora_qwen25_3b_es_ecu"
    MODEL_VERSION: str = "qlora-qwen25-3b-es-ecu-config3"
    MAX_LENGTH: int = 128
    EXPLANATION_ENABLED: bool = True
    EXPLANATION_TOP_K: int = Field(default=8, ge=1, le=50)

    # --- Umbrales (congelados en validación) ----------------------------------
    # p(odio) >= 0.5 -> etiqueta "odio" (umbral de clase usado en la evaluación).
    CLASS_THRESHOLD: float = Field(default=0.50, ge=0.0, le=1.0)
    # p(odio) >= HIDE_THRESHOLD -> censura suave (oculto con advertencia, al final del hilo).
    HIDE_THRESHOLD: float = Field(default=0.80, ge=0.0, le=1.0)
    # confianza = max(p, 1-p) < HITL_TAU -> DERIVAR a revisión humana.
    HITL_TAU: float = Field(default=0.65, ge=0.5, le=1.0)
    # Reportes de usuarios distintos que ocultan el contenido y lo envían a revisión.
    REPORT_THRESHOLD: int = Field(default=5, ge=1)

    # --- Persistencia ---------------------------------------------------------
    DATABASE_URL: str = "sqlite:///./hatespeech.db"

    # --- Seguridad ------------------------------------------------------------
    JWT_SIGNING_KEY: str = "change-me"
    JWT_ALGORITHM: str = "HS256"
    INTERNAL_API_KEY: str = "change-me-internal"
    # "claim": el token trae is_staff; "allowlist": STAFF_USER_IDS; "claim_or_allowlist": cualquiera.
    STAFF_CHECK_MODE: Literal["claim", "allowlist", "claim_or_allowlist"] = "claim_or_allowlist"
    STAFF_USER_IDS: list[str] = []

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
