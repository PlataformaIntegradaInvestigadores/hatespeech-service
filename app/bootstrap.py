"""Raíz de composición: único lugar que conoce las clases concretas.

Elige el clasificador (Qwen o mock) y la persistencia (SQLAlchemy), y los inyecta en
los casos de uso. Todo lo demás depende solo de los puertos.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.adapters.outbound.classifiers.mock import MockClassifier
from app.adapters.outbound.persistence.repository import SqlAlchemyModerationRepository
from app.application.moderation_service import ModerationService
from app.application.ports import HateSpeechClassifier
from app.core.config import Settings


def build_classifier(settings: Settings) -> HateSpeechClassifier:
    if settings.MODEL_BACKEND == "qwen":
        # Import diferido: torch/transformers solo existen en la imagen con GPU.
        from app.adapters.outbound.classifiers.qwen import QwenQLoRAClassifier

        return QwenQLoRAClassifier(
            base_model=settings.BASE_MODEL,
            adapter_path=settings.ADAPTER_PATH,
            max_length=settings.MAX_LENGTH,
            top_k=settings.EXPLANATION_TOP_K,
        )
    return MockClassifier(top_k=settings.EXPLANATION_TOP_K)


def build_moderation_service(session: Session, settings: Settings) -> ModerationService:
    return ModerationService(SqlAlchemyModerationRepository(session), settings)
