from __future__ import annotations

from app.core.config import Settings
from app.services.classifiers.base import HateSpeechClassifier
from app.services.classifiers.mock import MockClassifier


def build_classifier(settings: Settings) -> HateSpeechClassifier:
    if settings.MODEL_BACKEND == "qwen":
        # Import diferido: torch/transformers solo existen en la imagen con GPU.
        from app.services.classifiers.qwen import QwenQLoRAClassifier

        return QwenQLoRAClassifier(
            base_model=settings.BASE_MODEL,
            adapter_path=settings.ADAPTER_PATH,
            max_length=settings.MAX_LENGTH,
            top_k=settings.EXPLANATION_TOP_K,
        )
    return MockClassifier(top_k=settings.EXPLANATION_TOP_K)
