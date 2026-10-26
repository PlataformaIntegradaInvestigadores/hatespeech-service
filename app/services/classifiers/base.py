from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ClassificationResult:
    """Salida cruda del modelo: probabilidad de la clase "odio" y explicación por palabra."""

    p_hate: float
    explanation: list[dict] = field(default_factory=list)
    explanation_method: str | None = None


class HateSpeechClassifier(ABC):
    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """Carga pesos o recursos. Debe ser idempotente."""

    @property
    @abstractmethod
    def ready(self) -> bool: ...

    @abstractmethod
    def predict(self, text: str, explain: bool = True) -> ClassificationResult: ...
