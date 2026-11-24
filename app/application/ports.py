"""Puertos: lo que los casos de uso necesitan del exterior.

La aplicación solo conoce estas interfaces; los adaptadores de
`app/adapters/outbound` las implementan (Qwen/mock, SQLAlchemy) y
`app/bootstrap.py` decide cuáles se usan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from app.domain.entities import ContentReport, ModerationCase, ModerationDecision


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

    @property
    def device(self) -> str:
        """Dispositivo donde corre el modelo; se expone en /health."""
        return "cpu"

    @abstractmethod
    def predict(self, text: str, explain: bool = True) -> ClassificationResult: ...


class ModerationConfig(Protocol):
    """Umbrales y versión del modelo. `app.core.config.Settings` la cumple."""

    CLASS_THRESHOLD: float
    HIDE_THRESHOLD: float
    HITL_TAU: float
    REPORT_THRESHOLD: int
    MODEL_VERSION: str

    @property
    def review_band_min(self) -> float | None: ...


class ModerationRepository(Protocol):
    """Persistencia de casos, reportes y decisiones (una unidad de trabajo por petición)."""

    def find_case(self, content_type: str, content_id: str) -> ModerationCase | None: ...

    def get_case(self, case_id: int) -> ModerationCase | None: ...

    def add_case(self, case: ModerationCase) -> None:
        """Agrega el caso y le asigna `id`."""

    def count_reports(self, case_id: int, since: datetime | None = None) -> int:
        """Reportes del caso; con `since`, solo los posteriores a esa fecha."""

    def has_report(self, case_id: int, reporter_id: str) -> bool: ...

    def add_report(self, report: ContentReport) -> bool:
        """Guarda el reporte; False si el mismo usuario ya lo había guardado (carrera)."""

    def list_cases(
        self, review_status: str | None, content_type: str | None
    ) -> list[ModerationCase]:
        """Casos del más reciente al más antiguo. `review_status=None` = todos los que
        alguna vez entraron a la cola (review_status distinto de "none")."""

    def add_decision(self, decision: ModerationDecision) -> None: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def refresh(self, case: ModerationCase) -> None: ...
