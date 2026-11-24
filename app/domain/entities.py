"""Entidades de moderación: casos, reportes y decisiones (auditoría HITL).

Son clases de Python puras. El adaptador de persistencia las mapea a tablas
(`app/adapters/outbound/persistence/orm.py`) sin que el dominio conozca SQLAlchemy.
Las relaciones `ModerationCase.reports`, `ModerationCase.decisions` y `.case` las
agrega ese mapeo; no son campos del constructor para que se carguen desde la base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(eq=False)
class ModerationCase:
    """Estado de moderación de un post o comentario de social-service."""

    content_type: str
    content_id: str
    id: int | None = None
    author_id: str | None = None
    text: str | None = None

    # Última clasificación del modelo
    p_hate: float | None = None
    label: str | None = None
    confidence: float | None = None
    decision: str | None = None  # AUTO | DERIVAR
    explanation: list | None = None
    explanation_method: str | None = None
    model_backend: str | None = None
    model_version: str | None = None
    thresholds: dict | None = None
    classified_at: datetime | None = None

    # Estado efectivo
    visibility: str = "visible"  # visible | hidden | blocked
    review_status: str = "none"  # none | pending | resolved
    reasons: list = field(default_factory=list)
    resolved_at: datetime | None = None
    resolved_by: str | None = None

    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)


@dataclass(eq=False)
class ContentReport:
    """Reporte de un usuario. Un usuario cuenta una sola vez por contenido."""

    case_id: int
    reporter_id: str
    reason: str | None = None
    id: int | None = None
    created_at: datetime = field(default_factory=utcnow)


@dataclass(eq=False)
class ModerationDecision:
    """Auditoría de cada decisión humana: quién, cuándo, estado anterior y nuevo."""

    case_id: int
    moderator_id: str
    decision: str  # approve | keep_hidden | block
    previous_visibility: str
    previous_review_status: str
    new_visibility: str
    notes: str | None = None
    previous_reasons: list = field(default_factory=list)
    p_hate_at_decision: float | None = None
    id: int | None = None
    created_at: datetime = field(default_factory=utcnow)
