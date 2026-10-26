"""Modelo de datos de moderación: casos, reportes y decisiones (auditoría HITL)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class ModerationCase(Base):
    """Estado de moderación de un post o comentario de social-service."""

    __tablename__ = "moderation_cases"
    __table_args__ = (UniqueConstraint("content_type", "content_id", name="uq_case_content"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    content_type: Mapped[str] = mapped_column(String(16), index=True)
    content_id: Mapped[str] = mapped_column(String(64), index=True)
    author_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Última clasificación del modelo
    p_hate: Mapped[float | None] = mapped_column(Float, nullable=True)
    label: Mapped[str | None] = mapped_column(String(16), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision: Mapped[str | None] = mapped_column(String(16), nullable=True)  # AUTO | DERIVAR
    explanation: Mapped[list | None] = mapped_column(JSON, nullable=True)
    explanation_method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_backend: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    thresholds: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    classified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Estado efectivo
    visibility: Mapped[str] = mapped_column(String(16), default="visible")  # visible|hidden|blocked
    review_status: Mapped[str] = mapped_column(String(16), default="none", index=True)
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    reports: Mapped[list[ContentReport]] = relationship(
        back_populates="case", cascade="all, delete-orphan"
    )
    decisions: Mapped[list[ModerationDecision]] = relationship(
        back_populates="case", cascade="all, delete-orphan", order_by="ModerationDecision.id"
    )


class ContentReport(Base):
    """Reporte de un usuario. Un usuario cuenta una sola vez por contenido."""

    __tablename__ = "content_reports"
    __table_args__ = (UniqueConstraint("case_id", "reporter_id", name="uq_report_reporter"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("moderation_cases.id"), index=True)
    reporter_id: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    case: Mapped[ModerationCase] = relationship(back_populates="reports")


class ModerationDecision(Base):
    """Auditoría de cada decisión humana: quién, cuándo, estado anterior y nuevo."""

    __tablename__ = "moderation_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("moderation_cases.id"), index=True)
    moderator_id: Mapped[str] = mapped_column(String(64))
    decision: Mapped[str] = mapped_column(String(16))  # approve | keep_hidden | block
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    previous_visibility: Mapped[str] = mapped_column(String(16))
    previous_review_status: Mapped[str] = mapped_column(String(16))
    previous_reasons: Mapped[list] = mapped_column(JSON, default=list)
    new_visibility: Mapped[str] = mapped_column(String(16))
    p_hate_at_decision: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    case: Mapped[ModerationCase] = relationship(back_populates="decisions")
