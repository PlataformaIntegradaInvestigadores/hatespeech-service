"""Tablas de moderación y su mapeo a las entidades del dominio.

Mapeo imperativo: las entidades de `app.domain.entities` siguen siendo clases puras y
SQLAlchemy las instrumenta aquí. `start_mappers()` se llama al importar este módulo.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import registry, relationship

from app.domain.entities import ContentReport, ModerationCase, ModerationDecision, utcnow

metadata = MetaData()
mapper_registry = registry(metadata=metadata)

moderation_cases = Table(
    "moderation_cases",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("content_type", String(16), nullable=False, index=True),
    Column("content_id", String(64), nullable=False, index=True),
    Column("author_id", String(64), nullable=True),
    Column("text", Text, nullable=True),
    Column("p_hate", Float, nullable=True),
    Column("label", String(16), nullable=True),
    Column("confidence", Float, nullable=True),
    Column("decision", String(16), nullable=True),
    Column("explanation", JSON, nullable=True),
    Column("explanation_method", String(64), nullable=True),
    Column("model_backend", String(32), nullable=True),
    Column("model_version", String(128), nullable=True),
    Column("thresholds", JSON, nullable=True),
    Column("classified_at", DateTime(timezone=True), nullable=True),
    Column("visibility", String(16), nullable=False, default="visible"),
    Column("review_status", String(16), nullable=False, default="none", index=True),
    Column("reasons", JSON, nullable=False, default=list),
    Column("resolved_at", DateTime(timezone=True), nullable=True),
    Column("resolved_by", String(64), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, default=utcnow),
    Column("updated_at", DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow),
    UniqueConstraint("content_type", "content_id", name="uq_case_content"),
)

content_reports = Table(
    "content_reports",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("case_id", Integer, ForeignKey("moderation_cases.id"), nullable=False, index=True),
    Column("reporter_id", String(64), nullable=False),
    Column("reason", String(255), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, default=utcnow),
    UniqueConstraint("case_id", "reporter_id", name="uq_report_reporter"),
)

moderation_decisions = Table(
    "moderation_decisions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("case_id", Integer, ForeignKey("moderation_cases.id"), nullable=False, index=True),
    Column("moderator_id", String(64), nullable=False),
    Column("decision", String(16), nullable=False),
    Column("notes", Text, nullable=True),
    Column("previous_visibility", String(16), nullable=False),
    Column("previous_review_status", String(16), nullable=False),
    Column("previous_reasons", JSON, nullable=False, default=list),
    Column("new_visibility", String(16), nullable=False),
    Column("p_hate_at_decision", Float, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, default=utcnow),
)


def start_mappers() -> None:
    if mapper_registry.mappers:
        return
    mapper_registry.map_imperatively(
        ModerationCase,
        moderation_cases,
        properties={
            "reports": relationship(
                ContentReport, back_populates="case", cascade="all, delete-orphan"
            ),
            "decisions": relationship(
                ModerationDecision,
                back_populates="case",
                cascade="all, delete-orphan",
                order_by=moderation_decisions.c.id,
            ),
        },
    )
    mapper_registry.map_imperatively(
        ContentReport,
        content_reports,
        properties={"case": relationship(ModerationCase, back_populates="reports")},
    )
    mapper_registry.map_imperatively(
        ModerationDecision,
        moderation_decisions,
        properties={"case": relationship(ModerationCase, back_populates="decisions")},
    )


start_mappers()
