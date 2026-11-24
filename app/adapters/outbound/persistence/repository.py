"""Implementación SQLAlchemy de `ModerationRepository` sobre una sesión por petición."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.adapters.outbound.persistence import orm  # noqa: F401  (registra el mapeo)
from app.domain.entities import ContentReport, ModerationCase, ModerationDecision


class SqlAlchemyModerationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def find_case(self, content_type: str, content_id: str) -> ModerationCase | None:
        return self.session.scalar(
            select(ModerationCase).where(
                ModerationCase.content_type == content_type,
                ModerationCase.content_id == content_id,
            )
        )

    def get_case(self, case_id: int) -> ModerationCase | None:
        return self.session.get(ModerationCase, case_id)

    def add_case(self, case: ModerationCase) -> None:
        self.session.add(case)
        self.session.flush()

    def count_reports(self, case_id: int, since: datetime | None = None) -> int:
        q = select(func.count(ContentReport.id)).where(ContentReport.case_id == case_id)
        if since is not None:
            q = q.where(ContentReport.created_at > since)
        return int(self.session.scalar(q) or 0)

    def has_report(self, case_id: int, reporter_id: str) -> bool:
        found = self.session.scalar(
            select(ContentReport).where(
                ContentReport.case_id == case_id, ContentReport.reporter_id == reporter_id
            )
        )
        return found is not None

    def add_report(self, report: ContentReport) -> bool:
        try:
            with self.session.begin_nested():
                self.session.add(report)
        except IntegrityError:  # carrera entre dos peticiones del mismo usuario
            return False
        return True

    def list_cases(
        self, review_status: str | None, content_type: str | None
    ) -> list[ModerationCase]:
        q = select(ModerationCase)
        if review_status is None:
            q = q.where(ModerationCase.review_status != "none")
        else:
            q = q.where(ModerationCase.review_status == review_status)
        if content_type:
            q = q.where(ModerationCase.content_type == content_type)
        q = q.order_by(ModerationCase.updated_at.desc(), ModerationCase.id.desc())
        return list(self.session.scalars(q))

    def add_decision(self, decision: ModerationDecision) -> None:
        self.session.add(decision)

    def commit(self) -> None:
        self.session.commit()

    def rollback(self) -> None:
        self.session.rollback()

    def refresh(self, case: ModerationCase) -> None:
        self.session.refresh(case)
