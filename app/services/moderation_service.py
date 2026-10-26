"""Casos de uso de moderación: clasificar, reportar, cola HITL y decisiones auditadas."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.db import ContentReport, ModerationCase, ModerationDecision, utcnow
from app.services import policy
from app.services.classifiers.base import HateSpeechClassifier

DECISION_TO_VISIBILITY = {"approve": "visible", "keep_hidden": "hidden", "block": "blocked"}


class CaseNotFoundError(Exception):
    pass


class SelfReportError(Exception):
    pass


class ModerationService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    # ------------------------------------------------------------------ helpers
    def thresholds(self) -> dict:
        s = self.settings
        return {
            "class_threshold": s.CLASS_THRESHOLD,
            "hide_threshold": s.HIDE_THRESHOLD,
            "hitl_tau": s.HITL_TAU,
            "report_threshold": s.REPORT_THRESHOLD,
        }

    def find_case(self, content_type: str, content_id: str) -> ModerationCase | None:
        return self.db.scalar(
            select(ModerationCase).where(
                ModerationCase.content_type == content_type,
                ModerationCase.content_id == content_id,
            )
        )

    def _get_or_create(self, content_type: str, content_id: str) -> ModerationCase:
        case = self.find_case(content_type, content_id)
        if case is None:
            case = ModerationCase(
                content_type=content_type,
                content_id=content_id,
                visibility="visible",
                review_status="none",
                reasons=[],
            )
            self.db.add(case)
            self.db.flush()
        return case

    def report_count(self, case: ModerationCase, since=None) -> int:
        q = select(func.count(ContentReport.id)).where(ContentReport.case_id == case.id)
        if since is not None:
            q = q.where(ContentReport.created_at > since)
        return int(self.db.scalar(q) or 0)

    @staticmethod
    def _apply_reasons(case: ModerationCase, reasons: list[str]) -> None:
        case.reasons = sorted(set(reasons))
        if case.visibility == "blocked":
            return
        hides = {policy.REASON_HIDDEN_BY_MODEL, policy.REASON_REPORTED}
        case.visibility = "hidden" if hides & set(case.reasons) else "visible"
        case.review_status = "pending" if case.reasons else "none"

    # ------------------------------------------------------------------ clasificar
    def classify(
        self,
        classifier: HateSpeechClassifier,
        text: str,
        content_type: str,
        content_id: str,
        author_id: str | None = None,
        explain: bool = True,
    ) -> tuple[ModerationCase, policy.PolicyOutcome]:
        s = self.settings
        result = classifier.predict(text, explain=explain)
        outcome = policy.evaluate(result.p_hate, s.CLASS_THRESHOLD, s.HIDE_THRESHOLD, s.HITL_TAU)

        case = self._get_or_create(content_type, content_id)
        text_changed = case.text is not None and case.text != text
        case.text = text
        case.author_id = author_id or case.author_id
        case.p_hate = result.p_hate
        case.label = outcome.label
        case.confidence = outcome.confidence
        case.decision = outcome.decision
        case.explanation = result.explanation
        case.explanation_method = result.explanation_method
        case.model_backend = classifier.name
        case.model_version = s.MODEL_VERSION
        case.thresholds = self.thresholds()
        case.classified_at = utcnow()

        locked = case.visibility == "blocked" or (
            case.review_status == "resolved" and not text_changed
        )
        if not locked:
            keep = [r for r in case.reasons if r not in policy.MODEL_REASONS]
            if text_changed and case.review_status == "resolved":
                # El autor editó un contenido ya revisado: se reevalúa desde cero.
                keep = []
                case.resolved_at = None
                case.resolved_by = None
            self._apply_reasons(case, keep + list(outcome.reasons))
        self.db.commit()
        return case, outcome

    # ------------------------------------------------------------------ reportes
    def report(
        self,
        content_type: str,
        content_id: str,
        reporter_id: str,
        reason: str | None = None,
        text: str | None = None,
        author_id: str | None = None,
    ) -> tuple[ModerationCase, bool, int, bool]:
        """Registra un reporte (idempotente por usuario).

        Devuelve (caso, creado, total_reportes, umbral_alcanzado). Tras una decisión humana
        solo cuentan los reportes posteriores a esa decisión para reabrir el caso.
        """
        case = self._get_or_create(content_type, content_id)
        if text and case.text is None:
            case.text = text
        if author_id and case.author_id is None:
            case.author_id = author_id
        if case.author_id is not None and case.author_id == reporter_id:
            self.db.rollback()
            raise SelfReportError("Un usuario no puede reportar su propio contenido")

        created = False
        exists = self.db.scalar(
            select(ContentReport).where(
                ContentReport.case_id == case.id, ContentReport.reporter_id == reporter_id
            )
        )
        if exists is None:
            try:
                with self.db.begin_nested():
                    self.db.add(
                        ContentReport(case_id=case.id, reporter_id=reporter_id, reason=reason)
                    )
                created = True
            except IntegrityError:  # carrera entre dos peticiones del mismo usuario
                created = False

        total = self.report_count(case)
        relevant = self.report_count(case, since=case.resolved_at) if case.resolved_at else total
        crossed = relevant >= self.settings.REPORT_THRESHOLD
        if crossed and case.visibility != "blocked":
            # Si ya hubo decisión humana, los nuevos reportes reabren el caso desde cero.
            reasons = [] if case.review_status == "resolved" else list(case.reasons)
            self._apply_reasons(case, reasons + [policy.REASON_REPORTED])
        self.db.commit()
        return case, created, total, crossed

    # ------------------------------------------------------------------ cola HITL
    def list_queue(
        self,
        status: str = "pending",
        reason: str | None = None,
        content_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ModerationCase], int]:
        q = select(ModerationCase)
        if status != "all":
            q = q.where(ModerationCase.review_status == status)
        else:
            q = q.where(ModerationCase.review_status != "none")
        if content_type:
            q = q.where(ModerationCase.content_type == content_type)
        rows = list(
            self.db.scalars(q.order_by(ModerationCase.updated_at.desc(), ModerationCase.id.desc()))
        )
        if reason:
            # reasons es JSON; se filtra en Python para ser portable entre SQLite y Postgres.
            rows = [c for c in rows if reason in (c.reasons or [])]
        return rows[offset : offset + limit], len(rows)

    def get_case(self, case_id: int) -> ModerationCase:
        case = self.db.get(ModerationCase, case_id)
        if case is None:
            raise CaseNotFoundError(case_id)
        return case

    def decide(
        self, case_id: int, moderator_id: str, decision: str, notes: str | None = None
    ) -> tuple[ModerationCase, ModerationDecision]:
        case = self.get_case(case_id)
        new_visibility = DECISION_TO_VISIBILITY[decision]
        audit = ModerationDecision(
            case_id=case.id,
            moderator_id=moderator_id,
            decision=decision,
            notes=notes,
            previous_visibility=case.visibility,
            previous_review_status=case.review_status,
            previous_reasons=list(case.reasons or []),
            new_visibility=new_visibility,
            p_hate_at_decision=case.p_hate,
        )
        self.db.add(audit)
        case.visibility = new_visibility
        case.review_status = "resolved"
        case.resolved_at = utcnow()
        case.resolved_by = moderator_id
        self.db.commit()
        self.db.refresh(case)
        return case, audit
