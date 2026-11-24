"""Conversión de filas ORM a los esquemas públicos."""

from app.domain.entities import ModerationCase
from app.models.schemas import DecisionOut, ReviewCase, ReviewCaseDetail


def explanation_of(case: ModerationCase) -> dict:
    """La saliencia se guarda con las claves de los scripts XAI de la tesis
    (palabra/ini/fin/saliencia); la API la expone en inglés."""
    tokens = [
        {
            "token": item["palabra"],
            "saliency": item["saliencia"],
            "start": item["ini"],
            "end": item["fin"],
        }
        for item in case.explanation or []
    ]
    return {"method": case.explanation_method, "top_tokens": tokens}


def _base(case: ModerationCase) -> dict:
    return {
        "id": case.id,
        "content_type": case.content_type,
        "content_id": case.content_id,
        "author_id": case.author_id,
        "text": case.text,
        "label": case.label,
        "p_hate": case.p_hate,
        "confidence": case.confidence,
        "decision": case.decision,
        "explanation": explanation_of(case),
        "visibility": case.visibility,
        "review_status": case.review_status,
        "reasons": list(case.reasons or []),
        "report_count": len(case.reports),
        "model_version": case.model_version,
        "created_at": case.created_at,
        "updated_at": case.updated_at,
        "resolved_at": case.resolved_at,
        "resolved_by": case.resolved_by,
    }


def review_case(case: ModerationCase) -> ReviewCase:
    return ReviewCase(**_base(case))


def review_case_detail(case: ModerationCase) -> ReviewCaseDetail:
    return ReviewCaseDetail(
        **_base(case),
        decisions=[DecisionOut.model_validate(d) for d in case.decisions],
        report_reasons=[r.reason for r in case.reports if r.reason],
    )
