from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_moderation_service
from app.api.serializers import review_case, review_case_detail
from app.core.security import Principal, require_staff
from app.models.schemas import (
    ContentType,
    DecisionOut,
    DecisionRequest,
    DecisionResponse,
    Reason,
    ReviewCaseDetail,
    ReviewQueuePage,
)
from app.services.moderation_service import CaseNotFoundError, ModerationService

router = APIRouter(prefix="/review-queue")


def _get_or_404(svc: ModerationService, case_id: int):
    try:
        return svc.get_case(case_id)
    except CaseNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Caso no encontrado") from exc


@router.get("", response_model=ReviewQueuePage, summary="Cola de revisión humana (HITL)")
def list_review_queue(
    status_: Literal["pending", "resolved", "all"] = Query("pending", alias="status"),
    reason: Reason | None = None,
    content_type: ContentType | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _: Principal = Depends(require_staff),
    svc: ModerationService = Depends(get_moderation_service),
):
    rows, total = svc.list_queue(status_, reason, content_type, limit, offset)
    return ReviewQueuePage(
        total=total, limit=limit, offset=offset, items=[review_case(c) for c in rows]
    )


@router.get("/{case_id}", response_model=ReviewCaseDetail, summary="Detalle de un caso")
def get_review_case(
    case_id: int,
    _: Principal = Depends(require_staff),
    svc: ModerationService = Depends(get_moderation_service),
):
    return review_case_detail(_get_or_404(svc, case_id))


@router.post(
    "/{case_id}/decision", response_model=DecisionResponse, summary="Decisión del moderador"
)
def decide(
    case_id: int,
    body: DecisionRequest,
    principal: Principal = Depends(require_staff),
    svc: ModerationService = Depends(get_moderation_service),
):
    """`approve` → visible, `keep_hidden` → oculto con advertencia, `block` → bloqueado.
    Cada decisión queda en la bitácora con el estado previo."""
    _get_or_404(svc, case_id)
    case, audit = svc.decide(case_id, principal.user_id, body.decision, body.notes)
    return DecisionResponse(case=review_case(case), decision=DecisionOut.model_validate(audit))
