from fastapi import APIRouter, Depends

from app.api.deps import get_moderation_service
from app.core.security import Principal, get_principal
from app.models.schemas import ContentType, ModerationStatus
from app.services.moderation_service import ModerationService

router = APIRouter()


@router.get(
    "/moderation-status/{content_type}/{content_id}",
    response_model=ModerationStatus,
    summary="Estado de moderación de un contenido",
)
def moderation_status(
    content_type: ContentType,
    content_id: str,
    _: Principal = Depends(get_principal),
    svc: ModerationService = Depends(get_moderation_service),
):
    """Lo consulta social-service al armar un hilo: si `show_warning`, se muestra la
    advertencia y `sort_last` indica moverlo al final. Un contenido nunca visto devuelve
    `classified=false` y visible."""
    case = svc.find_case(content_type, content_id)
    if case is None:
        return ModerationStatus(
            content_type=content_type,
            content_id=content_id,
            classified=False,
            visibility="visible",
            show_warning=False,
            sort_last=False,
            review_status="none",
            reasons=[],
            report_count=0,
        )
    hidden = case.visibility == "hidden"
    return ModerationStatus(
        content_type=content_type,
        content_id=content_id,
        classified=case.p_hate is not None,
        visibility=case.visibility,
        show_warning=hidden,
        sort_last=hidden,
        review_status=case.review_status,
        reasons=list(case.reasons or []),
        report_count=svc.report_count(case),
        label=case.label,
        p_hate=case.p_hate,
    )
