from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.api.deps import get_moderation_service
from app.core.config import settings
from app.core.security import Principal, get_principal
from app.models.schemas import ReportRequest, ReportResponse
from app.services.moderation_service import ModerationService, SelfReportError

router = APIRouter()


@router.post(
    "/reports",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Reportar un contenido (idempotente por usuario)",
)
def create_report(
    body: ReportRequest,
    response: Response,
    principal: Principal = Depends(get_principal),
    svc: ModerationService = Depends(get_moderation_service),
):
    """Con JWT, el denunciante es el usuario del token. Desde social-service (API key
    interna) se toma `reporter_id` del cuerpo o, en su defecto, `X-Acting-User-Id`."""
    if principal.via == "jwt":
        if body.reporter_id and body.reporter_id != principal.user_id:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN, "No puede reportar en nombre de otro usuario"
            )
        reporter_id = principal.user_id
    else:
        reporter_id = body.reporter_id or principal.user_id
    if not reporter_id:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "reporter_id es obligatorio")
    try:
        case, created, total, crossed = svc.report(
            body.content_type,
            body.content_id,
            reporter_id,
            reason=body.reason,
            text=body.text,
            author_id=body.author_id,
        )
    except SelfReportError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if not created:
        response.status_code = status.HTTP_200_OK
    return ReportResponse(
        case_id=case.id,
        content_type=case.content_type,
        content_id=case.content_id,
        created=created,
        report_count=total,
        report_threshold=settings.REPORT_THRESHOLD,
        threshold_reached=crossed,
        visibility=case.visibility,
        review_status=case.review_status,
    )
