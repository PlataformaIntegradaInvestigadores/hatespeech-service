from fastapi import APIRouter, Depends

from app.api.deps import get_classifier, get_moderation_service
from app.api.serializers import explanation_of
from app.core.config import settings
from app.core.security import Principal, require_internal
from app.models.schemas import ClassifyRequest, ClassifyResponse
from app.services.classifiers.base import HateSpeechClassifier
from app.services.moderation_service import ModerationService

router = APIRouter()


@router.post(
    "/classify", response_model=ClassifyResponse, summary="Clasificar un post o comentario"
)
def classify(
    body: ClassifyRequest,
    _: Principal = Depends(require_internal),
    classifier: HateSpeechClassifier = Depends(get_classifier),
    svc: ModerationService = Depends(get_moderation_service),
):
    """Clasifica el texto, aplica la política (censura suave + HITL) y registra el caso.

    `action` es lo que recomienda el modelo; `visibility` es el estado efectivo, que
    respeta reportes y decisiones humanas previas sobre el mismo contenido.
    """
    case, outcome = svc.classify(
        classifier,
        text=body.text,
        content_type=body.content_type,
        content_id=body.content_id,
        author_id=body.author_id,
        explain=body.explain and settings.EXPLANATION_ENABLED,
    )
    return ClassifyResponse(
        case_id=case.id,
        content_type=case.content_type,
        content_id=case.content_id,
        label=outcome.label,
        p_hate=case.p_hate,
        confidence=outcome.confidence,
        decision=outcome.decision,
        action=outcome.action,
        visibility=case.visibility,
        review_status=case.review_status,
        reasons=list(case.reasons),
        explanation=explanation_of(case),
        model={"backend": case.model_backend, "version": case.model_version},
        thresholds=svc.thresholds(),
    )
