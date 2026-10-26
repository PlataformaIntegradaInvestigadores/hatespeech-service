from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.services.classifiers.base import HateSpeechClassifier
from app.services.moderation_service import ModerationService


def get_moderation_service(db: Session = Depends(get_db)) -> ModerationService:
    return ModerationService(db, settings)


def get_classifier(request: Request) -> HateSpeechClassifier:
    clf = getattr(request.app.state, "classifier", None)
    if clf is None or not clf.ready:
        error = getattr(request.app.state, "classifier_error", None) or "modelo no cargado"
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"Clasificador no disponible: {error}"
        )
    return clf
