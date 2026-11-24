from collections.abc import Iterator

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.application.moderation_service import ModerationService
from app.application.ports import HateSpeechClassifier
from app.bootstrap import build_moderation_service
from app.core.config import settings


def get_db(request: Request) -> Iterator[Session]:
    session = request.app.state.sessionmaker()
    try:
        yield session
    finally:
        session.close()


def get_moderation_service(db: Session = Depends(get_db)) -> ModerationService:
    return build_moderation_service(db, settings)


def get_classifier(request: Request) -> HateSpeechClassifier:
    clf = getattr(request.app.state, "classifier", None)
    if clf is None or not clf.ready:
        error = getattr(request.app.state, "classifier_error", None) or "modelo no cargado"
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"Clasificador no disponible: {error}"
        )
    return clf
