import pytest

from app.core.config import Settings
from app.core.database import Base, build_engine, build_sessionmaker
from app.services.classifiers.mock import MockClassifier

NEUTRAL = "Buenos días a todos, excelente artículo"
DUDOSO = "eres un idiota"  # p≈0.545 con el mock -> confianza < 0.65 -> DERIVAR
ODIO = "Esa gente es escoria y basura, me dan asco"  # p >= 0.80 -> oculto
BANDA = "eres un idiota inútil, lárgate"  # p≈0.75 con el mock -> franja 0.65-0.80 -> revisión


@pytest.fixture
def settings():
    return Settings(DATABASE_URL="sqlite://", MODEL_BACKEND="mock", REPORT_THRESHOLD=5)


@pytest.fixture
def db(settings):
    engine = build_engine(settings.DATABASE_URL)
    Base.metadata.create_all(engine)
    session = build_sessionmaker(engine)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def classifier():
    clf = MockClassifier()
    clf.load()
    return clf


INTERNAL_KEY = "test-internal-key"
JWT_KEY = "test-signing-key"


def make_token(claims: dict | None = None, key: str = JWT_KEY, **extra) -> str:
    import time

    import jwt

    payload = {"token_type": "access", "exp": int(time.time()) + 600, "sub": "100"}
    payload.update(claims or {})
    payload.update(extra)
    return jwt.encode(payload, key, algorithm="HS256")


def bearer(**claims) -> dict:
    return {"Authorization": f"Bearer {make_token(claims)}"}


def internal(user_id: str | None = None, staff: bool = False) -> dict:
    h = {"X-Internal-Api-Key": INTERNAL_KEY}
    if user_id:
        h["X-Acting-User-Id"] = user_id
    if staff:
        h["X-Acting-User-Is-Staff"] = "true"
    return h


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
