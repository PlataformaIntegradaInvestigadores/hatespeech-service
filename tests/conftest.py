import pytest

from app.core.config import Settings
from app.core.database import Base, build_engine, build_sessionmaker
from app.services.classifiers.mock import MockClassifier

NEUTRAL = "Buenos días a todos, excelente artículo"
DUDOSO = "eres un idiota"  # p≈0.545 con el mock -> confianza < 0.65 -> DERIVAR
ODIO = "Esa gente es escoria y basura, me dan asco"  # p >= 0.80 -> oculto


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
