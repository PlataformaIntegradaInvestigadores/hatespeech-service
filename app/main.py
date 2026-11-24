import logging
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from app.adapters.outbound.persistence.database import (
    build_engine,
    build_sessionmaker,
    create_schema,
    ping,
)
from app.api.v1 import api_router
from app.bootstrap import build_classifier
from app.core.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = build_engine(settings.DATABASE_URL)
    create_schema(engine)
    app.state.engine = engine
    app.state.sessionmaker = build_sessionmaker(engine)

    app.state.classifier = None
    app.state.classifier_error = None
    try:
        classifier = build_classifier(settings)
        classifier.load()
        app.state.classifier = classifier
        logger.info("Clasificador '%s' listo (%s)", classifier.name, settings.MODEL_VERSION)
    except Exception as exc:  # el servicio arranca y /health reporta el fallo
        app.state.classifier_error = str(exc)
        logger.error("No se pudo cargar el clasificador: %s", exc)

    yield
    engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS]
    + ["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def custom_openapi():
    """Recorta el prefijo interno /api/v1 y declara la ruta pública detrás del
    gateway (/api/hatespeech/), igual que el resto de servicios FastAPI."""
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    prefix = settings.API_V1_STR
    schema["paths"] = {
        (path[len(prefix) :] if path.startswith(prefix) else path): item
        for path, item in schema.get("paths", {}).items()
    }
    schema["servers"] = [{"url": "/api/hatespeech", "description": "Gateway"}]
    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "Centinela - servicio de moderación de discurso de odio"}


def _local_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "unknown"


def _db_ok() -> bool:
    engine = getattr(app.state, "engine", None)
    if engine is None:
        return False
    try:
        ping(engine)
        return True
    except Exception as exc:
        logger.error("Health check: base de datos no disponible: %s", exc)
        return False


@app.get("/health", include_in_schema=False)
def health():
    classifier = getattr(app.state, "classifier", None)
    model_ok = classifier is not None and classifier.ready
    db_ok = _db_ok()
    ok = model_ok and db_ok
    if not model_ok:
        logger.error(
            "Health check: clasificador no disponible: %s",
            getattr(app.state, "classifier_error", None),
        )

    payload = {
        "server_name": "hatespeech-service",
        "ip_address": _local_ip(),
        "global_status": "Online" if ok else "Offline",
        "groups": [
            {
                "group_name": "Moderación de discurso de odio",
                "group_status": "Operativo" if ok else "Caído",
                "services": [
                    {
                        "name": (
                            "modelo-qlora" if settings.MODEL_BACKEND == "qwen" else "modelo-mock"
                        ),
                        "status": "ok" if model_ok else "error",
                        "device": classifier.device if model_ok else None,
                    },
                    {"name": "base-de-datos", "status": "ok" if db_ok else "error"},
                ],
            }
        ],
    }
    if ok:
        return payload
    return JSONResponse(status_code=503, content=payload)
