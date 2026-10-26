import logging
import socket
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from app.core.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


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


@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "Centinela - servicio de moderación de discurso de odio"}


def _local_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "unknown"


@app.get("/health", include_in_schema=False)
def health():
    payload = {
        "server_name": "hatespeech-service",
        "ip_address": _local_ip(),
        "global_status": "Online",
        "groups": [],
    }
    return JSONResponse(status_code=200, content=payload)
