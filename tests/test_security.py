import pytest

from app.core import security
from app.core.config import settings
from tests.conftest import bearer, internal, make_token

CLASSIFY = {"text": "hola", "content_type": "post", "content_id": 1}


def test_sin_credenciales_401(client):
    r = client.post("/api/v1/classify", json=CLASSIFY)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_api_key_incorrecta_401(client):
    r = client.post("/api/v1/classify", json=CLASSIFY, headers={"X-Internal-Api-Key": "x"})
    assert r.status_code == 401


def test_token_firma_invalida(client):
    h = {"Authorization": f"Bearer {make_token(key='otra-clave-de-prueba-larga')}"}
    assert client.get("/api/v1/moderation-status/post/1", headers=h).status_code == 401


def test_token_expirado(client):
    h = {"Authorization": f"Bearer {make_token(exp=1)}"}
    assert client.get("/api/v1/moderation-status/post/1", headers=h).status_code == 401


def test_refresh_token_rechazado(client):
    h = bearer(token_type="refresh")
    assert client.get("/api/v1/moderation-status/post/1", headers=h).status_code == 401


def test_token_sin_usuario(client):
    h = {"Authorization": f"Bearer {make_token(sub=None)}"}
    assert client.get("/api/v1/moderation-status/post/1", headers=h).status_code == 401


def test_usuario_normal_no_clasifica_ni_revisa(client):
    assert client.post("/api/v1/classify", json=CLASSIFY, headers=bearer()).status_code == 403
    assert client.get("/api/v1/review-queue", headers=bearer()).status_code == 403


def test_staff_por_claim_puede_revisar(client):
    assert client.get("/api/v1/review-queue", headers=bearer(is_staff=True)).status_code == 200


def test_interno_sin_moderador_identificado(client):
    r = client.get("/api/v1/review-queue", headers=internal(staff=True))
    assert r.status_code == 400
    assert client.get("/api/v1/review-queue", headers=internal("5")).status_code == 403


@pytest.mark.parametrize(
    "mode, claims, allow, expected",
    [
        ("claim", {"is_staff": True}, [], True),
        ("claim", {}, ["7"], False),
        ("allowlist", {"is_staff": True}, [], False),
        ("allowlist", {}, ["7"], True),
        ("claim_or_allowlist", {}, ["7"], True),
        ("claim_or_allowlist", {"is_staff": True}, [], True),
        ("claim_or_allowlist", {}, [], False),
    ],
)
def test_modos_de_staff(monkeypatch, mode, claims, allow, expected):
    monkeypatch.setattr(settings, "STAFF_CHECK_MODE", mode)
    monkeypatch.setattr(settings, "STAFF_USER_IDS", allow)
    assert security._staff_from_claims(claims, "7") is expected


def test_user_id_preferido_sobre_sub(client, monkeypatch):
    monkeypatch.setattr(settings, "STAFF_CHECK_MODE", "allowlist")
    monkeypatch.setattr(settings, "STAFF_USER_IDS", ["42"])
    h = bearer(user_id=42, sub="999")
    assert client.get("/api/v1/review-queue", headers=h).status_code == 200
