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


def test_jwt_no_abre_la_cola(client):
    assert client.get("/api/v1/review-queue", headers=bearer(is_staff=True)).status_code == 403


def test_interno_sin_admin_identificado(client):
    assert client.get("/api/v1/review-queue", headers=internal()).status_code == 400
    ok = client.get("/api/v1/review-queue", headers=internal("centinela-admin"))
    assert ok.status_code == 200


def test_user_id_preferido_sobre_sub(client):
    h = bearer(user_id=42, sub="999")
    assert client.get("/api/v1/moderation-status/post/1", headers=h).status_code == 200
