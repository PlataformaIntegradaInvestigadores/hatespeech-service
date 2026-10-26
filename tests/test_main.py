from fastapi.testclient import TestClient

from app.main import app


def test_root():
    with TestClient(app) as client:
        r = client.get("/")
    assert r.status_code == 200
    assert "moderación" in r.json()["message"]


def test_openapi_strips_internal_prefix():
    app.openapi_schema = None
    with TestClient(app) as client:
        schema = client.get("/api/v1/openapi.json").json()
        # Segunda llamada reutiliza el schema cacheado.
        client.get("/api/v1/openapi.json")
    assert schema["servers"][0]["url"] == "/api/hatespeech"
    assert all(not p.startswith("/api/v1") for p in schema["paths"])


def test_local_ip_handles_resolution_error(monkeypatch):
    import app.main as main

    def boom(_):
        raise OSError("sin DNS")

    monkeypatch.setattr(main.socket, "gethostbyname", boom)
    assert main._local_ip() == "unknown"
