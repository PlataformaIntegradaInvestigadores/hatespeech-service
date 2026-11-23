from tests.conftest import BANDA, DUDOSO, NEUTRAL, ODIO, bearer, internal

API = "/api/v1"


def classify(client, text, content_id, content_type="comment", **extra):
    body = {"text": text, "content_type": content_type, "content_id": content_id, **extra}
    r = client.post(f"{API}/classify", json=body, headers=internal())
    assert r.status_code == 200, r.text
    return r.json()


def test_health_mock_online(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["global_status"] == "Online"
    assert {s["name"] for s in body["groups"][0]["services"]} == {"modelo-mock", "base-de-datos"}
    assert body["groups"][0]["services"][0]["device"] == "cpu"


def test_health_offline_si_el_modelo_falla(client):
    client.app.state.classifier = None
    client.app.state.classifier_error = "sin GPU"
    assert client.get("/health").status_code == 503
    r = client.post(
        f"{API}/classify",
        json={"text": "x", "content_type": "post", "content_id": 1},
        headers=internal(),
    )
    assert r.status_code == 503
    assert "sin GPU" in r.json()["detail"]


def test_health_offline_si_la_bd_falla(client):
    class Roto:
        def connect(self):
            raise RuntimeError("bd caída")

    real = client.app.state.engine
    client.app.state.engine = Roto()
    try:
        assert client.get("/health").status_code == 503
    finally:
        client.app.state.engine = real


def test_classify_odio_oculta_con_explicacion(client):
    body = classify(client, ODIO, 1, author_id=10)
    assert body["label"] == "odio"
    assert body["action"] == body["visibility"] == "hidden"
    assert body["decision"] == "AUTO"
    assert body["reasons"] == ["hidden_by_model"]
    assert body["content_id"] == "1"
    assert body["explanation"]["method"] == "mock_lexico"
    assert body["explanation"]["top_tokens"]
    assert body["model"] == {"backend": "mock", "version": "qlora-qwen25-3b-es-ecu-config3"}
    assert body["thresholds"] == {
        "class_threshold": 0.5,
        "hide_threshold": 0.8,
        "hitl_tau": 0.65,
        "review_band_min": 0.65,
        "report_threshold": 5,
    }


def test_classify_sin_explicacion(client):
    body = classify(client, ODIO, 2, explain=False)
    assert body["explanation"]["top_tokens"] == []


def test_classify_dudoso_se_deriva(client):
    body = classify(client, DUDOSO, 3)
    assert (body["decision"], body["visibility"], body["reasons"]) == (
        "DERIVAR",
        "visible",
        ["deferred"],
    )


def test_classify_franja_065_080_va_a_revision_visible(client):
    body = classify(client, BANDA, 4)
    assert (body["label"], body["decision"], body["action"]) == ("odio", "AUTO", "visible")
    assert (body["visibility"], body["reasons"]) == ("visible", ["review_band"])
    mod = internal("centinela-admin")
    page = client.get(f"{API}/review-queue", params={"reason": "review_band"}, headers=mod).json()
    assert [c["content_id"] for c in page["items"]] == ["4"]


def test_classify_por_api_interna_y_validacion(client):
    h = internal()
    ok = client.post(
        f"{API}/classify",
        json={"text": NEUTRAL, "content_type": "post", "content_id": 4},
        headers=h,
    )
    assert ok.status_code == 200
    bad = client.post(
        f"{API}/classify", json={"text": "", "content_type": "story", "content_id": 4}, headers=h
    )
    assert bad.status_code == 422


def test_reportes_flujo_completo(client):
    classify(client, NEUTRAL, 50, author_id=1)
    for uid in range(2, 6):
        r = client.post(
            f"{API}/reports",
            json={"content_type": "comment", "content_id": 50},
            headers=bearer(user_id=uid),
        )
        assert r.status_code == 201 and not r.json()["threshold_reached"]
    dup = client.post(
        f"{API}/reports",
        json={"content_type": "comment", "content_id": 50},
        headers=bearer(user_id=2),
    )
    assert (
        dup.status_code == 200
        and dup.json()["created"] is False
        and dup.json()["report_count"] == 4
    )
    last = client.post(
        f"{API}/reports",
        json={"content_type": "comment", "content_id": 50, "reporter_id": 6, "reason": "insulto"},
        headers=internal(),
    )
    body = last.json()
    assert last.status_code == 201
    assert (body["report_count"], body["threshold_reached"], body["visibility"]) == (
        5,
        True,
        "hidden",
    )
    assert body["reasons"] == ["reported"]

    status = client.get(f"{API}/moderation-status/comment/50", headers=internal()).json()
    assert status["show_warning"] and status["sort_last"]
    assert status["reasons"] == ["reported"] and status["report_count"] == 5


def test_reporte_validaciones(client):
    classify(client, NEUTRAL, 60, author_id=1)
    own = client.post(
        f"{API}/reports",
        json={"content_type": "comment", "content_id": 60},
        headers=bearer(user_id=1),
    )
    assert own.status_code == 400
    spoof = client.post(
        f"{API}/reports",
        json={"content_type": "comment", "content_id": 60, "reporter_id": 9},
        headers=bearer(user_id=2),
    )
    assert spoof.status_code == 403
    anon = client.post(
        f"{API}/reports", json={"content_type": "comment", "content_id": 60}, headers=internal()
    )
    assert anon.status_code == 422
    via_header = client.post(
        f"{API}/reports", json={"content_type": "comment", "content_id": 60}, headers=internal("3")
    )
    assert via_header.status_code == 201


def test_estado_de_contenido_desconocido(client):
    r = client.get(f"{API}/moderation-status/post/404", headers=bearer())
    assert r.status_code == 200
    assert r.json()["classified"] is False and r.json()["visibility"] == "visible"
    assert client.get(f"{API}/moderation-status/story/1", headers=bearer()).status_code == 422


def test_cola_detalle_y_decision(client):
    classify(client, ODIO, 70)
    classify(client, DUDOSO, 71, content_type="post")
    classify(client, NEUTRAL, 72)
    mod = internal("centinela-admin")

    page = client.get(f"{API}/review-queue", headers=mod).json()
    assert page["total"] == 2
    deferred = client.get(f"{API}/review-queue", params={"reason": "deferred"}, headers=mod).json()
    assert [c["content_id"] for c in deferred["items"]] == ["71"]
    comments = client.get(
        f"{API}/review-queue", params={"content_type": "comment"}, headers=mod
    ).json()
    case_id = comments["items"][0]["id"]
    assert comments["items"][0]["text"] == ODIO

    client.post(
        f"{API}/reports",
        json={"content_type": "comment", "content_id": 70, "reason": "odio"},
        headers=bearer(user_id=8),
    )
    detail = client.get(f"{API}/review-queue/{case_id}", headers=mod).json()
    assert detail["report_reasons"] == ["odio"] and detail["decisions"] == []

    r = client.post(
        f"{API}/review-queue/{case_id}/decision",
        json={"decision": "approve", "notes": "contexto"},
        headers=mod,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["case"]["visibility"] == "visible" and body["case"]["review_status"] == "resolved"
    assert body["decision"]["moderator_id"] == "centinela-admin"
    assert body["decision"]["previous_visibility"] == "hidden"

    detail = client.get(f"{API}/review-queue/{case_id}", headers=mod).json()
    assert len(detail["decisions"]) == 1
    resolved = client.get(f"{API}/review-queue", params={"status": "resolved"}, headers=mod).json()
    assert resolved["total"] == 1
    status = client.get(f"{API}/moderation-status/comment/70", headers=internal()).json()
    assert status["visibility"] == "visible" and not status["show_warning"]


def test_cola_errores(client):
    mod = internal("centinela-admin")
    assert client.get(f"{API}/review-queue/999", headers=mod).status_code == 404
    assert (
        client.post(
            f"{API}/review-queue/999/decision", json={"decision": "block"}, headers=mod
        ).status_code
        == 404
    )
    assert (
        client.post(
            f"{API}/review-queue/1/decision", json={"decision": "borrar"}, headers=mod
        ).status_code
        == 422
    )
    assert client.get(f"{API}/review-queue", params={"limit": 0}, headers=mod).status_code == 422


def test_bloqueo_por_admin(client):
    case_id = classify(client, ODIO, 80)["case_id"]
    mod = internal("centinela-admin")
    r = client.post(
        f"{API}/review-queue/{case_id}/decision", json={"decision": "block"}, headers=mod
    )
    assert r.json()["case"]["visibility"] == "blocked"
    status = client.get(f"{API}/moderation-status/comment/80", headers=internal()).json()
    assert status["visibility"] == "blocked" and not status["show_warning"]


def test_arranque_sin_modelo_reporta_error(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main

    def falla(_settings):
        raise RuntimeError("CUDA no disponible")

    monkeypatch.setattr(main, "build_classifier", falla)
    with TestClient(main.app) as c:
        assert c.get("/health").status_code == 503
        assert c.app.state.classifier_error == "CUDA no disponible"
