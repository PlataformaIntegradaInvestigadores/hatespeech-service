import pytest

from app.services.moderation_service import (
    CaseNotFoundError,
    ModerationService,
    SelfReportError,
)
from tests.conftest import DUDOSO, NEUTRAL, ODIO


@pytest.fixture
def svc(db, settings):
    return ModerationService(db, settings)


def test_neutral_queda_visible_sin_revision(svc, classifier):
    case, out = svc.classify(classifier, NEUTRAL, "comment", "1", author_id="u1")
    assert out.decision == "AUTO"
    assert (case.visibility, case.review_status, case.reasons) == ("visible", "none", [])
    assert case.model_backend == "mock"
    assert case.thresholds["hitl_tau"] == 0.65


def test_odio_alto_se_oculta_y_entra_a_cola(svc, classifier):
    case, out = svc.classify(classifier, ODIO, "post", "9")
    assert out.action == "hidden"
    assert (case.visibility, case.review_status) == ("hidden", "pending")
    assert "hidden_by_model" in case.reasons
    assert case.explanation


def test_zona_dudosa_se_deriva_pero_sigue_visible(svc, classifier):
    case, out = svc.classify(classifier, DUDOSO, "comment", "2")
    assert out.decision == "DERIVAR"
    assert (case.visibility, case.review_status, case.reasons) == (
        "visible",
        "pending",
        ["deferred"],
    )


def test_reclasificar_mismo_contenido_actualiza_el_caso(svc, classifier):
    svc.classify(classifier, ODIO, "comment", "3")
    case, _ = svc.classify(classifier, NEUTRAL, "comment", "3")
    assert (case.visibility, case.review_status, case.reasons) == ("visible", "none", [])


def test_decision_humana_bloquea_reclasificacion_si_texto_igual(svc, classifier):
    case, _ = svc.classify(classifier, ODIO, "comment", "4")
    svc.decide(case.id, "mod1", "approve", "sátira, no es odio")
    case, _ = svc.classify(classifier, ODIO, "comment", "4")
    assert (case.visibility, case.review_status) == ("visible", "resolved")


def test_edicion_tras_revision_reevalua(svc, classifier):
    case, _ = svc.classify(classifier, NEUTRAL, "comment", "5")
    svc.decide(case.id, "mod1", "approve")
    case, _ = svc.classify(classifier, ODIO, "comment", "5")
    assert (case.visibility, case.review_status) == ("hidden", "pending")
    assert case.resolved_by is None


def test_bloqueado_es_definitivo(svc, classifier):
    case, _ = svc.classify(classifier, ODIO, "comment", "6")
    svc.decide(case.id, "mod1", "block")
    case, _ = svc.classify(classifier, NEUTRAL, "comment", "6")
    assert case.visibility == "blocked"


def test_reportes_idempotentes_y_umbral(svc, classifier):
    svc.classify(classifier, NEUTRAL, "comment", "7", author_id="autor")
    for i in range(4):
        case, created, total, crossed = svc.report("comment", "7", f"r{i}")
        assert created and not crossed
    case, created, total, crossed = svc.report("comment", "7", "r0")
    assert (created, total, crossed) == (False, 4, False)
    case, created, total, crossed = svc.report("comment", "7", "r4", reason="insulto")
    assert (created, total, crossed) == (True, 5, True)
    assert (case.visibility, case.review_status, case.reasons) == (
        "hidden",
        "pending",
        ["reported"],
    )


def test_reporte_de_contenido_no_clasificado_crea_caso(svc):
    case, created, total, _ = svc.report("post", "77", "r1", text="hola", author_id="a1")
    assert created and total == 1
    assert (case.text, case.author_id, case.p_hate) == ("hola", "a1", None)


def test_no_se_puede_autoreportar(svc, classifier):
    svc.classify(classifier, NEUTRAL, "comment", "8", author_id="autor")
    with pytest.raises(SelfReportError):
        svc.report("comment", "8", "autor")


def test_tras_aprobar_solo_cuentan_reportes_nuevos(svc, settings, classifier):
    settings.REPORT_THRESHOLD = 2
    svc.classify(classifier, NEUTRAL, "comment", "10")
    svc.report("comment", "10", "a")
    case, *_ = svc.report("comment", "10", "b")
    assert case.visibility == "hidden"
    svc.decide(case.id, "mod", "approve")
    case, _, total, crossed = svc.report("comment", "10", "c")
    assert (total, crossed, case.visibility) == (3, False, "visible")
    case, _, total, crossed = svc.report("comment", "10", "d")
    assert (total, crossed, case.visibility, case.review_status) == (4, True, "hidden", "pending")


def test_reporte_no_reabre_bloqueado(svc, settings, classifier):
    settings.REPORT_THRESHOLD = 1
    case, _ = svc.classify(classifier, ODIO, "comment", "11")
    svc.decide(case.id, "mod", "block")
    case, *_ = svc.report("comment", "11", "x")
    assert case.visibility == "blocked"


def test_cola_filtros_y_paginacion(svc, classifier):
    svc.classify(classifier, ODIO, "comment", "20")
    svc.classify(classifier, DUDOSO, "post", "21")
    svc.classify(classifier, NEUTRAL, "post", "22")
    rows, total = svc.list_queue()
    assert total == 2
    rows, total = svc.list_queue(reason="deferred")
    assert [c.content_id for c in rows] == ["21"]
    rows, total = svc.list_queue(content_type="comment")
    assert [c.content_id for c in rows] == ["20"]
    rows, total = svc.list_queue(limit=1, offset=1)
    assert total == 2 and len(rows) == 1
    svc.decide(rows[0].id, "mod", "keep_hidden")
    assert svc.list_queue(status="resolved")[1] == 1
    assert svc.list_queue(status="all")[1] == 2


def test_decision_guarda_auditoria(svc, classifier):
    case, _ = svc.classify(classifier, ODIO, "comment", "30")
    case, audit = svc.decide(case.id, "mod7", "keep_hidden", "insulto directo")
    assert audit.previous_visibility == "hidden"
    assert audit.previous_review_status == "pending"
    assert audit.previous_reasons == ["hidden_by_model"]
    assert audit.new_visibility == "hidden"
    assert audit.p_hate_at_decision == case.p_hate
    assert case.resolved_by == "mod7" and len(case.decisions) == 1


def test_caso_inexistente(svc):
    with pytest.raises(CaseNotFoundError):
        svc.get_case(999)
