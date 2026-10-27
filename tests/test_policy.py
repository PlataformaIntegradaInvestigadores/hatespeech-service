import pytest

from app.services import policy


@pytest.mark.parametrize(
    "p, label, decision, action, reasons",
    [
        (0.10, "no_odio", "AUTO", "visible", ()),
        (0.40, "no_odio", "DERIVAR", "visible", ("deferred",)),
        (0.50, "odio", "DERIVAR", "visible", ("deferred",)),
        (0.70, "odio", "AUTO", "visible", ()),
        (0.80, "odio", "AUTO", "hidden", ("hidden_by_model",)),
        (0.97, "odio", "AUTO", "hidden", ("hidden_by_model",)),
    ],
)
def test_politica_tau_ecu_sin_franja(p, label, decision, action, reasons):
    out = policy.evaluate(p, class_threshold=0.5, hide_threshold=0.8, tau=0.65)
    assert (out.label, out.decision, out.action, out.reasons) == (label, decision, action, reasons)
    assert out.confidence == pytest.approx(max(p, 1 - p))


def test_tau_es_alternativo_deriva_mas():
    # Con tau ES = 0.80, p=0.75 (confianza 0.75) ya no es AUTO.
    assert policy.evaluate(0.75, 0.5, 0.8, 0.80).decision == "DERIVAR"
    assert policy.evaluate(0.75, 0.5, 0.8, 0.65).decision == "AUTO"


def test_oculto_y_derivado_a_la_vez():
    out = policy.evaluate(0.82, 0.5, 0.8, 0.90)
    assert out.reasons == ("hidden_by_model", "deferred")


@pytest.mark.parametrize(
    "p, decision, action, reasons",
    [
        (0.10, "AUTO", "visible", ()),
        (0.64, "DERIVAR", "visible", ("deferred",)),
        (0.65, "AUTO", "visible", ("review_band",)),
        (0.70, "AUTO", "visible", ("review_band",)),
        (0.7999, "AUTO", "visible", ("review_band",)),
        (0.80, "AUTO", "hidden", ("hidden_by_model",)),
    ],
)
def test_franja_de_revision(p, decision, action, reasons):
    out = policy.evaluate(p, 0.5, 0.8, 0.65, review_band_min=0.65)
    assert (out.decision, out.action, out.reasons) == (decision, action, reasons)


def test_franja_no_duplica_la_derivacion_por_tau():
    # Con tau = 0.80, p = 0.70 ya se deriva por τ: un solo motivo.
    assert policy.evaluate(0.70, 0.5, 0.8, 0.80, review_band_min=0.65).reasons == ("deferred",)
