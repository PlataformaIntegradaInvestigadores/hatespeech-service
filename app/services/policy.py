"""Política de moderación: traduce p(odio) a etiqueta, decisión HITL y acción visible.

Reglas (umbrales congelados en validación de la TIC, configurables por entorno):
- etiqueta "odio" si p >= CLASS_THRESHOLD (0.5);
- confianza = max(p, 1 - p); si confianza < HITL_TAU el caso se DERIVA a revisión humana,
  si no, el modelo decide (AUTO);
- censura suave: si p >= HIDE_THRESHOLD (0.80) el contenido se oculta con advertencia y se
  ordena al final del hilo.
"""

from __future__ import annotations

from dataclasses import dataclass

LABEL_HATE = "odio"
LABEL_NOT_HATE = "no_odio"
AUTO = "AUTO"
DERIVAR = "DERIVAR"

REASON_HIDDEN_BY_MODEL = "hidden_by_model"
REASON_REPORTED = "reported"
REASON_DEFERRED = "deferred"
MODEL_REASONS = (REASON_HIDDEN_BY_MODEL, REASON_DEFERRED)


@dataclass(frozen=True)
class PolicyOutcome:
    label: str
    confidence: float
    decision: str
    action: str  # visible | hidden
    reasons: tuple[str, ...]


def evaluate(
    p_hate: float, class_threshold: float, hide_threshold: float, tau: float
) -> PolicyOutcome:
    label = LABEL_HATE if p_hate >= class_threshold else LABEL_NOT_HATE
    confidence = max(p_hate, 1.0 - p_hate)
    decision = AUTO if confidence >= tau else DERIVAR
    action = "hidden" if p_hate >= hide_threshold else "visible"
    reasons: list[str] = []
    if action == "hidden":
        reasons.append(REASON_HIDDEN_BY_MODEL)
    if decision == DERIVAR:
        reasons.append(REASON_DEFERRED)
    return PolicyOutcome(label, round(confidence, 6), decision, action, tuple(reasons))
