"""Clasificador léxico determinista para desarrollo y CI (sin GPU).

NO es el modelo de la TIC: solo imita su contrato (p_odio + saliencia por palabra)
para poder probar la política de moderación, la persistencia y la API.
"""

from __future__ import annotations

from app.application.ports import ClassificationResult, HateSpeechClassifier
from app.domain.xai import normalizar, palabras, top_palabras

# Pesos arbitrarios: cada término suma a un logit que se pasa por una función acotada.
LEXICO: dict[str, float] = {
    "odio": 1.4,
    "asco": 1.2,
    "basura": 1.2,
    "escoria": 1.6,
    "lacra": 1.6,
    "parasitos": 1.3,
    "parásitos": 1.3,
    "inutil": 0.8,
    "inútil": 0.8,
    "idiota": 1.0,
    "largate": 0.9,
    "lárgate": 0.9,
    "matar": 1.5,
}


class MockClassifier(HateSpeechClassifier):
    name = "mock"

    def __init__(self, top_k: int = 8) -> None:
        self.top_k = top_k
        self._ready = False

    def load(self) -> None:
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def predict(self, text: str, explain: bool = True) -> ClassificationResult:
        palabras_txt = [(m["palabra"], m["ini"], m["fin"]) for m in palabras(text, [], [])]
        pesos = [LEXICO.get(p.lower(), 0.0) for p, _, _ in palabras_txt]
        score = sum(pesos)
        # 0.1 sin términos; satura hacia 0.99 con varios términos.
        p_hate = round(min(0.99, 0.1 + score / (score + 1.0) * 0.89), 6)
        if not explain:
            return ClassificationResult(p_hate=p_hate)
        offsets = [(a, b) for _, a, b in palabras_txt]
        sal = normalizar(pesos) if score > 0 else [0.0] * len(pesos)
        items = palabras(text, offsets, sal)
        return ClassificationResult(
            p_hate=p_hate,
            explanation=top_palabras(items, self.top_k),
            explanation_method="mock_lexico",
        )
