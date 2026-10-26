"""Utilidades puras de explicabilidad (sin torch), replicadas de los scripts de la TIC.

- `textos_por_offset` y `palabras`: idénticas a `xai_val_ecu.py` (Fase 7d). El texto de cada
  subtoken sale del `offset_mapping` para no romper tildes/eñes del BPE byte-level, y la
  saliencia de los subtokens se suma por palabra (regex \\w+ sobre el texto original).
- `normalizar`: saliencia normalizada a suma 1 (misma fórmula de `saliencia_gradiente`).
- `top_palabras`: ranking por saliencia para mostrar al moderador.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_WORD_RE = re.compile(r"\w+")


def textos_por_offset(text: str, offsets: Sequence[tuple[int, int]]) -> list[str]:
    return [text[a:b] if b > a else "" for a, b in offsets]


def palabras(
    text: str, offsets: Sequence[tuple[int, int]], sal: Sequence[float]
) -> list[dict]:
    """Agrega la saliencia de los subtokens a palabras (regex \\w+ sobre el texto)."""
    out = []
    for m in _WORD_RE.finditer(text):
        a, b = m.span()
        tot = sum(
            float(s_) for (x, y), s_ in zip(offsets, sal, strict=False) if y > x and x < b and y > a
        )
        out.append({"palabra": m.group(0), "ini": a, "fin": b, "saliencia": round(tot, 6)})
    return out


def normalizar(valores: Sequence[float]) -> list[float]:
    total = float(sum(abs(v) for v in valores)) + 1e-12
    return [abs(float(v)) / total for v in valores]


def top_palabras(items: list[dict], k: int) -> list[dict]:
    """Palabras con mayor saliencia (descarta las de saliencia 0)."""
    candidatas = [p for p in items if p["saliencia"] > 0]
    candidatas.sort(key=lambda p: p["saliencia"], reverse=True)
    return candidatas[:k]
