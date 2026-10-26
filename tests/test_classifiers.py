import pytest

from app.core.config import Settings
from app.services.classifiers import build_classifier
from app.services.classifiers.mock import MockClassifier
from app.services.xai import normalizar, palabras, textos_por_offset, top_palabras


def test_textos_por_offset_respeta_tildes():
    text = "año niño"
    assert textos_por_offset(text, [(0, 3), (3, 3), (4, 8)]) == ["año", "", "niño"]


def test_palabras_suma_subtokens():
    text = "hola mundo"
    # "mundo" partido en dos subtokens
    offsets = [(0, 4), (4, 7), (7, 10)]
    sal = [0.2, 0.5, 0.3]
    out = palabras(text, offsets, sal)
    assert [p["palabra"] for p in out] == ["hola", "mundo"]
    assert out[1]["saliencia"] == pytest.approx(0.8)
    assert (out[1]["ini"], out[1]["fin"]) == (5, 10)


def test_normalizar_y_top():
    assert sum(normalizar([1, -1, 2])) == pytest.approx(1.0)
    items = [
        {"palabra": "a", "saliencia": 0.0},
        {"palabra": "b", "saliencia": 0.7},
        {"palabra": "c", "saliencia": 0.3},
    ]
    assert [p["palabra"] for p in top_palabras(items, 5)] == ["b", "c"]
    assert len(top_palabras(items, 1)) == 1


def test_mock_texto_neutral_es_bajo():
    clf = MockClassifier()
    clf.load()
    assert clf.ready
    r = clf.predict("Buenos días a todos, gran trabajo")
    assert r.p_hate == pytest.approx(0.1)
    assert r.explanation == []
    assert r.explanation_method == "mock_lexico"


def test_mock_texto_ofensivo_es_alto_y_explica():
    clf = MockClassifier(top_k=2)
    clf.load()
    r = clf.predict("Esa gente es escoria y basura, me dan asco")
    assert r.p_hate >= 0.8
    assert len(r.explanation) == 2
    assert r.explanation[0]["palabra"].lower() == "escoria"


def test_mock_sin_explicacion():
    clf = MockClassifier()
    clf.load()
    r = clf.predict("idiota", explain=False)
    assert r.explanation == [] and r.explanation_method is None
    assert 0.1 < r.p_hate < 0.8


def test_factory_mock_por_defecto():
    clf = build_classifier(Settings(MODEL_BACKEND="mock"))
    assert isinstance(clf, MockClassifier)


def test_factory_qwen_construye_sin_cargar_pesos():
    # La construcción no importa torch; los pesos se cargan recién en load().
    clf = build_classifier(
        Settings(MODEL_BACKEND="qwen", ADAPTER_PATH="/models/x", MAX_LENGTH=64, EXPLANATION_TOP_K=3)
    )
    assert clf.name == "qwen"
    assert clf.ready is False
    assert (clf.adapter_path, clf.max_length, clf.top_k) == ("/models/x", 64, 3)
    with pytest.raises(RuntimeError):
        clf.predict("hola")
