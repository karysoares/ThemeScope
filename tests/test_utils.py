from src.utils import chunkar_por_sentenca, parsear_json_llm, similaridade_cosseno


def test_chunkar_por_sentenca_retorna_janela_deslizante():
    texto = "A. B. C. D."
    chunks = chunkar_por_sentenca(texto, tamanho_chunk=2, passo=1)
    assert chunks == ["A. B.", "B. C.", "C. D."]


def test_similaridade_cosseno_intervalo_valido():
    import numpy as np

    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    assert similaridade_cosseno(a, b) == 1.0
    assert 0.0 <= similaridade_cosseno(a, c) <= 1.0


def test_parsear_json_llm_com_fence_markdown():
    bruto = '```json\n{"alignment_score": 0.7}\n```'
    data = parsear_json_llm(bruto)
    assert data["alignment_score"] == 0.7
