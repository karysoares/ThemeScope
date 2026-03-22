from fastapi.testclient import TestClient

import main
from src.utils import RespostaAlinhamento, SpanEvidencia


client = TestClient(main.app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_avaliar_retorna_contrato_com_mock(monkeypatch):
    def _fake_avaliar(**kwargs):
        return RespostaAlinhamento(
            student_id=kwargs["student_id"],
            text_id=kwargs["text_id"],
            theme_id=kwargs["theme_id"],
            alignment_score=0.66,
            evidence_spans=[
                SpanEvidencia(
                    start_char=0,
                    end_char=10,
                    span_text="trecho",
                    label="parcial",
                )
            ],
            model_version="hibrido-v1.0.0",
            metadados={"experimento": "hibrido", "llm_acionado": False},
        )

    monkeypatch.setattr(main.exp3, "avaliar", _fake_avaliar)

    payload = {
        "student_id": "aluno_01",
        "text_id": "redacao_01",
        "student_text": "Texto de teste",
        "theme_id": "tema_01",
        "theme_description": "Tema de teste",
        "experimento": "exp3_hibrido",
        "modo_prompt": "few_shot",
        "chave_api": "sk-teste",
    }
    resp = client.post("/avaliar", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["alignment_score"] == 0.66
    assert data["model_version"] == "hibrido-v1.0.0"
    assert data["metadados"]["experimento"] == "hibrido"


def test_avaliar_sem_chave_funciona_no_provedor_local(monkeypatch):
    def _fake_avaliar(**kwargs):
        return RespostaAlinhamento(
            student_id=kwargs["student_id"],
            text_id=kwargs["text_id"],
            theme_id=kwargs["theme_id"],
            alignment_score=0.5,
            evidence_spans=[],
            model_version="hibrido-v1.0.0",
            metadados={"experimento": "hibrido", "llm_acionado": False},
        )

    monkeypatch.setattr(main.exp3, "avaliar", _fake_avaliar)

    payload = {
        "student_id": "aluno_01",
        "text_id": "redacao_01",
        "student_text": "Texto de teste",
        "theme_id": "tema_01",
        "theme_description": "Tema de teste",
    }
    resp = client.post("/avaliar", json=payload)
    assert resp.status_code == 200
