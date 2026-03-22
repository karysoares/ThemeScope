"""
avaliador_alinhamento.py
------------------------
Módulo principal para estimar o grau de alinhamento temático entre a
redação de um estudante e a proposta apresentada.

Decisões de design:
- LLM (OpenAI GPT-4o) como núcleo de inferência zero-shot.
- temperature=0 para saída determinística e reprodutível.
- Saída estruturada via JSON Schema no prompt — sem function calling externo.
- Ancoragem de spans por busca de substring no texto original.
- Retry automático com backoff exponencial (até 3 tentativas).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests



VERSAO_MODELO = "text-alinhamento-v1.0.0"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODELO = "gpt-4o"
MAX_TOKENS = 1024
TEMPERATURA = 0         
MAX_TENTATIVAS = 3
DELAY_BASE_SEGUNDOS = 2


@dataclass
class RequisicaoAlinhamento:
    """Contrato de entrada para o serviço de alinhamento temático."""

    student_id: str | int
    text_id: str | int
    student_text: str
    theme_id: str | int
    theme_description: str

    def para_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "text_id": self.text_id,
            "student_text": self.student_text,
            "theme_id": self.theme_id,
            "theme_description": self.theme_description,
        }


@dataclass
class SpanEvidencia:
    """Trecho do texto que justifica o score de alinhamento."""

    start_char: int
    end_char: int
    span_text: str
    label: str  # "alinhado" | "fora_do_tema" | "parcial"

    def para_dict(self) -> dict[str, Any]:
        return {
            "start_char": self.start_char,
            "end_char": self.end_char,
            "span_text": self.span_text,
            "label": self.label,
        }


@dataclass
class RespostaAlinhamento:
    """Contrato de saída para o serviço de alinhamento"""

    student_id: str | int
    text_id: str | int
    theme_id: str | int
    alignment_score: float
    evidence_spans: list[SpanEvidencia] = field(default_factory=list)
    model_version: str = VERSAO_MODELO

    def para_dict(self) -> dict[str, Any]:
        return {
            "student_id": self.student_id,
            "text_id": self.text_id,
            "theme_id": self.theme_id,
            "alignment_score": round(self.alignment_score, 4),
            "evidence_spans": [s.para_dict() for s in self.evidence_spans],
            "model_version": self.model_version,
        }


PROMPT_SISTEMA = """Você é um sistema especialista em avaliação de redações escolares brasileiras.

Sua tarefa é analisar o grau de alinhamento entre o texto do estudante e o tema proposto.

Retorne SOMENTE um objeto JSON válido, sem markdown, sem texto adicional, seguindo exatamente o schema:

{
  "alignment_score": <float entre 0.0 e 1.0>,
  "raciocinio": "<uma frase explicando o score>",
  "evidence_spans": [
    {"span_text": "<trecho EXATO do texto do estudante>", "label": "<alinhado|fora_do_tema|parcial>"}
  ]
}

Regras obrigatórias:
- alignment_score: 0.0 = completamente fora do tema, 1.0 = perfeitamente alinhado.
- span_text deve ser uma substring EXATA do texto do estudante (case-sensitive).
- label deve ser um dos três valores: "alinhado", "fora_do_tema" ou "parcial".
- Retorne entre 2 e 5 spans evidenciais.
- Nenhum texto fora do JSON.

Tarefa:
- Analise o grau de alinhamento entre o texto do estudante e o tema proposto -> "Os impactos do uso excessivo de redes sociais na saúde mental dos jovens"
- Retorne um score entre 0.0 e 1.0, onde 0.0 significa completamente fora do tema e 1.0 significa perfeitamente alinhado.
- Retorne uma frase explicando o score.
- Retorne entre 2 e 5 spans evidenciais.
- Nenhum texto fora do JSON.

"""


def _construir_prompt_usuario(req: RequisicaoAlinhamento) -> str:
    """Monta o prompt do usuário com tema e redação."""
    return (
        f"TEMA DA REDAÇÃO:\n{req.theme_description}\n\n"
        f"TEXTO DO ESTUDANTE:\n{req.student_text}"
    )


def _ancorar_spans(
    texto_original: str,
    spans_brutos: list[dict[str, str]],
) -> list[SpanEvidencia]:
    """
    Localiza cada span_text dentro do texto_original para calcular
    start_char e end_char. Spans não localizáveis são descartados.

    Args:
        texto_original: Redação completa do estudante.
        spans_brutos: Lista de dicts com 'span_text' e 'label'.

    Returns:
        Lista de SpanEvidencia com offsets de caractere resolvidos.
    """
    anchorados: list[SpanEvidencia] = []
    inicio_busca = 0

    for bruto in spans_brutos:
        span_text = bruto.get("span_text", "").strip()
        label = bruto.get("label", "parcial")

        if not span_text:
            continue

        idx = texto_original.find(span_text, inicio_busca)

        if idx == -1:
            idx = texto_original.lower().find(span_text.lower(), inicio_busca)

        if idx == -1:
            idx = texto_original.lower().find(span_text.lower())

        if idx == -1:
            continue  

        fim = idx + len(span_text)
        anchorados.append(
            SpanEvidencia(
                start_char=idx,
                end_char=fim,
                span_text=texto_original[idx:fim],
                label=label,
            )
        )
        inicio_busca = max(inicio_busca, fim)

    return anchorados


def _chamar_openai(prompt_usuario: str, chave_api: str) -> str:
    """
    Chama a API OpenAI Chat Completions com retry e backoff exponencial.

    Args:
        prompt_usuario: Mensagem formatada do usuário.
        chave_api: Chave de API da OpenAI.

    Returns:
        Texto bruto retornado pelo modelo.

    Raises:
        RuntimeError: Se todas as tentativas falharem.
    """
    cabecalhos = {
        "Authorization": f"Bearer {chave_api}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODELO,
        "temperature": TEMPERATURA,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": PROMPT_SISTEMA},
            {"role": "user", "content": prompt_usuario},
        ],
    }

    ultimo_erro: Exception | None = None
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            resposta = requests.post(
                OPENAI_API_URL,
                headers=cabecalhos,
                json=payload,
                timeout=30,
            )
            resposta.raise_for_status()
            dados = resposta.json()
            return dados["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError) as exc:
            ultimo_erro = exc
            if tentativa < MAX_TENTATIVAS:
                time.sleep(DELAY_BASE_SEGUNDOS ** tentativa)

    raise RuntimeError(
        f"Falha na API OpenAI após {MAX_TENTATIVAS} tentativas: {ultimo_erro}"
    )


def _parsear_resposta_llm(texto_bruto: str) -> dict[str, Any]:
    """
    Parseia a resposta do LLM, removendo eventuais fences de markdown.

    Args:
        texto_bruto: String retornada pelo modelo.

    Returns:
        Dict com os campos do schema de saída.

    Raises:
        ValueError: Se a resposta não for JSON válido.
    """
    limpo = re.sub(r"^```(?:json)?\s*", "", texto_bruto.strip(), flags=re.IGNORECASE)
    limpo = re.sub(r"\s*```$", "", limpo)

    try:
        return json.loads(limpo)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Modelo retornou resposta não-JSON: {texto_bruto!r}"
        ) from exc


class AvaliadorAlinhamentoTematico:
    """
    Estima o alinhamento temático entre a redação de um estudante e o tema proposto.

    Utiliza GPT-4o da OpenAI em modo zero-shot com saída estruturada JSON,
    garantindo reprodutibilidade via temperature=0 e versionamento de prompt.

    Exemplo de uso:
        avaliador = AvaliadorAlinhamentoTematico()
        req = RequisicaoAlinhamento(
            student_id="aluno_01",
            text_id="redacao_01",
            student_text="O uso excessivo de redes sociais...",
            theme_id="tema_01",
            theme_description="Discuta os impactos do uso excessivo de redes sociais..."
        )
        resultado = avaliador.avaliar(req)
        print(resultado.para_dict())
    """

    def __init__(self, chave_api: str | None = None) -> None:
        """
        Inicializa o avaliador.

        Args:
            chave_api: Chave de API da OpenAI. Usa OPENAI_API_KEY do ambiente
                       se não fornecida explicitamente.

        Raises:
            ValueError: Se nenhuma chave de API for encontrada.
        """
        self.chave_api = chave_api or os.environ.get("OPENAI_API_KEY", "")
        if not self.chave_api:
            raise ValueError(
                "Chave de API da OpenAI é obrigatória. Passe chave_api= ou "
                "defina a variável de ambiente OPENAI_API_KEY."
            )

    def avaliar(self, requisicao: RequisicaoAlinhamento) -> RespostaAlinhamento:
        """
        Avalia o alinhamento entre a redação do estudante e o tema proposto.

        Args:
            requisicao: RequisicaoAlinhamento com texto e tema.

        Returns:
            RespostaAlinhamento com alignment_score (0–1) e evidence_spans.

        Raises:
            RuntimeError: Em caso de falha na API após retentativas.
            ValueError: Em caso de saída malformada do modelo.
        """
        prompt_usuario = _construir_prompt_usuario(requisicao)
        texto_bruto = _chamar_openai(prompt_usuario, self.chave_api)
        parseado = _parsear_resposta_llm(texto_bruto)

        score = float(parseado.get("alignment_score", 0.0))
        score = max(0.0, min(1.0, score))  # Garante intervalo [0, 1]

        spans_brutos = parseado.get("evidence_spans", [])
        spans = _ancorar_spans(requisicao.student_text, spans_brutos)

        return RespostaAlinhamento(
            student_id=requisicao.student_id,
            text_id=requisicao.text_id,
            theme_id=requisicao.theme_id,
            alignment_score=score,
            evidence_spans=spans,
            model_version=VERSAO_MODELO,
        )
