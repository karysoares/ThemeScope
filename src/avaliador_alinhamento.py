"""
Avaliador de alinhamento temático — fachada sobre o Experimento 2 (prompt, zero-shot).

Mantém o contrato histórico (RequisicaoAlinhamento) sem duplicar cliente HTTP ou parsing JSON.
Para novos usos, prefira ``src.experimentos.exp2_prompt_engineering.avaliar`` diretamente.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .experimentos import exp2_prompt_engineering as exp2
from .utils import RespostaAlinhamento, SpanEvidencia, provedor_llm_atual


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


class AvaliadorAlinhamentoTematico:
    """
    Estima o alinhamento temático via LLM (zero-shot), usando o mesmo pipeline que Exp2.

    - ``THEMESCOPE_LLM_PROVIDER=openai``: exige ``OPENAI_API_KEY`` (ou ``chave_api``).
    - ``THEMESCOPE_LLM_PROVIDER=ollama``: não exige chave; usa Ollama local.
    """

    def __init__(self, chave_api: str | None = None) -> None:
        self.chave_api = chave_api or os.environ.get("OPENAI_API_KEY", "")
        if provedor_llm_atual() == "openai" and not self.chave_api:
            raise ValueError(
                "Chave de API da OpenAI é obrigatória com THEMESCOPE_LLM_PROVIDER=openai. "
                "Passe chave_api= ou defina OPENAI_API_KEY."
            )

    def avaliar(self, requisicao: RequisicaoAlinhamento) -> RespostaAlinhamento:
        return exp2.avaliar(
            student_id=requisicao.student_id,
            text_id=requisicao.text_id,
            student_text=requisicao.student_text,
            theme_id=requisicao.theme_id,
            theme_description=requisicao.theme_description,
            chave_api=self.chave_api,
            modo="zero_shot",
        )


__all__ = [
    "AvaliadorAlinhamentoTematico",
    "RequisicaoAlinhamento",
    "RespostaAlinhamento",
    "SpanEvidencia",
    "VERSAO_MODELO",
]

VERSAO_MODELO = exp2.VERSAO_ZERO_SHOT
