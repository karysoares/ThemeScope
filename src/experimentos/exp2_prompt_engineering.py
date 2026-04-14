"""
src/experimentos/exp2_prompt_engineering.py
--------------------------------------------
Experimento 2 — Prompt Engineering

Compara duas estratégias de prompting para o mesmo modelo (GPT-4o):

  A) Zero-shot
     - System prompt com instrução + schema JSON.
     - Sem exemplos de entrada/saída.
     - Menor custo de tokens; comportamento depende do RLHF do modelo.

  B) Few-shot (3 exemplos calibrados)
     - System prompt + 3 pares (input, output esperado) como "âncoras".
     - Exemplos cobrem alto, parcial e baixo alinhamento.
     - Mais tokens (~900 extras), mas maior consistência de calibração.
     - Os exemplos são embutidos como mensagens user/assistant no histórico.

Ambas as estratégias retornam JSON estruturado com:
  - alignment_score (float 0-1)
  - raciocinio (string)
  - evidence_spans (lista de {span_text, label})

Saída inclui metadados de custo estimado (tokens) para comparação.
"""

from __future__ import annotations

from typing import Literal

from ..utils import (
    RespostaAlinhamento,
    ancorar_spans,
    chamar_chat,
    parsear_json_llm,
    provedor_llm_atual,
)

VERSAO_ZERO_SHOT = "prompt-zero-shot-v1.0.0"
VERSAO_FEW_SHOT  = "prompt-few-shot-v1.0.0"

SYSTEM_PROMPT_BASE = """Você é um sistema especialista em avaliação de redações escolares brasileiras.

Analise o grau de alinhamento entre o texto do estudante e o tema proposto.

Retorne SOMENTE um objeto JSON válido, sem markdown, sem texto adicional:

{
  "alignment_score": <float entre 0.0 e 1.0>,
  "raciocinio": "<uma frase explicando o score>",
  "evidence_spans": [
    {"span_text": "<substring EXATA do texto>", "label": "<alinhado|fora_do_tema|parcial>"}
  ]
}

Regras:
- alignment_score: 0.0 = completamente fora do tema; 1.0 = perfeitamente alinhado.
- span_text deve ser uma substring EXATA do texto do estudante.
- label: "alinhado" (aborda o tema diretamente), "fora_do_tema" (assunto irrelevante),
  "parcial" (relacionado mas tangencial).
- Retorne entre 2 e 5 spans.
- Nenhum texto fora do JSON."""


EXEMPLOS_FEW_SHOT = [
    # Exemplo 1 — Alto alinhamento
    {
        "user": (
            "TEMA:\nDiscuta os impactos do uso excessivo de redes sociais "
            "na saúde mental dos jovens.\n\n"
            "TEXTO:\nO uso desregrado do Instagram e TikTok tem contribuído para "
            "o aumento da ansiedade entre adolescentes. A comparação constante com "
            "padrões de beleza irreais leva à baixa autoestima. Além disso, a "
            "privação de sono causada pelo uso noturno agrava quadros depressivos."
        ),
        "assistant": (
            '{"alignment_score": 0.92, '
            '"raciocinio": "O texto aborda diretamente os impactos das redes sociais '
            'na saúde mental, citando ansiedade, autoestima e depressão.", '
            '"evidence_spans": ['
            '{"span_text": "aumento da ansiedade entre adolescentes", "label": "alinhado"}, '
            '{"span_text": "privação de sono causada pelo uso noturno agrava quadros depressivos", "label": "alinhado"}'
            ']}'
        ),
    },
    # Exemplo 2 — Alinhamento parcial
    {
        "user": (
            "TEMA:\nDiscuta os impactos do uso excessivo de redes sociais "
            "na saúde mental dos jovens.\n\n"
            "TEXTO:\nA tecnologia transformou a educação moderna. Plataformas digitais "
            "permitem acesso ao conhecimento em qualquer lugar. O uso de redes sociais "
            "pode tanto ajudar quanto prejudicar os estudantes. É necessário equilíbrio "
            "no consumo de conteúdo digital para preservar o bem-estar."
        ),
        "assistant": (
            '{"alignment_score": 0.42, '
            '"raciocinio": "O texto menciona redes sociais e bem-estar superficialmente, '
            'mas dedica a maior parte à tecnologia educacional, que não é o tema central.", '
            '"evidence_spans": ['
            '{"span_text": "O uso de redes sociais pode tanto ajudar quanto prejudicar", "label": "parcial"}, '
            '{"span_text": "tecnologia transformou a educação moderna", "label": "fora_do_tema"}'
            ']}'
        ),
    },
    # Exemplo 3 — Baixo alinhamento
    {
        "user": (
            "TEMA:\nDiscuta os impactos do uso excessivo de redes sociais "
            "na saúde mental dos jovens.\n\n"
            "TEXTO:\nAs mudanças climáticas representam o maior desafio do século XXI. "
            "O desmatamento da Amazônia ameaça a biodiversidade brasileira. "
            "Políticas de energia renovável são urgentes para reduzir emissões de CO2. "
            "A educação ambiental deve ser prioridade nas escolas."
        ),
        "assistant": (
            '{"alignment_score": 0.04, '
            '"raciocinio": "O texto trata inteiramente de meio ambiente e clima, '
            'sem qualquer menção a redes sociais ou saúde mental.", '
            '"evidence_spans": ['
            '{"span_text": "mudanças climáticas representam o maior desafio do século XXI", "label": "fora_do_tema"}, '
            '{"span_text": "desmatamento da Amazônia ameaça a biodiversidade", "label": "fora_do_tema"}'
            ']}'
        ),
    },
]


def _prompt_usuario(theme_description: str, student_text: str) -> str:
    return f"TEMA:\n{theme_description}\n\nTEXTO:\n{student_text}"


def _mensagens_zero_shot(
    theme_description: str, student_text: str
) -> list[dict[str, str]]:
    """Constrói histórico de mensagens para zero-shot."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_BASE},
        {"role": "user",   "content": _prompt_usuario(theme_description, student_text)},
    ]


def _mensagens_few_shot(
    theme_description: str, student_text: str
) -> list[dict[str, str]]:
    """
    Constrói histórico de mensagens para few-shot.
    Os 3 exemplos são inseridos como pares user/assistant antes da consulta real.
    """
    msgs: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT_BASE}
    ]
    for ex in EXEMPLOS_FEW_SHOT:
        msgs.append({"role": "user",      "content": ex["user"]})
        msgs.append({"role": "assistant", "content": ex["assistant"]})
    msgs.append({"role": "user", "content": _prompt_usuario(theme_description, student_text)})
    return msgs

def avaliar(
    student_id: str | int,
    text_id: str | int,
    student_text: str,
    theme_id: str | int,
    theme_description: str,
    chave_api: str,
    modo: Literal["zero_shot", "few_shot"] = "few_shot",
) -> RespostaAlinhamento:
    """
    Avalia o alinhamento temático via prompting (zero-shot ou few-shot).

    Args:
        student_id:        Identificador do estudante.
        text_id:           Identificador da redação.
        student_text:      Texto completo da redação.
        theme_id:          Identificador do tema.
        theme_description: Enunciado/proposta do tema.
        chave_api:         Chave de API da OpenAI.
        modo:              "zero_shot" ou "few_shot".

    Returns:
        RespostaAlinhamento com alignment_score, evidence_spans e metadados de custo.
    """
    if modo == "zero_shot":
        mensagens = _mensagens_zero_shot(theme_description, student_text)
        versao    = VERSAO_ZERO_SHOT
    else:
        mensagens = _mensagens_few_shot(theme_description, student_text)
        versao    = VERSAO_FEW_SHOT

    # Estima tokens de entrada (aproximação: 1 token ≈ 4 chars em português)
    tokens_entrada_estimados = sum(len(m["content"]) // 4 for m in mensagens)

    texto_bruto = chamar_chat(mensagens, chave_api)
    parseado    = parsear_json_llm(texto_bruto)

    score = float(parseado.get("alignment_score", 0.0))
    score = max(0.0, min(1.0, score))

    spans_brutos = parseado.get("evidence_spans", [])
    spans = ancorar_spans(student_text, spans_brutos)

    tokens_saida_estimados = len(texto_bruto) // 4

    return RespostaAlinhamento(
        student_id=student_id,
        text_id=text_id,
        theme_id=theme_id,
        alignment_score=score,
        evidence_spans=spans,
        model_version=versao,
        metadados={
            "experimento":               "prompt_engineering",
            "modo":                      modo,
            "modelo_chat":               "llama3.1:8b" if provedor_llm_atual() == "ollama" else "gpt-4o",
            "provedor_llm":              provedor_llm_atual(),
            "raciocinio":                parseado.get("raciocinio", ""),
            "tokens_entrada_estimados":  tokens_entrada_estimados,
            "tokens_saida_estimados":    tokens_saida_estimados,
            "custo_usd_estimado":        round(
                0.0 if provedor_llm_atual() == "ollama" else (
                    tokens_entrada_estimados * 2.5e-6 +
                    tokens_saida_estimados   * 10e-6
                ), 6
            ),
            "num_exemplos_few_shot":     len(EXEMPLOS_FEW_SHOT) if modo == "few_shot" else 0,
        },
    )
