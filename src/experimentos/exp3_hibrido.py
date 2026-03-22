from __future__ import annotations

from typing import Literal

from ..utils import (
    RespostaAlinhamento,
    chunkar_por_sentenca,
    similaridade_cosseno, # Ainda usado em _calcular_score_embedding
    CUSTO_LLM_POR_REQ_USD,
)
from . import exp2_prompt_engineering as prompt
from .embedding_logic import ( # Importa as funções movidas
    _obter_embeddings_para_analise,
    _similaridades_para_score,
    _selecionar_evidencias,
)

VERSAO = "hibrido-v1.0.1" 

# Limiares de decisão — calibrar com dados reais anotados
LIMIAR_BAIXO: float = 0.35   # abaixo → embedding é suficiente (claramente fora)
LIMIAR_ALTO:  float = 0.65   # acima  → embedding é suficiente (claramente alinhado)

MODO_PROMPT_PADRAO: Literal["zero_shot", "few_shot"] = "few_shot" # Tipagem corrigida


def _calcular_score_embedding(
    student_text: str,
    theme_description: str,
    chave_api: str,
) -> tuple[float, list[str], list[float], float]:
    """
    Calcula score de embedding e retorna chunks, similaridades e custo para reuso.

    Returns:
        (score, chunks, sims, custo_embedding_usd)
    """
    chunks    = chunkar_por_sentenca(student_text)
    entradas  = [theme_description, *chunks]

    vetores, custo_embedding = _obter_embeddings_para_analise(entradas, chave_api)

    if not vetores: # Caso não haja embeddings válidos
        return 0.0, chunks, [], custo_embedding

    emb_tema  = vetores[0]
    embs      = vetores[1:]
    sims      = [similaridade_cosseno(e, emb_tema) for e in embs]
    score     = _similaridades_para_score(sims)
    return score, chunks, sims, custo_embedding


def avaliar(
    student_id: str | int,
    text_id: str | int,
    student_text: str,
    theme_id: str | int,
    theme_description: str,
    chave_api: str,
    limiar_baixo: float = LIMIAR_BAIXO,
    limiar_alto:  float = LIMIAR_ALTO,
    modo_prompt:  Literal["zero_shot", "few_shot"] = MODO_PROMPT_PADRAO, # Tipagem corrigida
) -> RespostaAlinhamento:
    """
    Avalia o alinhamento via pipeline híbrido (embedding gate + LLM condicional).

    Args:
        student_id:        Identificador do estudante.
        text_id:           Identificador da redação.
        student_text:      Texto completo da redação.
        theme_id:          Identificador do tema.
        theme_description: Enunciado/proposta do tema.
        chave_api:         Chave de API da OpenAI.
        limiar_baixo:      Score abaixo do qual o embedding decide sozinho.
        limiar_alto:       Score acima do qual o embedding decide sozinho.
        modo_prompt:       "zero_shot" ou "few_shot" para a zona de incerteza.

    Returns:
        RespostaAlinhamento com metadados de decisão e custo.
    """
    score_emb, chunks, sims, custo_embedding_base = _calcular_score_embedding(
        student_text, theme_description, chave_api
    )

    zona_incerteza = limiar_baixo < score_emb < limiar_alto

    if not zona_incerteza:
        # Casos claros: usa resultado do embedding diretamente
        evidencias = _selecionar_evidencias(student_text, chunks, sims)

        decisao = "alto_alinhamento" if score_emb >= limiar_alto else "baixo_alinhamento"

        return RespostaAlinhamento(
            student_id=student_id,
            text_id=text_id,
            theme_id=theme_id,
            alignment_score=score_emb,
            evidence_spans=evidencias,
            model_version=VERSAO,
            metadados={
                "experimento":       "hibrido",
                "decisao_gate":      decisao,
                "llm_acionado":      False,
                "score_embedding":   round(score_emb, 4),
                "score_final":       round(score_emb, 4),
                "limiar_baixo":      limiar_baixo,
                "limiar_alto":       limiar_alto,
                "num_chunks":        len(chunks),
                "custo_usd_estimado": round(custo_embedding_base, 8), # Custo real do embedding
            },
        )

    resposta_llm = prompt.avaliar(
        student_id=student_id,
        text_id=text_id,
        student_text=student_text,
        theme_id=theme_id,
        theme_description=theme_description,
        chave_api=chave_api,
        modo=modo_prompt,
    )

    custo_llm = resposta_llm.metadados.get("custo_usd_estimado", CUSTO_LLM_POR_REQ_USD)

    return RespostaAlinhamento(
        student_id=student_id,
        text_id=text_id,
        theme_id=theme_id,
        alignment_score=resposta_llm.alignment_score,
        evidence_spans=resposta_llm.evidence_spans,
        model_version=VERSAO,
        metadados={
            "experimento":        "hibrido",
            "decisao_gate":       "incerteza_llm_acionado",
            "llm_acionado":       True,
            "score_embedding":    round(score_emb, 4),
            "score_final":        resposta_llm.alignment_score,
            "delta_emb_vs_llm":   round(
                abs(resposta_llm.alignment_score - score_emb), 4
            ),
            "raciocinio_llm":     resposta_llm.metadados.get("raciocinio", ""),
            "modo_prompt":        modo_prompt,
            "limiar_baixo":       limiar_baixo,
            "limiar_alto":        limiar_alto,
            "num_chunks":         len(chunks),
            "custo_usd_estimado": round(custo_embedding_base + custo_llm, 6), # Soma o custo do embedding base
        },
    )

def relatorio_custo_qualidade(
    resultados: list[RespostaAlinhamento],
) -> dict[str, object]:
    """
    Gera relatório resumindo custo e taxa de acionamento do LLM
    para uma lista de resultados do experimento híbrido.

    Args:
        resultados: Lista de RespostaAlinhamento do experimento híbrido.

    Returns:
        Dict com métricas de custo, taxa de LLM e score médio.
    """
    total = len(resultados)
    if total == 0:
        return {}

    # Validação para garantir que os resultados são do experimento híbrido
    if resultados and resultados[0].metadados.get("experimento") != "hibrido":
        raise ValueError("relatorio_custo_qualidade espera resultados do experimento híbrido.")

    llm_acionados   = [r for r in resultados if r.metadados.get("llm_acionado")]
    apenas_emb      = [r for r in resultados if not r.metadados.get("llm_acionado")]
    custo_total     = sum(r.metadados.get("custo_usd_estimado", 0) for r in resultados)
    score_medio     = sum(r.alignment_score for r in resultados) / total

    custo_llm_puro = total * CUSTO_LLM_POR_REQ_USD # Usa a constante do utils

    return {
        "total_avaliacoes":        total,
        "llm_acionado_n":          len(llm_acionados),
        "llm_acionado_pct":        round(len(llm_acionados) / total * 100, 1),
        "apenas_embedding_n":      len(apenas_emb),
        "apenas_embedding_pct":    round(len(apenas_emb) / total * 100, 1),
        "custo_total_usd":         round(custo_total, 6),
        "custo_medio_por_req_usd": round(custo_total / total, 6),
        "custo_llm_puro_usd":      round(custo_llm_puro, 6),
        "economia_pct":            round(
            (1 - custo_total / custo_llm_puro) * 100, 1
        ) if custo_llm_puro > 0 else 0,
        "score_medio":             round(score_medio, 4),
    }
