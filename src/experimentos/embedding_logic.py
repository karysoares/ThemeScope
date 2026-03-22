from __future__ import annotations

import numpy as np
from typing import Any

from ..utils import (
    SpanEvidencia,
    obter_embeddings, # Função unificada de obtenção de embeddings
    similaridade_cosseno,
    ancorar_spans,
    CUSTO_EMBEDDING_POR_TOKEN_USD
)

def _obter_embeddings_para_analise(
    entradas: list[str],
    chave_api: str,
) -> tuple[list[np.ndarray], float]:
    """
    Obtém embeddings para uma lista de textos e calcula o custo estimado.

    Args:
        entradas: Lista de textos para obter embeddings.
        chave_api: Chave de API para o provedor de embeddings (e.g., OpenAI).

    Returns:
        Uma tupla contendo:
        - Uma lista de arrays numpy, cada um sendo o embedding de um texto.
        - O custo estimado em USD para a operação de embedding.
    """
    if not entradas:
        return [], 0.0

    vetores = obter_embeddings(entradas, chave_api)
    num_tokens_total = sum(len(text.split()) for text in entradas) # Aproximação
    custo_estimado = num_tokens_total * CUSTO_EMBEDDING_POR_TOKEN_USD

    return vetores, custo_estimado


def _similaridades_para_score(sims: list[float]) -> float:
    """
    Converte uma lista de similaridades de chunks em um score único.
    Usa a média das 3 maiores similaridades para robustez.

    Args:
        sims: Lista de scores de similaridade.

    Returns:
        Um score float único.
    """
    if not sims:
        return 0.0
    # Pega as 3 maiores similaridades ou todas se houver menos de 3
    top_sims = sorted(sims, reverse=True)[:3]
    return float(np.mean(top_sims))


def _selecionar_evidencias(
    texto_original: str,
    chunks: list[str],
    similaridades: list[float],
    limiar_evidencia: float = 0.5, # Limiar para considerar um chunk como evidência
    top_n_evidencias: int = 5,    
) -> list[SpanEvidencia]:
    """
    Seleciona os chunks mais relevantes como evidências e os ancora no texto original.

    Args:
        texto_original: O texto completo do qual os chunks foram extraídos.
        chunks: Lista dos chunks de texto.
        similaridades: Lista dos scores de similaridade correspondentes a cada chunk.
        limiar_evidencia: Score mínimo para um chunk ser considerado uma evidência.
        top_n_evidencias: Número máximo de evidências a serem retornadas.

    Returns:
        Uma lista de objetos SpanEvidencia com os trechos de evidência ancorados.
    """
    if not chunks or not similaridades:
        return []

    candidatos = [
        {"span_text": chunk, "label": "alinhado", "score": sim}
        for chunk, sim in zip(chunks, similaridades)
        if sim >= limiar_evidencia
    ]
    candidatos_ordenados = sorted(candidatos, key=lambda x: x["score"], reverse=True)
    spans_brutos = [
        {"span_text": c["span_text"], "label": c["label"]}
        for c in candidatos_ordenados[:top_n_evidencias]
    ]

    return ancorar_spans(texto_original, spans_brutos)
