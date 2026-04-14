"""
src/experimentos/exp1_embedding_baseline.py
--------------------------------------------
Experimento 1 — Embedding Baseline

Pipeline:
  1. Embedda o tema (proposta) como vetor de referência (sentence-transformers local).
  2. Divide a redação em chunks de N sentenças (sliding window).
  3. Embedda cada chunk localmente.
  4. Calcula similaridade cosseno entre cada chunk e o tema.
  5. Agrega as similaridades por média ponderada (mesma função que o gate do Exp3).
  6. Seleciona os K chunks com maior e menor similaridade como evidências.

Vantagens:
  - Custo US$0 por requisição (após download do modelo local).
  - Latência sub-segundo para textos escolares típicos.
  - Reprodutível: embeddings são determinísticos.

Limitações:
  - Não captura raciocínio causal ou argumentativo.
  - Sensível a textos que usam vocabulário do tema mas de forma tangencial.
  - Score pode ser inflado por redações que listam palavras-chave sem desenvolvê-las.
"""

from __future__ import annotations

import numpy as np

from ..local_embeddings import MODELO_EMBEDDING_LOCAL, codificar_textos
from ..utils import (
    RespostaAlinhamento,
    SpanEvidencia,
    agregar_similaridades_chunks,
    chunkar_por_sentenca,
    similaridade_cosseno,
)

VERSAO = "embedding-baseline-v1.0.0"

TOP_K_EVIDENCIAS = 2
TAMANHO_CHUNK = 3
PASSO_CHUNK = 1


def _selecionar_evidencias(
    texto_original: str,
    chunks: list[str],
    sims: list[float],
    top_k: int = TOP_K_EVIDENCIAS,
) -> list[SpanEvidencia]:
    """
    Seleciona os chunks de maior e menor similaridade como evidências,
    ancorando-os no texto original.

    Args:
        texto_original: Redação completa.
        chunks:         Lista de chunks de texto.
        sims:           Similaridade de cada chunk com o tema.
        top_k:          Número de chunks a selecionar em cada extremo.

    Returns:
        Lista de SpanEvidencia com offsets resolvidos.
    """
    if not chunks:
        return []

    indices_ordenados = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    indices_top      = indices_ordenados[:top_k]
    indices_bottom   = indices_ordenados[-top_k:]
    indices_usados   = set()
    evidencias: list[SpanEvidencia] = []
    inicio_busca = 0

    for idx in indices_top + indices_bottom:
        if idx in indices_usados:
            continue
        indices_usados.add(idx)

        chunk_text = chunks[idx]
        label = "alinhado" if idx in indices_top else "fora_do_tema"
        sim = sims[idx]
        if 0.3 < sim < 0.6:
            label = "parcial"

        pos = texto_original.find(chunk_text, inicio_busca)
        if pos == -1:
            pos = texto_original.lower().find(chunk_text.lower())
        if pos == -1:
            continue

        fim = pos + len(chunk_text)
        evidencias.append(SpanEvidencia(
            start_char=pos,
            end_char=fim,
            span_text=texto_original[pos:fim],
            label=label,
        ))
        inicio_busca = max(inicio_busca, fim)

    return evidencias


def avaliar(
    student_id: str | int,
    text_id: str | int,
    student_text: str,
    theme_id: str | int,
    theme_description: str,
    chave_api: str,
    tamanho_chunk: int = TAMANHO_CHUNK,
    passo_chunk: int = PASSO_CHUNK,
) -> RespostaAlinhamento:
    """
    Avalia o alinhamento usando apenas embeddings (sem LLM de chat).

    Args:
        student_id:        Identificador do estudante.
        text_id:           Identificador da redação.
        student_text:      Texto completo da redação.
        theme_id:          Identificador do tema.
        theme_description: Enunciado/proposta do tema.
        chave_api:         Reservado para compatibilidade de assinatura; não usado (embeddings locais).
        tamanho_chunk:     Número de sentenças por chunk.
        passo_chunk:       Passo da janela deslizante.

    Returns:
        RespostaAlinhamento com alignment_score e evidence_spans.
    """
    chunks = chunkar_por_sentenca(student_text, tamanho_chunk, passo_chunk)

    entradas = [theme_description, *chunks]
    vetores = codificar_textos(entradas)
    emb_tema = vetores[0]
    embs_chunks = vetores[1:]

    sims = [similaridade_cosseno(emb_chunk, emb_tema) for emb_chunk in embs_chunks]

    score = agregar_similaridades_chunks(sims)

    evidencias = _selecionar_evidencias(student_text, chunks, sims)

    return RespostaAlinhamento(
        student_id=student_id,
        text_id=text_id,
        theme_id=theme_id,
        alignment_score=score,
        evidence_spans=evidencias,
        model_version=VERSAO,
        metadados={
            "experimento":        "embedding_baseline",
            "modelo_embedding":   MODELO_EMBEDDING_LOCAL,
            "provedor_embedding": "local_sentence_transformers",
            "num_chunks":         len(chunks),
            "sim_por_chunk":      [round(s, 4) for s in sims],
            "sim_media_simples":  round(float(np.mean(sims)), 4),
            "sim_media_ponderada": score,
            "tamanho_chunk":      tamanho_chunk,
            "passo_chunk":        passo_chunk,
            "custo_usd_estimado": 0.0,
        },
    )
