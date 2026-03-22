"""
Utilitários compartilhados entre os três experimentos:
  - Cliente HTTP para a API OpenAI (embeddings e chat)
  - Chunking de texto por sentença
  - Similaridade cosseno (numpy puro, sem dependências extras)
  - Contrato de saída padronizado RespostaAlinhamento
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import numpy as np
import requests

OPENAI_URL_CHAT       = "https://api.openai.com/v1/chat/completions"
OPENAI_URL_EMBEDDINGS = "https://api.openai.com/v1/embeddings"
OLLAMA_URL_CHAT       = os.environ.get("THEMESCOPE_OLLAMA_URL", "http://localhost:11434/api/chat")
MODELO_CHAT           = "gpt-4o"
MODELO_CHAT_LOCAL     = os.environ.get("THEMESCOPE_OLLAMA_MODEL", "llama3.1:8b")
MODELO_EMBEDDING      = "text-embedding-3-small"   # 1.536 dimensões, ~US$0,02/1M tokens
PROVEDOR_LLM          = os.environ.get("THEMESCOPE_LLM_PROVIDER", "ollama").lower()
TEMPERATURA           = 0        
MAX_TOKENS_CHAT       = 1024
MAX_TENTATIVAS        = int(os.environ.get("THEMESCOPE_MAX_TENTATIVAS", "6"))
DELAY_BASE_S          = float(os.environ.get("THEMESCOPE_DELAY_BASE_S", "2"))
DELAY_MAX_S           = float(os.environ.get("THEMESCOPE_DELAY_MAX_S", "90"))
CUSTO_LLM_POR_REQ_USD = 0.006 # Custo estimado por requisição LLM (para relatórios)
CUSTO_EMBEDDING_POR_TOKEN_USD = 0.00000002 # Custo estimado por token de embedding (text-embedding-3-small)

@dataclass
class SpanEvidencia:
    """Trecho do texto com posição e rótulo."""
    start_char: int
    end_char:   int
    span_text:  str
    label:      str   # "alinhado" | "fora_do_tema" | "parcial"

    def para_dict(self) -> dict[str, Any]:
        return {
            "start_char": self.start_char,
            "end_char":   self.end_char,
            "span_text":  self.span_text,
            "label":      self.label,
        }


@dataclass
class RespostaAlinhamento:
    """Contrato de saída completo."""
    student_id:     str | int
    text_id:        str | int
    theme_id:       str | int
    alignment_score: float
    evidence_spans: list[SpanEvidencia] = field(default_factory=list)
    model_version:  str = "desconhecido"
    metadados:      dict[str, Any] = field(default_factory=dict)

    def para_dict(self) -> dict[str, Any]:
        return {
            "student_id":     self.student_id,
            "text_id":        self.text_id,
            "theme_id":       self.theme_id,
            "alignment_score": round(float(self.alignment_score), 4),
            "evidence_spans": [s.para_dict() for s in self.evidence_spans],
            "model_version":  self.model_version,
            "metadados":      self.metadados,
        }

def _cabecalhos_openai(chave_api: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {chave_api}",
        "Content-Type":  "application/json",
    }


def _sleep_backoff(tentativa: int, resposta: requests.Response | None = None) -> None:
    """
    Aguarda com backoff exponencial + jitter, respeitando Retry-After quando disponível.
    """
    retry_after = 0.0
    if resposta is not None:
        try:
            retry_after = float(resposta.headers.get("Retry-After", "0").strip() or "0")
        except ValueError:
            retry_after = 0.0

    backoff = min(DELAY_BASE_S ** tentativa, DELAY_MAX_S)
    if resposta is not None and resposta.status_code == 429:
        backoff = max(backoff, min(15 * tentativa, DELAY_MAX_S))
    jitter = random.uniform(0, 0.5)
    espera = max(retry_after, backoff + jitter)
    time.sleep(espera)

_R = TypeVar('_R')

def _chamar_api_com_retry(
    api_call_func: Callable[[], _R],
    api_name: str,
    max_tentativas: int = MAX_TENTATIVAS,
) -> _R:
    """
    Função genérica para chamar APIs com retry e backoff.
    """
    ultimo_erro: Exception | None = None
    ultima_resposta: requests.Response | None = None
    for tentativa in range(1, max_tentativas + 1):
        try:
            return api_call_func()
        except requests.HTTPError as exc:
            ultimo_erro = exc
            ultima_resposta = exc.response
            if tentativa < max_tentativas:
                _sleep_backoff(tentativa, ultima_resposta)
        except Exception as exc:
            ultimo_erro = exc
            if tentativa < max_tentativas:
                _sleep_backoff(tentativa)
    raise RuntimeError(f"{api_name} falhou após {max_tentativas} tentativas: {ultimo_erro}")


def _chamar_openai_chat(
    mensagens: list[dict[str, str]],
    chave_api: str,
    temperatura: float,
    max_tokens: int,
) -> str:
    """Chama a API OpenAI Chat Completions."""
    payload = {
        "model":       MODELO_CHAT,
        "temperature": temperatura,
        "max_tokens":  max_tokens,
        "messages":    mensagens,
    }
    def _call():
        r = requests.post(
            OPENAI_URL_CHAT,
            headers=_cabecalhos_openai(chave_api),
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    return _chamar_api_com_retry(_call, "OpenAI Chat API")


def _chamar_ollama_chat(
    mensagens: list[dict[str, str]],
    temperatura: float,
    max_tokens: int,
) -> str:
    """Chama a API Ollama Chat."""
    payload = {
        "model": MODELO_CHAT_LOCAL,
        "messages": mensagens,
        "stream": False,
        "options": {
            "temperature": temperatura,
            "num_predict": max_tokens,
        },
    }
    def _call():
        r = requests.post(
            OLLAMA_URL_CHAT,
            json=payload,
            timeout=90,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
    return _chamar_api_com_retry(_call, "Ollama Chat API")


def chamar_chat(
    mensagens: list[dict[str, str]],
    chave_api: str,
    temperatura: float = TEMPERATURA,
    max_tokens: int = MAX_TOKENS_CHAT,
) -> str:
    """
    Chama a API de Chat Completions do provedor configurado.

    Args:
        mensagens:  Lista de dicts com 'role' e 'content'.
        chave_api:  Chave de API da OpenAI (ignorada se PROVEDOR_LLM for 'ollama').
        temperatura: Temperatura de amostragem (0 = determinístico).
        max_tokens: Limite de tokens na resposta.

    Returns:
        Texto bruto retornado pelo modelo.

    Raises:
        RuntimeError: Se todas as tentativas falharem.
    """
    if PROVEDOR_LLM == "ollama":
        return _chamar_ollama_chat(mensagens, temperatura, max_tokens)
    return _chamar_openai_chat(mensagens, chave_api, temperatura, max_tokens)


def obter_embedding_openai(texto: str, chave_api: str) -> np.ndarray:
    """
    Obtém o embedding de um texto via OpenAI text-embedding-3-small.

    Args:
        texto:     Texto a ser embeddado.
        chave_api: Chave de API da OpenAI.

    Returns:
        Array numpy de shape (1536,) normalizado (L2).

    Raises:
        RuntimeError: Se todas as tentativas falharem.
    """
    payload = {"model": MODELO_EMBEDDING, "input": texto}
    def _call():
        r = requests.post(
            OPENAI_URL_EMBEDDINGS,
            headers=_cabecalhos_openai(chave_api),
            json=payload,
            timeout=20,
        )
        r.raise_for_status()
        vetor = np.array(r.json()["data"][0]["embedding"], dtype=np.float32)
        return vetor
    return _chamar_api_com_retry(_call, "OpenAI Embedding API")


def obter_embeddings_batch_openai(
    textos: list[str], chave_api: str
) -> list[np.ndarray]:
    """
    Obtém embeddings de múltiplos textos em uma única chamada à API OpenAI.

    Args:
        textos:    Lista de textos.
        chave_api: Chave de API da OpenAI.

    Returns:
        Lista de arrays numpy, um por texto, na mesma ordem.
    """
    payload = {"model": MODELO_EMBEDDING, "input": textos}
    def _call():
        r = requests.post(
            OPENAI_URL_EMBEDDINGS,
            headers=_cabecalhos_openai(chave_api),
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        dados = r.json()["data"]
        dados_ordenados = sorted(dados, key=lambda x: x["index"])
        return [
            np.array(item["embedding"], dtype=np.float32)
            for item in dados_ordenados
        ]
    return _chamar_api_com_retry(_call, "OpenAI Embedding Batch API")


def obter_embeddings(textos: list[str], chave_api: str) -> list[np.ndarray]:
    """
    Obtém embeddings para uma lista de textos.
    Atualmente, apenas OpenAI é suportado para embeddings.
    """
    if PROVEDOR_LLM == "ollama":
        raise RuntimeError("Embeddings não disponíveis com provedor 'ollama'. Use THEMESCOPE_LLM_PROVIDER=openai.")

    if len(textos) == 1:
        return [obter_embedding_openai(textos[0], chave_api)]

    return obter_embeddings_batch_openai(textos, chave_api)

def similaridade_cosseno(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcula a similaridade cosseno entre dois vetores.

    text-embedding-3 já retorna vetores L2-normalizados, portanto
    similaridade_cosseno = produto_interno. Mantemos o cálculo explícito
    para robustez caso outro modelo seja usado no futuro.

    Args:
        a, b: Arrays numpy 1-D.

    Returns:
        Float em [-1, 1]. Valores típicos de embeddings semânticos: [0, 1].
    """
    norma_a = np.linalg.norm(a)
    norma_b = np.linalg.norm(b)
    if norma_a < 1e-10 or norma_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norma_a * norma_b))

def chunkar_por_sentenca(
    texto: str,
    tamanho_chunk: int = 3,
    passo: int = 1,
) -> list[str]:
    """
    Divide o texto em chunks de N sentenças com passo configurável (sliding window).

    Args:
        texto:         Texto completo da redação.
        tamanho_chunk: Número de sentenças por chunk.
        passo:         Avanço em sentenças entre chunks consecutivos.

    Returns:
        Lista de strings, cada uma representando um chunk.

    Example:
        >>> chunkar_por_sentenca("A. B. C. D.", tamanho_chunk=2, passo=1)
        ['A. B.', 'B. C.', 'C. D.']
    """
    # Regex para separar sentenças em português (preserva ponto final)
    sentencas = re.split(r'(?<=[\.!?])\s+', texto.strip())
    sentencas = [s.strip() for s in sentencas if s.strip()]

    if not sentencas:
        return [texto] # Retorna o texto original se não houver sentenças válidas

    if len(sentencas) <= tamanho_chunk:
        return [" ".join(sentencas)] # Retorna o texto completo como um único chunk

    chunks = []
    for i in range(0, len(sentencas) - tamanho_chunk + 1, passo):
        chunk = " ".join(sentencas[i: i + tamanho_chunk])
        chunks.append(chunk)

    return chunks if chunks else [" ".join(sentencas)] # Garante que sempre retorna algo

def ancorar_spans(
    texto_original: str,
    spans_brutos: list[dict[str, str]],
) -> list[SpanEvidencia]:
    """
    Localiza cada span_text dentro de texto_original para obter start_char/end_char.
    Spans não localizáveis são descartados, com um aviso.

    Args:
        texto_original: Redação completa do estudante.
        spans_brutos:   Lista de dicts com 'span_text' e 'label'.

    Returns:
        Lista de SpanEvidencia com offsets de caractere resolvidos.
    """
    anchorados: list[SpanEvidencia] = []
    inicio_busca = 0

    for bruto in spans_brutos:
        span_text = bruto.get("span_text", "").strip()
        label     = bruto.get("label", "parcial")

        if not span_text:
            continue

        idx = texto_original.find(span_text, inicio_busca)

        if idx == -1:
            idx = texto_original.lower().find(span_text.lower(), inicio_busca)

        if idx == -1:
            idx = texto_original.lower().find(span_text.lower())

        if idx == -1:
            # print(f"WARNING: Span '{span_text}' não encontrado no texto original. Descartando.") # Para debug
            continue

        fim = idx + len(span_text)
        anchorados.append(SpanEvidencia(
            start_char=idx,
            end_char=fim,
            span_text=texto_original[idx:fim],
            label=label,
        ))
        inicio_busca = max(inicio_busca, fim)

    return anchorados

def parsear_json_llm(texto_bruto: str) -> dict[str, Any]:
    """
    Parseia resposta JSON do LLM, removendo fences de markdown se presentes.

    Args:
        texto_bruto: String retornada pelo modelo.

    Returns:
        Dict parseado.

    Raises:
        ValueError: Se a string não for JSON válido.
    """
    limpo = re.sub(r"^```(?:json)?\s*", "", texto_bruto.strip(), flags=re.IGNORECASE)
    limpo = re.sub(r"\s*```$", "", limpo)
    try:
        return json.loads(limpo)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Resposta não-JSON do modelo: {texto_bruto!r}") from exc
