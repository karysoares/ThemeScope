"""
Embeddings locais via sentence-transformers (um único ponto de carregamento do modelo).
Usado pelo Exp1 e pelo gate de embedding do Exp3 quando THEMESCOPE_LLM_PROVIDER=ollama.

O import de ``sentence_transformers`` é lazy para não quebrar importação da API/testes
quando o pacote não está instalado no ambiente.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODELO_EMBEDDING_LOCAL = "paraphrase-multilingual-MiniLM-L12-v2"

_modelo: Any = None


def _carregar_modelo() -> SentenceTransformer:
    global _modelo
    if _modelo is None:
        from sentence_transformers import SentenceTransformer as ST

        _modelo = ST(MODELO_EMBEDDING_LOCAL)
    return _modelo


def codificar_textos(textos: list[str]) -> list[np.ndarray]:
    """Retorna vetores L2-normalizados, dtype float32."""
    if not textos:
        return []
    modelo = _carregar_modelo()
    vetores = modelo.encode(
        textos,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [np.array(v, dtype=np.float32) for v in vetores]
