from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, ConfigDict, Field

from src.experimentos import exp1_embedding_baseline as exp1
from src.experimentos import exp2_prompt_engineering as exp2
from src.experimentos import exp3_hibrido as exp3
from src.utils import PROVEDOR_LLM


class AvaliacaoRequest(BaseModel):
    student_id: str | int
    text_id: str | int
    student_text: str = Field(min_length=1)
    theme_id: str | int
    theme_description: str = Field(min_length=1)
    experimento: Literal["exp1_embedding", "exp2_prompt", "exp3_hibrido"] = "exp3_hibrido"
    modo_prompt: Literal["zero_shot", "few_shot"] = "few_shot"


class AvaliacaoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    student_id: str | int
    text_id: str | int
    theme_id: str | int
    alignment_score: float
    evidence_spans: list[dict[str, Any]]
    model_version: str
    metadados: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(
    title="ThemeScope API",
    description="API para avaliação de alinhamento temático de redações.",
    version="1.0.0",
)


def _resolver_chave_api(authorization: str | None) -> str:
    """
    Resolve a chave de API a partir do header Authorization ou variável de ambiente.
    Espera o formato "Bearer SEU_TOKEN_AQUI".
    """
    if authorization and authorization.startswith("Bearer "):
        return authorization.split(" ")[1]

    chave_env = os.getenv("OPENAI_API_KEY")
    if chave_env:
        return chave_env

    raise HTTPException(
        status_code=401, # Unauthorized
        detail=(
            "Chave de API não fornecida. Por favor, inclua um header 'Authorization: Bearer SEU_TOKEN_AQUI' "
            "ou defina a variável de ambiente OPENAI_API_KEY."
        ),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/avaliar", response_model=AvaliacaoResponse)
def avaliar(
    payload: AvaliacaoRequest,
    authorization: str | None = Header(default=None, alias="Authorization")
) -> dict[str, Any]:

    chave_api = ""
    if PROVEDOR_LLM == "openai" and payload.experimento in {"exp2_prompt", "exp3_hibrido", "exp1_embedding"}:
        chave_api = _resolver_chave_api(authorization)

    kwargs = dict(
        student_id=payload.student_id,
        text_id=payload.text_id,
        student_text=payload.student_text,
        theme_id=payload.theme_id,
        theme_description=payload.theme_description,
        chave_api=chave_api, 
    )

    try:
        if payload.experimento == "exp1_embedding":
            resultado = exp1.avaliar(**kwargs)
        elif payload.experimento == "exp2_prompt":
            resultado = exp2.avaliar(**kwargs, modo=payload.modo_prompt)
        else: # exp3_hibrido
            resultado = exp3.avaliar(**kwargs, modo_prompt=payload.modo_prompt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=f"Falha ao consultar modelo: {exc}") from exc
    except Exception as exc: # pragma: no cover - fallback defensivo
        raise HTTPException(status_code=500, detail=f"Erro interno inesperado: {exc}") from exc

    return resultado.para_dict()


if __name__ == "__main__":
    import uvicorn
    print("Iniciando ThemeScope API...")
    print(f"Provedor LLM configurado: {PROVEDOR_LLM}")
    print("Acesse http://127.0.0.1:8000/docs para a documentação interativa.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
