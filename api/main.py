from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from src.experimentos import exp1_embedding_baseline as exp1
from src.experimentos import exp2_prompt_engineering as exp2
from src.experimentos import exp3_hibrido as exp3
from src.utils import provedor_llm_atual

logger = logging.getLogger(__name__)

_MAX_STUDENT_TEXT = 80_000
_MAX_THEME_TEXT = 20_000


class AvaliacaoRequest(BaseModel):
    student_id: str | int
    text_id: str | int
    student_text: str = Field(min_length=1, max_length=_MAX_STUDENT_TEXT)
    theme_id: str | int
    theme_description: str = Field(min_length=1, max_length=_MAX_THEME_TEXT)
    experimento: Literal["exp1_embedding", "exp2_prompt", "exp3_hibrido"] = "exp1_embedding"
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


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


def _precisa_chave_openai(experimento: str) -> bool:
    """Exp1 é só embedding local; Exp2/Exp3 podem chamar APIs pagas quando o provedor é OpenAI."""
    if provedor_llm_atual() != "openai":
        return False
    return experimento in {"exp2_prompt", "exp3_hibrido"}


def _resolver_chave_api(authorization: str | None) -> str:
    """
    Resolve a chave a partir do header Authorization ou OPENAI_API_KEY.
    Formato esperado: ``Bearer <token>``.
    """
    if authorization and authorization.startswith("Bearer "):
        return authorization.split(" ", 1)[1].strip()

    chave_env = os.getenv("OPENAI_API_KEY")
    if chave_env:
        return chave_env

    raise HTTPException(
        status_code=401,
        detail=(
            "Chave de API não fornecida. Inclua 'Authorization: Bearer <token>' "
            "ou defina OPENAI_API_KEY."
        ),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/avaliar", response_model=AvaliacaoResponse)
def avaliar(
    request: Request,
    payload: AvaliacaoRequest,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> dict[str, Any]:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    chave_api = ""
    if _precisa_chave_openai(payload.experimento):
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
        else:
            resultado = exp3.avaliar(**kwargs, modo_prompt=payload.modo_prompt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail={"message": "Falha ao consultar modelo.", "request_id": request_id},
        ) from exc
    except Exception as exc:
        logger.exception("Erro não tratado em /avaliar (request_id=%s)", request_id)
        raise HTTPException(
            status_code=500,
            detail={"message": "Erro interno.", "request_id": request_id},
        ) from exc

    return resultado.para_dict()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    print("Iniciando ThemeScope API...")
    print(f"Provedor LLM configurado: {provedor_llm_atual()}")
    print("Acesse http://127.0.0.1:8000/docs para a documentação interativa.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
