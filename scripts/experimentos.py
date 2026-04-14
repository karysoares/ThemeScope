"""
executar_experimentos.py
------------------------
Runner principal: avalia todas as redações sintéticas nos três experimentos,
salva resultados em JSON e imprime tabela comparativa + relatório de métricas.

Execução:
    export OPENAI_API_KEY
    python executar_experimentos.py

Saídas geradas em ./resultados/:
    resultados_<timestamp>.json   — outputs brutos de todos os experimentos
    metricas_<timestamp>.json     — MAE, acerto de faixa, ranking
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dados_sinteticos.redacoes_sinteticas import (
    TODAS_AS_REDACOES,
    TEMA_DESCRICAO,
    TEMA_ID,
)
from src.experimentos import exp1_embedding_baseline as exp1
from src.experimentos import exp2_prompt_engineering as exp2
from src.experimentos import exp3_hibrido as exp3
from src.metricas import calcular_metricas, imprimir_relatorio
from src.utils import RespostaAlinhamento, provedor_llm_atual

W = 70
PAUSA_PADRAO_S = 2.0
PAUSA_MODO_ECONOMICO_S = 2.0


def sep(char="─"):
    print(char * W)


def titulo(txt):
    print(f"\n{'=' * W}\n  {txt}\n{'=' * W}")


def _pausa_entre_chamadas(segundos: float) -> None:
    if segundos > 0:
        time.sleep(segundos)

def salvar_resultados(
    todos: dict[str, list[RespostaAlinhamento]],
    pasta: str = "resultados",
) -> Path:
    """
    Persiste os outputs brutos de todos os experimentos em JSON.

    Args:
        todos: Dict {nome_experimento: [RespostaAlinhamento, ...]}.
        pasta: Diretório de saída (criado se não existir).

    Returns:
        Caminho do arquivo gerado.
    """
    Path(pasta).mkdir(exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = Path(pasta) / f"resultados_{ts}.json"
    payload = {nome: [r.para_dict() for r in rs] for nome, rs in todos.items()}
    caminho.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return caminho


def salvar_metricas(metricas: dict, pasta: str = "resultados") -> Path:
    """Persiste o relatório de métricas em JSON."""
    Path(pasta).mkdir(exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    caminho = Path(pasta) / f"metricas_{ts}.json"
    caminho.write_text(json.dumps(metricas, ensure_ascii=False, indent=2))
    return caminho

def avaliar_redacao(
    redacao, chave_api: str, pausa_s: float
) -> dict[str, RespostaAlinhamento]:

    kwargs = dict(
        student_id        = redacao.student_id,
        text_id           = redacao.text_id,
        student_text      = redacao.student_text,
        theme_id          = TEMA_ID,
        theme_description = TEMA_DESCRICAO,
        chave_api         = chave_api,
    )

    resultados: dict[str, RespostaAlinhamento] = {}

    print("    [Exp 1]  embedding...    ", end="", flush=True)

    t0 = time.perf_counter()
    resultados["exp1_embedding"] = exp1.avaliar(**kwargs)
    dt = time.perf_counter() - t0

    r = resultados["exp1_embedding"]

    print(
        f"score={r.alignment_score:.3f}  "
        f"chunks={r.metadados['num_chunks']}  "
        f"({dt:.1f}s)"
    )

    _pausa_entre_chamadas(pausa_s)

    print("    [Exp 2a] zero-shot...   ", end="", flush=True)

    t0 = time.perf_counter()
    resultados["exp2_zero_shot"] = exp2.avaliar(**kwargs, modo="zero_shot")
    dt = time.perf_counter() - t0

    r = resultados["exp2_zero_shot"]

    print(
        f"score={r.alignment_score:.3f}  "
        f"tokens_in≈{r.metadados['tokens_entrada_estimados']}  "
        f"({dt:.1f}s)"
    )

    _pausa_entre_chamadas(pausa_s)

    print("    [Exp 2b] few-shot...    ", end="", flush=True)

    t0 = time.perf_counter()
    resultados["exp2_few_shot"] = exp2.avaliar(**kwargs, modo="few_shot")
    dt = time.perf_counter() - t0

    r = resultados["exp2_few_shot"]

    print(
        f"score={r.alignment_score:.3f}  "
        f"tokens_in≈{r.metadados['tokens_entrada_estimados']}  "
        f"({dt:.1f}s)"
    )

    _pausa_entre_chamadas(pausa_s)

    print("    [Exp 3]  híbrido...     ", end="", flush=True)

    t0 = time.perf_counter()
    resultados["exp3_hibrido"] = exp3.avaliar(**kwargs)
    dt = time.perf_counter() - t0

    r = resultados["exp3_hibrido"]

    llm_tag = "LLM" if r.metadados.get("llm_acionado") else "Emb"

    print(
        f"score={r.alignment_score:.3f}  "
        f"gate={llm_tag}  "
        f"emb={r.metadados.get('score_embedding', 0):.3f}  "
        f"({dt:.1f}s)"
    )

    for nome, r in resultados.items():

        print(
            f"      → {nome:<16} "
            f"score={r.alignment_score:.3f} "
            f"model={r.model_version}"
        )

    return resultados

def imprimir_tabela_scores(
    acumulado: dict[str, list[RespostaAlinhamento]]
) -> None:
    experimentos = list(acumulado.keys())

    titulo("TABELA COMPARATIVA DE SCORES")
    cab = f"{'Redação':<24}  {'Nível':<8}  {'Esperado':>10}"
    for e in experimentos:
        cab += f"  {e[:12]:>12}"
    print(f"  {cab}")
    sep("·")

    for i, red in enumerate(TODAS_AS_REDACOES):
        faixa = f"[{red.score_minimo_esperado:.2f},{red.score_maximo_esperado:.2f}]"
        linha = f"  {red.text_id:<24}  {red.alinhamento_esperado:<8}  {faixa:>10}"
        for e in experimentos:
            score = acumulado[e][i].alignment_score
            ok    = red.score_minimo_esperado <= score <= red.score_maximo_esperado
            marca = " " if ok else "*"
            linha += f"  {score:>11.3f}{marca}"
        print(linha)

    print("\n  (* = score fora do intervalo esperado)")

def imprimir_tabela_custo(
    acumulado: dict[str, list[RespostaAlinhamento]]
) -> None:
    titulo("ANÁLISE DE CUSTO ESTIMADO (US$)")

    custos_fs = [
        r.metadados.get("custo_usd_estimado", 0.0)
        for r in acumulado.get("exp2_few_shot", [])
    ]
    ref = (sum(custos_fs) / len(custos_fs)) if custos_fs else 0.007

    print(f"  {'Experimento':<28}  {'Custo médio/req':>16}  "
          f"{'Total (6 req)':>14}  {'vs. few-shot':>13}")
    sep("·")

    for nome, respostas in acumulado.items():
        custos = [r.metadados.get("custo_usd_estimado", 0.0) for r in respostas]
        medio  = sum(custos) / len(custos) if custos else 0.0
        total  = sum(custos)
        pct    = (medio / ref * 100) if ref else 0.0
        print(f"  {nome:<28}  US${medio:>12.6f}  US${total:>10.6f}  "
              f"{pct:>11.1f}%")

    hibridos = acumulado.get("exp3_hibrido", [])
    if hibridos:
        n_llm = sum(1 for r in hibridos if r.metadados.get("llm_acionado"))
        print(f"\n  Gate híbrido — LLM acionado: {n_llm}/{len(hibridos)} "
              f"({n_llm / len(hibridos) * 100:.0f}%)")
        for r in hibridos:
            tag   = "LLM" if r.metadados["llm_acionado"] else "Emb"
            delta = r.metadados.get("delta_emb_vs_llm", "n/a")
            d_str = f"{delta:.3f}" if isinstance(delta, float) else str(delta)
            print(f"    {r.text_id:<28}  gate={tag}  "
                  f"emb={r.metadados['score_embedding']:.3f}  "
                  f"Δ(emb↔llm)={d_str}")


def main() -> None:
    chave = os.environ.get("OPENAI_API_KEY", "")
    if provedor_llm_atual() == "openai" and not chave:
        print("Erro: defina OPENAI_API_KEY antes de executar com provedor OpenAI.")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    modo_economico = "--modo-economico" in sys.argv
    pausa_s = float(os.environ.get("THEMESCOPE_PAUSA_S", PAUSA_PADRAO_S))
    if modo_economico and "THEMESCOPE_PAUSA_S" not in os.environ:
        pausa_s = PAUSA_MODO_ECONOMICO_S

    titulo(" Comparação dos Experimentos de Alinhamento Temático")
    print(f"  Tema:        {TEMA_DESCRICAO}")
    print(f"  Redações:    {len(TODAS_AS_REDACOES)}")
    print(f"  Variantes:   Exp1·Embedding | Exp2a·ZeroShot | "
          f"Exp2b·FewShot | Exp3·Híbrido")
    print(f"  Provedor LLM: {provedor_llm_atual()}")
    print(f"  Pausa API:   {pausa_s:.1f}s entre chamadas")
    if modo_economico:
        print("  Modo:        econômico (throttling ativo)")

    acumulado: dict[str, list[RespostaAlinhamento]] = {
        "exp1_embedding":  [],
        "exp2_zero_shot":  [],
        "exp2_few_shot":   [],
        "exp3_hibrido":    [],
    }

    for red in TODAS_AS_REDACOES:
        sep()
        print(f"  {red.text_id}  [{red.alinhamento_esperado.upper()}]  "
              f"esperado=[{red.score_minimo_esperado:.2f},{red.score_maximo_esperado:.2f}]")
        sep("·")
        resultados_redacao = avaliar_redacao(red, chave, pausa_s)
        for nome, resp in resultados_redacao.items():
            acumulado[nome].append(resp)
        _pausa_entre_chamadas(pausa_s)

    imprimir_tabela_scores(acumulado)
    imprimir_tabela_custo(acumulado)

    intervalos = [
        (red.score_minimo_esperado, red.score_maximo_esperado)
        for red in TODAS_AS_REDACOES
    ]
    res_metricas = calcular_metricas(acumulado, intervalos)
    imprimir_relatorio(res_metricas)

    p_resultados = salvar_resultados(acumulado)
    metricas_ser = {
        nome: {
            "mae":               m.mae,
            "taxa_acerto_faixa": m.taxa_acerto_faixa,
            "score_medio":       m.score_medio,
            "spans_medio":       m.spans_medio,
            "erros_absolutos":   m.erros_absolutos,
            "acertos_de_faixa":  m.acertos_de_faixa,
        }
        for nome, m in res_metricas.por_experimento.items()
    }
    metricas_ser["resumo_global"] = res_metricas.resumo_global
    p_metricas = salvar_metricas(metricas_ser)

    sep("=")
    print(f"  Resultados salvos em: {p_resultados}")
    print(f"  Métricas salvas em:   {p_metricas}")
    sep("=")


if __name__ == "__main__":
    main()
