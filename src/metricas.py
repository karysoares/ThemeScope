"""
src/metricas.py
---------------
Módulo de avaliação comparativa entre os experimentos.

Calcula, para cada experimento e redação:
  - Erro absoluto vs. score esperado (média do intervalo [min, max])
  - Se a predição caiu dentro do intervalo esperado (acerto de faixa)
  - Erro médio absoluto (MAE) por experimento
  - Ranking de experimentos por MAE

Também compara cobertura de spans por redação.

Uso:
    from src.metricas import calcular_metricas, imprimir_relatorio
    resultado = calcular_metricas(resultados_dict)
    imprimir_relatorio(resultado)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any
# from src.experimentos.embedding import rodar_embedding
# from src.experimentos.llm import rodar_llm
# from src.experimentos.hibrido import rodar_hibrido

from .utils import RespostaAlinhamento


@dataclass
class MetricaExperimento:
    """Métricas de um experimento sobre todas as redações avaliadas."""
    nome: str
    scores: list[float]
    scores_esperados: list[float]
    erros_absolutos: list[float]
    acertos_de_faixa: list[bool]
    num_spans: list[int]

    @property
    def mae(self) -> float:
        if not self.erros_absolutos:
            return float("nan")
        return round(statistics.mean(self.erros_absolutos), 4)

    @property
    def taxa_acerto_faixa(self) -> float:
        if not self.acertos_de_faixa:
            return 0.0
        return round(
            sum(self.acertos_de_faixa) / len(self.acertos_de_faixa) * 100,
            1
        )

    @property
    def score_medio(self) -> float:
        return round(statistics.mean(self.scores), 4) if self.scores else float("nan")

    @property
    def spans_medio(self) -> float:
        return round(statistics.mean(self.num_spans), 2) if self.num_spans else 0.0

    @property
    def std(self) -> float:
        return (
            round(statistics.stdev(self.scores), 4)
            if len(self.scores) > 1
            else 0.0
        )


@dataclass
class ResultadoMetricas:
    """Resultado completo da comparação entre experimentos."""
    por_experimento: dict[str, MetricaExperimento]
    ranking_mae: list[str]
    resumo_global: dict[str, Any]
def calcular_metricas(
    resultados: dict[str, list[RespostaAlinhamento]],
    intervalos_esperados: list[tuple[float, float]],
) -> ResultadoMetricas:
    """
    Calcula métricas comparativas para cada experimento.
    """

    if not resultados:
        raise ValueError("Nenhum experimento fornecido")

    por_experimento: dict[str, MetricaExperimento] = {}

    for nome_exp, respostas in resultados.items():

        if len(respostas) != len(intervalos_esperados):
            raise ValueError(
                f"{nome_exp}: {len(respostas)} respostas != "
                f"{len(intervalos_esperados)} intervalos esperados"
            )

        scores: list[float] = []
        scores_esperados: list[float] = []
        erros_absolutos: list[float] = []
        acertos_de_faixa: list[bool] = []
        num_spans: list[int] = []

        for resp, (s_min, s_max) in zip(respostas, intervalos_esperados):

            if not 0 <= resp.alignment_score <= 1:
                raise ValueError(
                    f"{nome_exp}: alignment_score inválido "
                    f"{resp.alignment_score}"
                )

            centro = (s_min + s_max) / 2
            erro = abs(resp.alignment_score - centro)
            acerto = s_min <= resp.alignment_score <= s_max

            scores.append(resp.alignment_score)
            scores_esperados.append(centro)
            erros_absolutos.append(round(erro, 4))
            acertos_de_faixa.append(acerto)
            num_spans.append(len(resp.evidence_spans))

        por_experimento[nome_exp] = MetricaExperimento(
            nome=nome_exp,
            scores=scores,
            scores_esperados=scores_esperados,
            erros_absolutos=erros_absolutos,
            acertos_de_faixa=acertos_de_faixa,
            num_spans=num_spans,
        )

    # ranking mais inteligente: MAE primeiro, depois acerto
    ranking_mae = sorted(
        por_experimento.keys(),
        key=lambda k: (
            por_experimento[k].mae,
            -por_experimento[k].taxa_acerto_faixa,
        ),
    )

    melhor = por_experimento[ranking_mae[0]]

    mais_caro = max(
        resultados.keys(),
        key=lambda k: _custo_medio(resultados[k])
    )

    mais_barato = min(
        resultados.keys(),
        key=lambda k: _custo_medio(resultados[k])
    )

    resumo_global = {
        "num_redacoes": len(intervalos_esperados),
        "num_experimentos": len(resultados),
        "melhor_mae": {
            "experimento": melhor.nome,
            "mae": melhor.mae
        },
        "ranking_mae": ranking_mae,
        "mais_barato": mais_barato,
        "mais_caro": mais_caro,
    }

    return ResultadoMetricas(
        por_experimento=por_experimento,
        ranking_mae=ranking_mae,
        resumo_global=resumo_global,
    )

def _custo_medio(respostas: list[RespostaAlinhamento]) -> float:
    custos = [
        (r.metadados or {}).get("custo_usd_estimado", 0.0)
        for r in respostas
    ]
    return statistics.mean(custos) if custos else 0.0


SEP = "─" * 70


def imprimir_relatorio(resultado: ResultadoMetricas) -> None:

    print(f"\n{'=' * 70}")
    print("  RELATÓRIO DE MÉTRICAS — Comparação entre Experimentos")
    print(f"{'=' * 70}")

    print(f"\n🏆 Melhor experimento: {resultado.ranking_mae[0]}\n")

    print(
        f"  {'Experimento':<28} {'MAE':>6}  {'Acerto faixa':>13}  "
        f"{'Score médio':>12}  {'Spans/req':>9}"
    )

    print(f"  {SEP}")

    for nome in resultado.ranking_mae:
        m = resultado.por_experimento[nome]

        print(
            f"  {nome:<28} {m.mae:>6.4f}  "
            f"{m.taxa_acerto_faixa:>12.1f}%  "
            f"{m.score_medio:>12.4f}  "
            f"{m.spans_medio:>9.1f}"
        )

    print(f"\n  {'─' * 70}")
    print("  ERRO ABSOLUTO POR REDAÇÃO")
    print(f"  {'─' * 70}")

    primeiro = next(iter(resultado.por_experimento.values()))
    n = len(primeiro.scores)

    nomes_exp = list(resultado.por_experimento.keys())
    cabecalho_cols = "".join(f"  {ne[:8]:>9}" for ne in nomes_exp)

    print(f"  {'Redação / Esperado':<26}{cabecalho_cols}  {'Faixa ok?':>12}")
    print(f"  {'─' * 70}")

    for i in range(n):

        erros_str = "".join(
            f"  {resultado.por_experimento[ne].erros_absolutos[i]:>9.4f}"
            for ne in nomes_exp
        )

        acertos = [
            resultado.por_experimento[ne].acertos_de_faixa[i]
            for ne in nomes_exp
        ]

        acerto_str = "  " + " ".join("✓" if a else "✗" for a in acertos)
        centro = primeiro.scores_esperados[i]

        print(
            f"  {'redação ' + str(i+1) + f' (ctr={centro:.2f})':<26}"
            f"{erros_str}{acerto_str}"
        )

    print(f"\n  {'─' * 70}")
    print("  RESUMO GLOBAL")
    print(f"  {'─' * 70}")

    g = resultado.resumo_global

    print(f"  Redações avaliadas:  {g['num_redacoes']}")
    print(f"  Experimentos:        {g['num_experimentos']}")
    print(
        f"  Melhor MAE:          {g['melhor_mae']['experimento']} "
        f"(MAE={g['melhor_mae']['mae']:.4f})"
    )
    print(f"  Ranking por MAE:     {' > '.join(g['ranking_mae'])}")
    print(f"  Mais barato:         {g['mais_barato']}")
    print(f"  Mais caro:           {g['mais_caro']}")
    print(f"  {'=' * 70}\n")

if __name__ == "__main__":

    from src.experimentos import (
        exp1_embedding_baseline as exp1,
        exp2_prompt_engineering as exp2,
        exp3_hibrido as exp3,
    )

    tema = "Os impactos do uso excessivo de redes sociais na saúde mental dos jovens"

    redacoes = [
        ("1", "O uso intenso de redes sociais causa ansiedade entre jovens."),
        ("2", "A tecnologia digital transformou a educação moderna."),
        ("3", "As mudanças climáticas afetam o planeta.")
    ]

    resultados = {
        "embedding": [],
        "prompt": [],
        "hibrido": [],
    }

    for text_id, texto in redacoes:

        resultados["embedding"].append(
            exp1.avaliar(
                student_id="1",
                text_id=text_id,
                student_text=texto,
                theme_id="tema",
                theme_description=tema,
                chave_api=""
            )
        )

        resultados["prompt"].append(
            exp2.avaliar(
                student_id="1",
                text_id=text_id,
                student_text=texto,
                theme_id="tema",
                theme_description=tema,
                chave_api=""
            )
        )

        resultados["hibrido"].append(
            exp3.avaliar(
                student_id="1",
                text_id=text_id,
                student_text=texto,
                theme_id="tema",
                theme_description=tema,
                chave_api=""
            )
        )

    intervalos = [
        (0.8, 1.0),
        (0.3, 0.6),
        (0.0, 0.2),
    ]

    resultado = calcular_metricas(resultados, intervalos)

    imprimir_relatorio(resultado)