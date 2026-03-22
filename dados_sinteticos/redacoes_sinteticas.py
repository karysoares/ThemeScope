"""
dados_sinteticos/redacoes_sinteticas.py
---------------------------------------
Base sintética mínima para comparação dos experimentos.
"""

from __future__ import annotations

from dataclasses import dataclass


TEMA_ID = "tema_redes_sociais_saude_mental"
TEMA_DESCRICAO = (
    "Discuta os impactos do uso excessivo de redes sociais na saúde mental dos jovens."
)


@dataclass(frozen=True)
class RedacaoSintetica:
    student_id: str
    text_id: str
    student_text: str
    alinhamento_esperado: str
    score_minimo_esperado: float
    score_maximo_esperado: float


TODAS_AS_REDACOES = [
    RedacaoSintetica(
        student_id="aluno_01",
        text_id="redacao_alta_01",
        student_text=(
            "O uso exagerado de redes sociais entre adolescentes tem ampliado casos de "
            "ansiedade e comparação social. Muitos jovens passam horas acompanhando "
            "padrões irreais de vida e beleza, o que reduz a autoestima. Além disso, "
            "o uso noturno do celular prejudica o sono e piora sintomas depressivos."
        ),
        alinhamento_esperado="alto",
        score_minimo_esperado=0.75,
        score_maximo_esperado=1.00,
    ),
    RedacaoSintetica(
        student_id="aluno_02",
        text_id="redacao_alta_02",
        student_text=(
            "As redes sociais podem gerar dependência comportamental em jovens. "
            "Quando usadas sem limite, afetam a concentração escolar e provocam "
            "sensação constante de inadequação. Programas de educação digital e "
            "apoio psicológico são importantes para proteger a saúde mental."
        ),
        alinhamento_esperado="alto",
        score_minimo_esperado=0.70,
        score_maximo_esperado=0.95,
    ),
    RedacaoSintetica(
        student_id="aluno_03",
        text_id="redacao_parcial_01",
        student_text=(
            "A tecnologia mudou a forma de estudar. Plataformas online ajudam no acesso "
            "ao conhecimento e também facilitam a comunicação entre colegas. Contudo, "
            "as redes sociais podem distrair e causar estresse quando não há equilíbrio."
        ),
        alinhamento_esperado="parcial",
        score_minimo_esperado=0.35,
        score_maximo_esperado=0.65,
    ),
    RedacaoSintetica(
        student_id="aluno_04",
        text_id="redacao_parcial_02",
        student_text=(
            "Os jovens utilizam internet para lazer e estudo. Esse uso pode ser positivo, "
            "mas o excesso de tempo em aplicativos sociais atrapalha a rotina. O problema "
            "é que o texto não aprofunda totalmente as consequências emocionais."
        ),
        alinhamento_esperado="parcial",
        score_minimo_esperado=0.30,
        score_maximo_esperado=0.60,
    ),
    RedacaoSintetica(
        student_id="aluno_05",
        text_id="redacao_baixa_01",
        student_text=(
            "As mudanças climáticas afetam todo o planeta. O desmatamento e a poluição "
            "exigem políticas públicas e investimento em energia renovável. A educação "
            "ambiental é essencial para reduzir danos futuros."
        ),
        alinhamento_esperado="baixo",
        score_minimo_esperado=0.00,
        score_maximo_esperado=0.20,
    ),
    RedacaoSintetica(
        student_id="aluno_06",
        text_id="redacao_baixa_02",
        student_text=(
            "A mobilidade urbana depende de transporte público eficiente. Cidades com "
            "planejamento adequado reduzem congestionamentos e melhoram a qualidade de "
            "vida da população. Investimentos em metrô e ciclovias são prioritários."
        ),
        alinhamento_esperado="baixo",
        score_minimo_esperado=0.00,
        score_maximo_esperado=0.20,
    ),
]
