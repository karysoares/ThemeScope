# Análise dos Resultados do Experimento

Este documento resume a execução exibida no terminal para os 4 variantes:

- `exp1_embedding` (local)
- `exp2_zero_shot` (LLM local via Ollama)
- `exp2_few_shot` (LLM local via Ollama)
- `exp3_hibrido` (gate por embedding local + LLM condicional)

## Contexto da execução

- Tema: impactos do uso excessivo de redes sociais na saúde mental dos jovens
- Redações avaliadas: 6
- Provedor LLM: `ollama`
- Pausa entre chamadas: `5.0s`

## Resultado geral (métricas)

Ranking por MAE (menor é melhor):

1. `exp2_zero_shot` — MAE `0.0917`, acerto de faixa `100.0%`
2. `exp2_few_shot` — MAE `0.1083`, acerto de faixa `83.3%`
3. `exp1_embedding` — MAE `0.1437`, acerto de faixa `33.3%`
4. `exp3_hibrido` — MAE `0.1437`, acerto de faixa `33.3%`

## Leitura dos resultados

### 1) O melhor desempenho foi do `exp2_zero_shot`

- Obteve o menor erro médio absoluto.
- Acertou a faixa esperada em todas as 6 redações.
- Para este conjunto sintético, superou o few-shot.

### 2) O `exp2_few_shot` ficou bom, mas não melhor que zero-shot

- Teve MAE maior e menor taxa de acerto que o zero-shot.
- Indica que os exemplos few-shot atuais podem estar puxando a calibração para cima em alguns casos.

### 3) `exp1_embedding` e `exp3_hibrido` ficaram empatados

- Ambos com MAE `0.1437` e acerto de faixa `33.3%`.
- Isso ocorreu porque o gate do híbrido não acionou LLM em nenhuma redação (`LLM acionado: 0/6`).
- Na prática, o híbrido se comportou como o baseline de embedding nesta execução.

### 4) Ponto crítico do gate híbrido

- Scores de embedding ficaram majoritariamente fora da zona de incerteza definida (`0.35 < score < 0.65`).
- Com isso, o sistema decidiu `Emb` em todos os casos.
- Se o objetivo do híbrido é combinar custo baixo com ganho de qualidade, os limiares atuais precisam de calibração.

## Custo

- Todos os experimentos apareceram com custo estimado `US$ 0.000000`.
- Isso é coerente com a configuração atual:
  - embeddings locais no `exp1` e no gate do `exp3`;
  - custo de LLM local (Ollama) tratado como `0` no metadado.

## Tempo de execução observado

- `exp1` e `exp3` foram rápidos (tipicamente sub-segundo após carga inicial).
- `exp2` foi mais lento, com chamadas variando de ~7s a ~73s.
- A primeira redação foi mais lenta por aquecimento/download inicial de modelos locais.

## Conclusões práticas

- Para qualidade nesse dataset, `exp2_zero_shot` foi a melhor escolha.
- Para custo local/offline, `exp1` é o mais simples e rápido, porém menos preciso.
- O `exp3` ainda não entrega ganho sobre `exp1` com os limiares atuais.

## Próximos passos recomendados

1. Recalibrar `LIMIAR_BAIXO` e `LIMIAR_ALTO` do híbrido com base em validação (grid search simples).
2. Testar few-shot com exemplos mais curtos e mais próximos do dataset sintético.
3. Aumentar o conjunto de redações de validação para reduzir viés de amostra pequena (n=6).
4. Registrar métricas por classe (`alto`, `parcial`, `baixo`) para detectar onde cada experimento erra mais.
