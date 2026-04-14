# ThemeScope — Documentação Técnica Completa

## 1. Visão Geral

O ThemeScope é um sistema de avaliação automática de alinhamento temático de redações, projetado para analisar o grau de aderência entre um texto produzido por um estudante e o tema proposto. O sistema utiliza uma arquitetura modular baseada em três abordagens complementares: similaridade semântica via embeddings, avaliação direta por modelos de linguagem (LLM) e uma estratégia híbrida que combina ambas as técnicas com um mecanismo de gating.

O objetivo principal do projeto é construir um avaliador robusto, interpretável e escalável, capaz de operar em ambiente de produção com controle de custo e latência, mantendo alta qualidade na estimativa do alinhamento temático.

---

# 2. Arquitetura Geral

O fluxo completo do sistema segue o pipeline abaixo:

1. Recebimento da requisição pela API
2. Validação do payload
3. Seleção do experimento
4. Execução do avaliador
5. Cálculo do alignment score
6. Extração de evidências
7. Retorno estruturado em JSON

Representação:

```
Request
  ↓
API FastAPI
  ↓
Selecionar experimento
  ↓
Exp1 | Exp2 | Exp3
  ↓
Score + spans
  ↓
Response JSON
```

A arquitetura é modular, separando:

* lógica de inferência
* experimentos
* API
* métricas
* utilitários

Essa separação permite benchmarking, substituição de modelos e extensibilidade.

---

# 3. Componentes do Sistema

### 3.1. Avaliador legado (`avaliador_alinhamento.py`)
Fachada sobre o **Experimento 2** em modo **zero-shot** (`exp2_prompt_engineering`), reutilizando `chamar_chat`, parsing JSON e ancoragem de spans em `utils`. Mantém `RequisicaoAlinhamento` para compatibilidade; o contrato de saída é `RespostaAlinhamento` de `utils`.

Com `THEMESCOPE_LLM_PROVIDER=openai`, usa GPT-4o; com `ollama`, o modelo local configurado em `THEMESCOPE_OLLAMA_MODEL`. O prompt segue schema JSON obrigatório (alinhado ao Exp2).

O modelo retorna:

* alignment_score
* raciocínio
* evidence_spans

O alignment_score é normalizado para o intervalo [0,1], garantindo consistência na avaliação. Os spans retornados pelo modelo são ancorados no texto original via busca de substring, permitindo identificar exatamente quais trechos justificam o score.

Essa estratégia melhora a interpretabilidade e permite downstream analytics.

---

### 3.2. Experimento 1 — Embedding Baseline

O primeiro experimento utiliza embeddings semânticos para estimar alinhamento temático. O texto da redação é dividido em chunks por sentença e cada chunk é convertido em embedding vetorial. O tema também é convertido em embedding. A similaridade cosseno entre os vetores é calculada e agregada para produzir o score final.

Pipeline:

```
tema → embedding
texto → chunking
chunks → embeddings
cosine similarity
score médio
```

Motivações:

* extremamente rápido
* baixo custo
* determinístico
* escalável

Limitações:

* não entende estrutura argumentativa
* sensível a vocabulário
* pouca nuance semântica

Esse experimento serve como baseline e como etapa inicial do método híbrido.

---

### 3.3. Experimento 2 — Prompt Engineering

O segundo experimento utiliza LLM diretamente para avaliar a redação. Dois modos são suportados:

Zero-shot: o modelo recebe apenas instruções e o tema.
Few-shot: o modelo recebe exemplos calibrados.

O modo zero-shot é mais simples e rápido, porém mais instável. O modo few-shot melhora consistência e reduz variância, ao custo de maior latência e tokens.

A escolha de suportar ambos permite comparar custo-benefício.

---

### 3.4. Experimento 3 — Estratégia Híbrida

O experimento híbrido combina embeddings e LLM. O embedding é utilizado como gate inicial. Se o score semântico for alto, o sistema retorna diretamente o resultado do embedding. Caso contrário, o LLM é acionado para uma avaliação mais detalhada.

Fluxo:

```
embedding score
     ↓
threshold
 /       \
alto     baixo
 |         |
return     LLM
```

Motivações:

* reduzir chamadas LLM
* diminuir custo
* manter precisão
* melhorar latência

Essa abordagem é comum em sistemas de produção com LLM.

---

# 4. Interface de API (FastAPI)

A API expõe o endpoint:

POST /avaliar

Payload:

```
{
 "student_id": "1",
 "text_id": "redacao_01",
 "student_text": "...",
 "theme_id": "tema_01",
 "theme_description": "...",
 "experimento": "exp3_hibrido",
 "modo_prompt": "few_shot"
}
```

Resposta:

```
{
 "alignment_score": 0.82,
 "evidence_spans": [],
 "model_version": "exp3_hibrido"
}
```

A API permite escolher dinamicamente o experimento, possibilitando benchmark online e A/B testing.

---

# 5. Segurança e Autenticação

A API aceita chave via:

Header Authorization:

```
Authorization: Bearer sk-xxxx
```

ou variável ambiente:

```
OPENAI_API_KEY
```

Isso permite integração com gateways e ambientes cloud.

---

# Contrato de Saída

O objeto de resposta contém:

* student_id
* text_id
* theme_id
* alignment_score
* evidence_spans
* model_version
* metadados

O alignment_score é float contínuo entre 0 e 1.

Os evidence_spans possuem:

* start_char
* end_char
* span_text
* label

Essa estrutura permite rastreabilidade e interpretabilidade.

---

# Estratégia de Robustez

O avaliador implementa:

* retry automático
* backoff exponencial
* validação JSON
* clamp score [0,1]
* parsing defensivo

Essas decisões tornam o sistema production-ready.

---

# Suporte a Providers

O sistema suporta:

OpenAI
Ollama local

Configuração:

```
THEMESCOPE_LLM_PROVIDER=openai
```

ou

```
THEMESCOPE_LLM_PROVIDER=ollama
```

Isso permite rodar localmente ou em cloud.

---

# Métricas

O projeto calcula:

* MAE
* taxa acerto faixa
* score médio
* custo médio
* ranking experimentos

Essas métricas permitem avaliar a qualidade do sistema.

---

# Vantagens da Arquitetura

* modular
* extensível
* interpretável
* custo-aware
* multi-provider
* produção-ready
* híbrida

---

# Provedores de LLM e Estratégia de Custo

## Visão Geral

O ThemeScope foi projetado com suporte a múltiplos provedores de LLM, permitindo alternar entre execução local e execução em nuvem. Essa decisão arquitetural foi tomada para equilibrar custo, latência e qualidade de inferência.

O sistema atualmente suporta dois provedores:

* OpenAI (cloud)
* Ollama (local)

A escolha do provedor é feita via variável de ambiente:

```
THEMESCOPE_LLM_PROVIDER=openai
```

ou

```
THEMESCOPE_LLM_PROVIDER=ollama
```

Essa abstração permite que a mesma API funcione com diferentes backends sem alteração no código de negócio.

---

# OpenAI Provider

Quando o provedor configurado é OpenAI, o sistema utiliza:

* GPT-4o para avaliação semântica
* text-embedding-3-small para embeddings

Essa configuração prioriza qualidade de inferência e consistência.

### Vantagens

* maior precisão semântica
* melhor compreensão argumentativa
* resposta mais consistente
* melhor few-shot performance
* embeddings de alta qualidade

### Desvantagens

* custo por requisição
* latência maior
* dependência externa

### Uso recomendado

* benchmark
* avaliação final
* ambiente de produção premium
* datasets críticos

---

# Ollama Provider

Quando o provedor é Ollama, o sistema utiliza modelos locais:

Exemplos:

* llama3.1:8b
* mistral
* phi
* mixtral

O modelo é configurado via:

```
THEMESCOPE_OLLAMA_MODEL=llama3.1:8b
```

### Vantagens

* custo zero por requisição
* execução offline
* baixa latência local
* privacidade de dados
* ideal para batch

### Desvantagens

* menor precisão semântica
* variabilidade maior
* embeddings não disponíveis nativamente
* dependente de hardware

### Uso recomendado

* desenvolvimento
* testes locais
* batch evaluation
* fallback

---

# Estratégia Híbrida Multi-Provider

O ThemeScope permite combinar:

LLM local + embeddings cloud

Arquitetura:

```
Ollama → reasoning
OpenAI → embeddings
```

Isso reduz custo mantendo qualidade.

Exemplo:

```
THEMESCOPE_LLM_PROVIDER=ollama
OPENAI_API_KEY=sk-xxx
```

Benefícios:

* custo reduzido
* qualidade embeddings
* baixa latência
* arquitetura moderna

---

# Estratégia de Custo

O projeto foi desenhado para comparar custo entre abordagens:

| Método    | Custo       | Qualidade | Latência |
| --------- | ----------- | --------- | -------- |
| Embedding | muito baixo | médio     | baixo    |
| Zero-shot | médio       | bom       | médio    |
| Few-shot  | alto        | alto      | alto     |
| Híbrido   | baixo       | alto      | baixo    |

---

# Custo Embeddings

Modelo:

text-embedding-3-small

Preço aproximado:

```
$0.02 / 1M tokens
```

Custo por redação:

```
~0.00002 USD
```

Praticamente desprezível.

---

# Custo LLM

GPT-4o:

Estimativa:

```
~0.006 USD por requisição
```

Few-shot:

```
~0.007 USD
```

Zero-shot:

```
~0.003 USD
```

---

# Comparação de Custo

Para 10.000 redações:

Embedding:

```
$0.20
```

Zero-shot:

```
$30
```

Few-shot:

```
$70
```

Híbrido (30% LLM):

```
$9
```

Redução:

```
~87% mais barato
```

---

# Estratégia Híbrida de Economia

O experimento híbrido executa:

```
embedding score
     ↓
threshold
     ↓
LLM apenas quando necessário
```

Isso reduz chamadas LLM.

Exemplo:

```
LLM acionado: 2/6 redações
```

Economia:

```
66% menos chamadas
```

---

# Seleção Dinâmica do Experimento

A API permite selecionar:

```
exp1_embedding
exp2_prompt
exp3_hibrido
```

Isso permite:

* medir custo
* medir latência
* comparar qualidade
* A/B test

---

# Estratégia de Produção Recomendada

Configuração ideal:

```
LLM → Ollama
Embedding → OpenAI
Experimento → híbrido
```

Resultado:

* custo baixo
* boa precisão
* latência baixa
* escalável

---

# Monitoramento de Custo

O sistema registra:

* custo estimado embedding
* custo estimado LLM
* custo híbrido
* % LLM acionado

---

# Decisão Arquitetural

A escolha por suportar múltiplos provedores foi baseada em:

* flexibilidade
* redução de custo
* produção híbrida
* independência de vendor
* escalabilidade

---


# Conclusão

O ThemeScope implementa três abordagens complementares para avaliação automática de alinhamento temático. O experimento baseado em embeddings fornece uma solução rápida e barata. O experimento baseado em LLM oferece maior qualidade semântica. A estratégia híbrida combina ambos, resultando no melhor equilíbrio entre custo, latência e precisão.

Essa arquitetura é adequada para produção e pode ser estendida para múltiplos temas, avaliação por parágrafo, ensemble de modelos e calibração de score.
