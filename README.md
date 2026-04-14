# ThemeScope

Protótipo de NLP com **três experimentos comparativos** para estimar o alinhamento entre a redação de um estudante e a proposta temática.
O projeto está configurado para usar **LLM local com Ollama** por padrão.

---

## Início rápido

```bash
git clone <url>
cd ThemeScope
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Inicie Ollama e baixe um modelo local
ollama serve
ollama pull llama3.1:8b

python scripts/experimentos.py

pytest -q

uvicorn api.main:app --reload
```

---

## Como executar o projeto

### 1) Preparar o ambiente

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Configurar o LLM local 

```bash
ollama serve
ollama pull llama3.1:8b
export THEMESCOPE_LLM_PROVIDER=ollama
export THEMESCOPE_OLLAMA_MODEL=llama3.1:8b
```

Opcional (usar OpenAI em vez de Ollama):

```bash
export THEMESCOPE_LLM_PROVIDER=openai
export OPENAI_API_KEY=
```

### 3) Rodar os experimentos comparativos

```bash
python scripts/experimentos.py
```

Modo econômico (reduz chance de erro 429):

```bash
python scripts/experimentos.py --modo-economico
```

Opcional: ajustar pausa manual entre chamadas (em segundos):

```bash
export THEMESCOPE_PAUSA_S=3
python scripts/experimentos.py
```

Saídas esperadas:
- tabela comparativa no terminal
- arquivos em `resultados/` com resultados e métricas
- Obs.: `Exp1` usa embedding local gratuito. Na primeira execução ele baixa o modelo
  `paraphrase-multilingual-MiniLM-L12-v2`.

### 4) Rodar os testes

```bash
pytest -q
```

Lint (Ruff; configuração em `pyproject.toml`):

```bash
pip install ruff
ruff check src api tests
```

### 5) Subir a API HTTP

Na raiz do repositório:

```bash
uvicorn api.main:app --reload
```

Endpoints principais:
- `GET /health`
- `POST /avaliar`

Exemplo rápido com `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/avaliar" \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "aluno_01",
    "text_id": "redacao_01",
    "student_text": "O uso excessivo de redes sociais pode causar ansiedade em jovens.",
    "theme_id": "tema_01",
    "theme_description": "Discuta os impactos do uso excessivo de redes sociais na saúde mental dos jovens.",
    "experimento": "exp3_hibrido",
    "modo_prompt": "few_shot"
  }'
```

Acesse http://127.0.0.1:8000/docs#/default/avaliar_avaliar_post 

---

## Os três experimentos

### Exp 1 — Embedding Baseline
**Pipeline:** embedding local do tema → chunking da redação (sliding window de 3 sentenças) → similaridade cosseno por chunk → média ponderada.

- **Custo:** US$0/req (local/offline após download inicial do modelo)
- **Latência:** ~0.2–1.5s (depende da máquina)
- **Limitação:** não captura raciocínio causal; sensível a vocabulário temático sem desenvolvimento

### Exp 2 — Prompt Engineering (Zero-shot vs. Few-shot)
**Pipeline:** GPT-4o com output JSON estruturado. Compara dois modos:
- `zero_shot`: system prompt + schema → ~450 tokens de entrada
- `few_shot`: + 3 exemplos calibrados (alto/parcial/baixo) → ~1.350 tokens de entrada

- **Custo:** ~US$0,004–0,007/req
- **Latência:** ~2–4s
- **Vantagem:** raciocínio semântico profundo, explicitado no campo `raciocinio`
- **Execução recomendada:** Ollama local (`llama3.1:8b`)

### Exp 3 — Híbrido (Embedding Gate + LLM Condicional)
**Pipeline:** calcula score de embedding local → se fora da zona de incerteza (< 0,35 ou > 0,65) usa embedding diretamente; se na zona → aciona LLM few-shot.

- **Custo médio esperado:** depende da taxa de acionamento do LLM (parte de embedding é local/US$0)
- **Taxa de LLM esperada:** ~30–40% das requisições
- **Metadados:** `score_embedding`, `llm_acionado`, `delta_emb_vs_llm`, `custo_usd_estimado`

---

## Uso programático

```python
from src.experimentos import exp1_embedding_baseline as exp1
from src.experimentos import exp2_prompt_engineering as exp2
from src.experimentos import exp3_hibrido as exp3

kwargs = dict(
    student_id="aluno_01", text_id="redacao_01",
    student_text="O uso excessivo de redes sociais...",
    theme_id="tema_01",
    theme_description="Discuta os impactos do uso excessivo de redes sociais na saúde mental dos jovens.",
    chave_api="",  # obrigatório apenas se THEMESCOPE_LLM_PROVIDER=openai
)

r1  = exp1.avaliar(**kwargs)                    # embedding baseline
r2a = exp2.avaliar(**kwargs, modo="zero_shot")  # zero-shot prompt
r2b = exp2.avaliar(**kwargs, modo="few_shot")   # few-shot prompt
r3  = exp3.avaliar(**kwargs)                    # híbrido
```

---

## Contrato de saída (todos os experimentos)

```json
{
  "student_id": "aluno_01",
  "text_id": "redacao_01",
  "theme_id": "tema_01",
  "alignment_score": 0.87,
  "evidence_spans": [
    {"start_char": 0, "end_char": 52, "span_text": "...", "label": "alinhado"}
  ],
  "model_version": "hibrido-v1.0.0",
  "metadados": {
    "experimento": "hibrido",
    "llm_acionado": true,
    "score_embedding": 0.51,
    "custo_usd_estimado": 0.005
  }
}
```

`label`: `"alinhado"` | `"fora_do_tema"` | `"parcial"`



---

## Documentação adicional


- Documentação técnica: `documentacao.md`
- Análise da execução dos experimentos: `resultados_experimento_analise.md`

---

## Reprodutibilidade

- `temperature=0` em todas as chamadas de LLM
- Embeddings são determinísticos (mesmo input → mesmo vetor)
- Sem componentes estocásticos locais — sem necessidade de seed

## Ambiente

- Python >= 3.10
- Dependências em `requirements.txt` (FastAPI, numpy, requests, pytest, etc.)
- Ollama (padrão): `THEMESCOPE_LLM_PROVIDER=ollama` e modelo local carregado
- OpenAI (opcional): `THEMESCOPE_LLM_PROVIDER=openai` + `OPENAI_API_KEY`
