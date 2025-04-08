# Sistema de Busca Semântica com FastText

## Sobre o Projeto
Sistema de busca semântica que encontra empresas a partir de texto livre, utilizando FastText para lidar com variações de escrita e erros de digitação. O sistema foi desenvolvido para um case de Data Science, alcançando resultados competitivos nas métricas de avaliação.

### Principais Características
- Busca semântica usando FastText com pesos adaptativos
- Tratamento robusto de variações de escrita
- Sistema de pesos para priorizar nome fantasia vs razão social
- Filtragem por UF para melhor precisão
- Tokenização específica para nomes de empresas

## Estrutura do Projeto
```
case_embeddings/
├── dados/
│   └── train.parquet              # Dataset de treino
├── src/
│   ├── main.py                    # Script principal
│   └── utils/
│       └── functions.py           # Funções auxiliares
├── notebooks/
    ├── documentacao_solucao.ipynb # Documentação detalhada
│   ├── fasttext_model.ipynb       # Implementação 
│   └── testes.ipynb               # Testes e experimentos
└── requirements.txt               # Dependências do projeto
```

## Como Usar

### Via Notebook
Use o notebook `fasttext_model.ipynb` para:
- Treinar o modelo FastText
- Gerar embeddings
- Avaliar o desempenho
- Fazer buscas de exemplo

### Via Script Python
```python
from src.utils import clean_text, ft_embedding, get_topk

# Buscar empresas similares
resultados = get_topk('banco itau', uf='SP', k=5, base_busca=df_train)
```

## Features Principais

### 1. Pré-processamento
- Limpeza e normalização de texto
- Remoção de stopwords específicas do domínio
- Tratamento especial para números e abreviações

### 2. FastText Otimizado
- Vector size: 300
- Skip-gram model
- Subword embeddings (n-grams)
- Parâmetros otimizados para nomes de empresas

### 3. Sistema de Pesos Adaptativos
- Maior peso para palavras iniciais
- Ajuste baseado no tamanho dos tokens
- Priorização de nome fantasia

### 4. Avaliação
O modelo é avaliado usando duas métricas principais:
- **Top-1**: % de acerto na primeira sugestão
- **Top-5**: % de acerto entre as 5 primeiras sugestões

## Autor
Raianna Boni
