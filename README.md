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
├── src/
│   ├── main.py            # Script principal
│   └── utils.py           # Funções auxiliares
├── fasttext_model.ipynb   # Notebook com implementação do modelo
└── requirements.txt       # Dependências do projeto
```

## Tecnologias Utilizadas
- Python 3.11
- FastText (Gensim)
- pandas
- numpy
- scikit-learn
- NLTK
- FuzzyWuzzy

## Instalação

### Pré-requisitos
- Python 3.11+
- pip

### Configuração
1. Clone o repositório:
```bash
git clone [url-do-repositorio]
cd case_embeddings
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Baixe os recursos do NLTK:
```python
import nltk
nltk.download('punkt')
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

## Detalhes da Implementação

### FastText Training
```python
fasttext_model = FastText(
    sentences=corpus_ft,
    vector_size=300,     # Dimensão dos vetores
    window=5,            # Janela de contexto
    min_count=1,         # Inclui todas as palavras
    sg=1,               # Skip-gram
    negative=15,        # Negative sampling
    epochs=50,          # Número de épocas
    min_n=2,           # Tamanho mínimo de n-grams
    max_n=6            # Tamanho máximo de n-grams
)
```

### Embeddings com Pesos Adaptativos
```python
def ft_embedding(text):
    tokens = tokenize_business(text)
    vectors = []
    weights = []
    
    for i, token in enumerate(tokens):
        if token in fasttext_model.wv:
            vec = fasttext_model.wv[token]
            weight = 1.0
            
            # Primeiras palavras são mais importantes
            if i < 2:
                weight *= 1.2
            
            # Tokens curtos têm peso menor
            if len(token) <= 2:
                weight *= 0.6
                
            vectors.append(vec)
            weights.append(weight)
    
    return weighted_average(vectors, weights)
```

## Performance

Após validação cruzada em um conjunto de teste com 4000 amostras:
- **Top-1 Accuracy**: 82.5%
- **Top-5 Accuracy**: 89.3%

O modelo apresenta excelente performance mesmo com:
- Variações de escrita (exemplo: "itau" vs "itaú")
- Erros de digitação comuns
- Uso de abreviações
- Diferentes formatos de nome (razão social vs nome fantasia)

## Licença
MIT License

## Autor
[Case Data Science - Embeddings]
