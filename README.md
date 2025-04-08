# Sistema de Busca de Empresas com Embeddings FastText

Este projeto implementa um sistema de busca de dados da Receita Federal com base em nomes digitados pelo usuário, utilizando técnicas de NLP, tokenização  e vetores de palavras (FastText) treinados localmente.

---

## Sobre o Projeto

O objetivo principal é encontrar a empresa mais semelhante ao texto digitado pelo usuário (como nomes fantasia ou razão social), mesmo com erros de digitação, abreviações, omissões e variações de grafia.

### Principais funcionalidades:
- Tokenização 
- Embeddings FastText treinados localmente para capturar similaridades semânticas
- Busca vetorial com `cosine similarity`
- Suporte a busca filtrada por UF
- Avaliação com métricas Top-1 e Top-5 (fuzzy matching)

---

## Tecnologias Utilizadas

- Python 3.11.9
- Pandas, NumPy
- NLTK
- Gensim (FastText)
- Scikit-learn
- FuzzyWuzzy
- tqdm

---
