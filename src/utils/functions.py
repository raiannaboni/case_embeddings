# Imports
import pandas as pd
import numpy as np
import re
import unicodedata
from gensim.models import FastText
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from tqdm.notebook import tqdm

def clean_text(text):
    text = str(text).lower().strip()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Remove caracteres especiais
    text = re.sub(r'[^a-z0-9\s&.-]', ' ', text)
    
    # Trata números como tokens especiais
    text = re.sub(r'(\d+)', r' \1 ', text)
    
    # Remove stopwords específicas do domínio
    stopwords = ['ltda', 'me', 'epp', 'sa', 's/a', 'limitada', 'eireli']
    words = text.split()
    words = [w for w in words if w not in stopwords]
    
    # Normaliza espaços
    return ' '.join(words)


# Função para preparar o texto para busca, priorizando nome fantasia (dei mais peso ao nome fantasia por ser mais fácil o cliente usar os termos deste)
def prepare_search_text(row):
    razao = clean_text(str(row.razaosocial)) if pd.notna(row.razaosocial) else ''
    fantasia = clean_text(str(row.nome_fantasia)) if pd.notna(row.nome_fantasia) else ''
    
    # Se tiver ambos, combina dando mais peso ao nome fantasia
    if razao and fantasia:
        # Repete o nome fantasia para dar mais peso
        return f'{fantasia} {fantasia} {razao}'
    # Se tiver só um deles, usa o que tiver
    return fantasia or razao

# Tokenização dos nomes de empresas
def tokenize_business(text):
    text = clean_text(text)
    
    tokens = []
    for word in text.split():
        # Se tem número, mantém junto
        if any(c.isdigit() for c in word):
            tokens.append(word)
        else:
            # Tokeniza mas mantém palavras pequenas (possíveis iniciais)
            if len(word) <= 2:
                tokens.append(word)
            else:
                tokens.extend(word_tokenize(word))
    return tokens

# Tokenização dos nomes de empresas
def tokenize_business(text):
    text = clean_text(text)
    
    tokens = []
    for word in text.split():
        # Se tem número, mantém junto
        if any(c.isdigit() for c in word):
            tokens.append(word)
        else:
            # Tokeniza mas mantém palavras pequenas (possíveis iniciais)
            if len(word) <= 2:
                tokens.append(word)
            else:
                tokens.extend(word_tokenize(word))
    return tokens

def fasttext_model(df_train, df_test):
    
    corpus_ft = [tokenize_business(text) for text in df_train.search_text]

    fasttext_model = FastText(
        sentences=corpus_ft,
        vector_size=300,
        window=5, 
        min_count=1,
        sg=1,
        negative=15,  
        epochs=50,
        min_n=2,
        max_n=6
        )


    # Aplicando os embeddings
    print("Gerando embeddings FastText...")
    df_train['ft_emb'] = df_train.search_text.apply(ft_embedding)
    df_test['ft_emb'] = df_test.search_text.apply(ft_embedding)

    return fasttext_model

def ft_embedding(text):
    """Função para gerar embeddings com pesos adaptativos"""
    tokens = tokenize_business(text)
    vectors = []
    weights = []
    
    for i, token in enumerate(tokens):
        if token in fasttext_model.wv:
            vec = fasttext_model.wv[token]
            # Pesos adaptativos baseados nas características do token
            weight = 1.0
            
            # Aumentar o peso das primeiras palavras
            if i < 2:
                weight *= 1.2 

            # Diminuir peso dos tokens curtos
            if len(token) <= 2:
                weight *= 0.6  
                
            vectors.append(vec)
            weights.append(weight)
    
    if vectors:
        weights = np.array(weights).reshape(-1, 1) 
        weighted_vectors = np.array(vectors) * weights

        # Normaliza o embedding final
        emb = np.sum(weighted_vectors, axis=0) / np.sum(weights)
        return emb / np.linalg.norm(emb)
    
    return np.zeros(fasttext_model.vector_size)

def get_topk(user_input, uf=None, k=5, base_busca=None):
    texto = tokenize_business(user_input)
    data = base_busca.copy()
    
    if uf:
        data = data[data.uf == uf].reset_index(drop=True)    # verificar o valor de data
    

    emb = ft_embedding(texto)
    similaridade = cosine_similarity([emb], list(data.ft_emb.values))[0]
   
    top_indices = similaridade.argsort()[::-1][:k]

    return pd.DataFrame(data.iloc[top_indices])

def avaliar_modelo(df_teste, base_busca, batch_size=32):
    total = len(df_teste)
    top1 = 0
    top5 = 0
    
    # Pré-processamento dos alvos
    alvos = df_teste.search_text.apply(tokenize_business).values
    
    with tqdm(total=total, desc=f'Avaliando modelo') as pbar:
        for i in range(0, total, batch_size):
            batch = df_teste.iloc[i:i+batch_size]
            
            for j, row in enumerate(batch.itertuples()):
                entrada = row.user_input
                uf = row.uf if hasattr(row, 'uf') else None
                alvo = alvos[i+j]
                
                # Obtém os top-k resultados
                resultados = get_topk(entrada, uf, k=5, base_busca=base_busca)
                pred_empresas = resultados.search_text.apply(tokenize_business).values
                                
                # Verifica Top-1 (match exato ou alta similaridade para primeira predição)
                if fuzz.ratio(alvo, pred_empresas[0]) > 80:  
                    top1 += 1
                
                # Verifica Top-5 (match aproximado em qualquer posição)
                for pred in pred_empresas:
                    if fuzz.ratio(alvo, pred) > 80:
                        top5 += 1
                        break
                
                pbar.update(1)
    
    print(f'\nModelo: FastText')
    print(f'Top-1: {top1/total:.2%} | Top-5: {top5/total:.2%}')