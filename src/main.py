# Imports
import pandas as pd
import nltk
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from utils import avaliar_modelo, clean_text, ft_embedding, get_topk, prepare_search_text, tokenize_business

# Download necessário dos recursos NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')


df = pd.read_parquet('dados/train.parquet')
df = df[['user_input', 'uf', 'razaosocial', 'nome_fantasia']].reset_index(drop=True)


# Criação do campo de texto e do target
df['target_empresa'] = (df.razaosocial.fillna('') + ' ' + df.nome_fantasia.fillna('')).apply(clean_text)

# Separação treino/teste
df_train, df_test = train_test_split(df, 
                                     test_size=0.2, 
                                     random_state=42)

# Aplica o prepare_search_text em todos os DataFrames
df_train['search_text'] = df_train.apply(prepare_search_text, axis=1)
df_test['search_text'] = df_test.apply(prepare_search_text, axis=1)



# FastText com parâmetros otimizados para nomes de empresas
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

# Testando um um df menor
df_test_test = df_test[1000:2000]

# Avaliando o modelo
avaliar_modelo(df_test_test, base_busca=df_train)
get_topk('braspres', uf='SP', base_busca=df_test)