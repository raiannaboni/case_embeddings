# Imports
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from utils.functions import avaliar_modelo, clean_text, ft_embedding, get_topk, prepare_search_text, tokenize_business

nltk.download('punkt')
    
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





# Testando um um df menor
df_test_test = df_test[1000:2000]

# Avaliando o modelo
avaliar_modelo(df_test_test, base_busca=df_train)
get_topk('braspres', uf='SP', base_busca=df_test)