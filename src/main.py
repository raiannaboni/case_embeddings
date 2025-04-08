import os
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from utils.functions import avaliar_modelo, clean_text, prepare_search_text, fasttext_model

nltk.download('punkt')
nltk.download('stopwords')

def main():
    # Construir caminho absoluto para o arquivo de dados
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'dados', 'train.parquet')

    # Carregar o dataset
    df = pd.read_parquet(data_path)
    df = df[['user_input', 'uf', 'razaosocial', 'nome_fantasia']].reset_index(drop=True)

    # Criar o campo de texto e o target
    df['target_empresa'] = (df.razaosocial.fillna('') + ' ' + df.nome_fantasia.fillna('')).apply(clean_text)

    # Separar treino/teste
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Aplicar o prepare_search_text
    df_train['search_text'] = df_train.apply(prepare_search_text, axis=1)
    df_test['search_text'] = df_test.apply(prepare_search_text, axis=1)

    # Treinar o modelo FastText
    model = fasttext_model(df_train, df_test)

    # Avaliar o modelo
    resultados = avaliar_modelo(df_test, base_busca=df_train, model=model)
    print("Resultados da avaliação:", resultados)

if __name__ == '__main__':
    main()