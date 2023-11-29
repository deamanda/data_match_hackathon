import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

def create_embeddings(list_name):
    # функция создает и возвращает векторы текста
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    return model.encode(list_name)


def match(embeddings_factory, list_id_name_market, count_var=5):
    # подсчет расстояний между переданными векторами,
    # и возврат {count_var} ближайших, это будут рекомендованные варианты
    # получает список векторов производителя, список [product_key, product_name] маркетов, число нужных вариантов
    # позвращает индексы переданных векторов производителя
    if not (0 < count_var < 51):
        count_var = 5
    market_name = [row[1] for row in list_id_name_market]
    embeddings_market = create_embeddings(market_name)
    list_index = []
    for emb in embeddings_market[range(10)]:
        distances = cdist(np.expand_dims(emb, axis=0), embeddings_factory, metric='euclidean')
        index_vec_product = np.argsort(distances)[:, :count_var].tolist()
        list_index += index_vec_product
    return list_index


def main():
    # Модуль будет библиотекой
    # main нужен только для тестирования работы функций

    df_dealer = pd.read_csv('marketing_dealerprice.csv', delimiter=';')
    df_factory = pd.read_csv('marketing_product.csv', delimiter=';')
    dt_id = pd.read_csv('marketing_dealer.csv', delimiter=';')
    dt_key = pd.read_csv('marketing_productdealerkey.csv', delimiter=';')

    # тестовые списки и количество вариантов
    list_id_name_prod = df_factory[['id', 'name']].values.tolist()
    list_id_name_market = df_dealer[['product_key', 'product_name']].values.tolist()
    count_var = 6

    # создание векторов названий производителя
    # получаем от бэков список названий
    list_name_prod = [row[1] for row in list_id_name_prod]
    embeddings_factory = create_embeddings(list_name_prod)
#    embeddings_market = create_embeddings(list_id_name_market)

    # предсказание названий
    # получаем список векторов производителя, список названий на маркете, количество возвращаемых вариантов
    list_predisct = match(embeddings_factory, list_id_name_market, count_var)

    # тестовый вывод
    ind = 0
    for i in list_predisct:
        print(f"Для '{df_dealer.iloc[ind]['product_name']}' предсказано:")
        print(i)
        print(df_factory.iloc[i][['id', 'name']])
        ind += 1

    return 0

if __name__ == "__main__":
    main()