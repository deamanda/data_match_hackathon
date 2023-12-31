import numpy as np
import pandas as pd
import re
from transliterate import translit
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
# import torch
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 10)

def del_sub_str(string, sub_str): # заменяем подстроки на пробел
    for i in sub_str:
        string = string.replace(i, ' ')
    return string

def norm_name(string: str) -> (str, str) :
    '''
    Функция нормализации названий и ПОЛУЧЕНИЯ ОБЪЕМА из названия
    Форматирует строку по заданным правилам.
    Подготовка для расчета векторов.
    :param: (str) Строка которую нужно нормализовать
    :return: (str), (str) Отформатированная строка, Строка содержащая объем продукта из названия
    '''
    stop_words = ['и', 'для', 'из', 'с', 'c', 'в', 'ф', 'п', 'д']
    t = {
        '0.5':'500 ',
        '0.8':'800 ',
        '0.4':'400 ',
        '0.6':'600 ',
        '0.25':'250 ',
        '0.75':'750 ',
        '0.90':'900',
        '0.9':'900 ',
        '0.2':'200 ',
        '0.1':'100 ',
        '0.05':'50 '
    }
    string = string.lower()
    string = re.sub(r'(\d),(\d)', r'\1.\2', string)         # заменяем , на . в десятичных числах
    string = re.sub(r'(\d+)([лмлкг])', r'\1 \2', string)    # отодвигаем л, мл, кг от цифр, если написано слитно
    string = del_sub_str(string, ',-:"+«»()/\'')                 # удаляем все символы
    string = del_sub_str(string, ['prosept', 'просепт'])         # удаляем название производителя
    string = re.sub(r'\bлитр[о]?в?\b', ' л', string)        # заменяем литры на л
    string = string.rstrip('.')
    string = ' '.join([word.strip() for word in string.split() if not word in stop_words]) # удаляем стоп слова
    template = re.compile(r'([A-Za-z]+)')
    english = [s.strip() for s in template.findall(string) if s.strip()] # ищем английский текст
    for eng in english:                                                  # удаляем английский текст
        string = string.replace(eng, ' ')
    for num, num1 in t.items():                                          # заменяем десятичные дроби на целые значения
        if num in string:
            string = string.replace(num, num1)
            string = string.replace(' л', ' мл')
            string = string.replace(' кг', ' г')
    string = translit(' '.join(english), 'ru') + ' ' + string # добавляем в начало строки английский текст транслитом
    string = ' '.join([word.strip() for word in string.split() if word.strip()])  # удаляем лишние пробелы
    size = re.findall(r'(\d+\.\d+|\d+)\s*(?:л|г|кг|мл)\b', string) + ['0'] # находим объем в названии
    return string, size[0]


def df_correct(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    '''
    Проверяем датасет, заполняем/удаляем пропуски.
    :param df: Датасет для корректировки, Имя столбца по которому удаляем пропуски
    :return: Датасет без пропусков в нужном столбце
    '''
    df.dropna(subset=name_col, inplace=True)
    if name_col == 'name' :
        df['name_1c'] = df['name_1c'].fillna(df['name'])
    return df



def create_embeddings(model, list_name):
    '''
    Функция получает на вход обученную модель и список названий.
    Возвращает список векторов, расчитанных по названиям.
    :param model: Модель ML
    :param list_name: Список названий товаров
    :return: Список векторов (числового представления названия)
    '''
    return model.encode(list_name).tolist()

# def get_data_market(model, list_name):
#     result = []
#     list_name_norm = []
#     for name in list_name:
#         name_n, size = norm_name(name)
#         list_name_norm.append(name_n)
#         result.append([name_n, size])
#     result = result + create_embeddings(model, list_name_norm)
#     return result


def match(embeddings_factory, list_emb_market, count_var=0):
    '''
    Функция матчинга сопоставляет векторы из 2-х списков.
    Подсчитывает расстояние между векторами и возвращает 
    id {count_var} ближайших из списка производителя.
    :param embeddings_factory: list [id: int, size: str, vector: list[float * 312]] Список id, объема и ВЕКТОРОВ названий производителя
    :param list_emb_market: list [size: str, vector: list[float * 312]] Список объемов и ВЕКТОРОВ названий маркетов
    :param count_var: int Число вариантов, которое необходимо вернуть. 0 - вернуть все предсказания.
    :return: list[id: list[int * count_var]] Список предсказаний для каждого элемента list_name_market
    '''
    if not (0 < count_var < 51):
        count_var = None

    predict = []
    for emb, size in list_emb_market:
        index_predict = []
        emp_temp_id = embeddings_factory[embeddings_factory[:, 1] == size, 0]
        emp_temp_vector = embeddings_factory[embeddings_factory[:, 1] == size, 2]
        if not len(emp_temp_vector):
            emp_temp_id = embeddings_factory[:, 0]
            emp_temp_vector = embeddings_factory[:, 2]
        distances = cdist(np.expand_dims(emb, axis=0), list(emp_temp_vector), metric='euclidean')
        index_predict = np.argsort(distances)[:, :count_var].tolist()
        predict.append([emp_temp_id[i] for i in index_predict[0]])
    return predict



def main():
    # Модуль будет библиотекой
    # main нужен только для тестирования работы функций

    # имя столбеца в данных производителя по которому считаем векторы
    NAME_PRODUCT = 'name_1c'
    COUNT_VAR = 0

    df_dealer = pd.read_csv('marketing_dealerprice.csv', delimiter=';')
    df_factory = pd.read_csv('marketing_product.csv', delimiter=';')
    dt_id = pd.read_csv('marketing_dealer.csv', delimiter=';')
    dt_key = pd.read_csv('marketing_productdealerkey.csv', delimiter=';')
    model = SentenceTransformer('model_50e02')
    #model = SentenceTransformer('cointegrated/rubert-tiny2')
    #model = SentenceTransformer('d0rj/e5-small-en-ru')


    # удаляем пропуски в данных и оставляем только нужные столбцы
    df_factory = df_factory[['id','name', 'name_1c']]
    df_factory = df_correct(df_factory, 'name') # NAME_PRODUCT
    df_dealer = df_dealer[['product_name', 'product_key']]
    df_dealer = df_correct(df_dealer, 'product_name')

    # удаляем дубликаты названий маркетов, сбрасываем индекс
    df_dealer.drop_duplicates(subset='product_name', inplace=True)
    df_factory.reset_index(inplace=True, drop=True)
    df_dealer.reset_index(inplace=True, drop=True)

    # добавляем столбцы с форматированным названием и объемом
    df_factory[['name_1c_norm', 'size']] = df_factory[NAME_PRODUCT].apply(norm_name).apply(pd.Series)
    df_dealer[['product_name_norm', 'size']] = df_dealer['product_name'].apply(norm_name).apply(pd.Series)


    # добавляем столбцы с векторами
    df_factory['vectors'] = create_embeddings(model, df_factory['name_1c_norm'].values)
    df_dealer['vectors'] = create_embeddings(model, df_dealer['product_name_norm'].values)

    # подготовка данных для матчинга
    list_emb_market = df_dealer[['vectors', 'size']].values
    list_emb_product = df_factory[['id', 'size', 'vectors']].values


    # list_emb_market = get_data_market(model, df_dealer['product_name_norm'].values)
    # print(list_emb_market[0])
    # #list_emb_product = df_factory[['id', 'size', 'vectors']].values

    # отправили на матчинг всю базу маркетов
    list_predisct = match(list_emb_product, list_emb_market, COUNT_VAR)

    # записываем результат в таблицу и считаем угаданные
    df_result = df_dealer[['product_name', 'product_key']]
    df_result['predict_1'] = [i[0] for i in list_predisct]
    df_result['predict_all'] = list_predisct

    # добавляем в результат id
    df_result = pd.merge(df_result, dt_key, left_on='product_key', right_on='key', how='inner')
    df_result = df_result.drop(['product_key', 'key', 'id', 'dealer_id'], axis=1)
    df_result = df_result.rename(columns={'product_id': 'id'})

    # считаем правильные ответы
    df_result['true_1'] = df_result['id'] == df_result['predict_1']
    df_result['true_2'] = df_result.apply(lambda x: x['id'] in x['predict_all'][:2], axis=1)
    df_result['true_5'] = df_result.apply(lambda x: x['id'] in x['predict_all'][:5], axis=1)
    df_result['true_10'] = df_result.apply(lambda x: x['id'] in x['predict_all'][:10], axis=1)
    df_result['true_20'] = df_result.apply(lambda x: x['id'] in x['predict_all'][:20], axis=1)
    df_result['true_all'] = df_result.apply(lambda x: x['id'] in x['predict_all'], axis=1)

    print(f'Полученная точность:')
    print(f'первый - {df_result["true_1"].sum()/len(df_result["true_1"])}')
    print(f'первый/второй - {df_result["true_2"].sum()/len(df_result["true_1"])}')
    print(f'в пятерке - {df_result["true_5"].sum()/len(df_result["true_1"])}')
    print(f'в десятке - {df_result["true_10"].sum()/len(df_result["true_1"])}')
    print(f'в двадцатке - {df_result["true_20"].sum()/len(df_result["true_1"])}')
    print(f'они вообще есть? - {df_result["true_all"].sum()/len(df_result["true_1"])}')

    print(df_result)

    return 0

if __name__ == "__main__":
    main()