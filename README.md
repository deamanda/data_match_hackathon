## Команда 8
Антон Борисовский

Алексей Поляков

Курбан Абдурахманов


## Описание
Используемая модель - rubert-tiny2

Векторизуем текст.

Евклидовы расстояния между векторами считаем с помощью scipy.cdist

Выбираем заданное количество ближайших.

Аналитическая преобработка названий значимого результата не дала.


## Запуск
```
pip install requirement.txt
python3 match.py
```