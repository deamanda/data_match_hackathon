## Команда 8
Антон Борисовский

Алексей Поляков

Курбан Абдурахманов


## Описание
Решается задача матчинга названий товаров у производителя и розничных продавцов.    
Для решения поставленной задачи мы использовали модель  - rubert-tiny2.     
Для повышения точности работы модели, мы дообучили ее на предобработанных и размеченных данных заказчика.    
Преобработка заключается в приведении всех строк к одному виду.    
- английский текст в названиях был перенесен в начало,    
- название производителя удалено    
- объем упаковки выделен в отдельный признак для возможной фильтрации по нему    
- удалены стоп-слова, знаки припинания, приведена к единому виду запись объема.    
Получившиеся названия векторизированы моделью.    
Для определения совпадений подсчитывается евклидово расстояние между получившимися векторами.    
Наименьшее расстояние считается наибольшим совпадением.    
Для повышения точности модели, из названий розничных продавцов выделяется объем упаковки, после чего сравненгие с базой производителя производится только по этому объему.
Если объем выделить не удалось, сравнение производится со всей базой.    
Функция матчинга дает возможность указать необходимое количество возвращаемых вариантов.    

Точность модели считали как отношение верно предсказанных названий на первом месте (или в первой пятерке) ко всем предсказаниям.   


Достигнутая в настоящее время точность:
название предсказывается на 1 месте -  84,15%
название предсказывается в первой пятерке - 93.41%


## Запуск
```
pip install requirement.txt
python3 match.py
```