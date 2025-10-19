# Лабораторная работа №4.1
## Цель работы
Сравнить производительность и эффективность различных подходов к хранению и обработке больших данных на примере реляционной базы данных PostgreSQL и документо ориентированной базы данных MongoDB.
## Индивидуальные задания
### Задание №12. Сортировка
### Для PostgreSQL
1. Создать таблицу logs (100 000 записей) с полем timestamp.
2. Выполнить запрос SELECT * FROM logs ORDER BY timestamp DESC LIMIT 1000.
### Для MongoDB
1. Создать коллекцию logs (100 000 записей).
2. Выполнить запрос find().sort("timestamp", -1).limit(1000).
### Анализ в Jupyter Notebook
Сравнить производительность операций сортировки большого объема данных. Проанализировать, как индексирование поля сортировки влияет на результат в обеих СУБД.

## Подготовка окружения
1. Загрузка файла docker-compose.yml:
   
   <img width="700" height="420" alt="image" src="images/Снимок%20экрана%202025-10-19%20151628.png" />

2. Запуск всех сервисов:

   <img width="880" height="180" alt="image" src="images/Снимок%20экрана%202025-10-19%20152151.png" />

## Шаги решения индивидуальных заданий

### Шаг 1. Установка и импорт необходимых библиотек

Установка библиотек:
```
!pip install pandas numpy pymongo psycopg2-binary sqlalchemy matplotlib seaborn
```

Результат выполнения:

   <img width="660" height="400" alt="image" src="images/Снимок%20экрана%202025-10-19%20172632.png" />

Импорт библиотек:
```
import pandas as pd
import numpy as np
from pymongo import MongoClient
import psycopg2
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Настройка для отображения графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```
