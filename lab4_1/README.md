# Лабораторная работа №4.1
## Цель работы
Сравнить производительность и эффективность различных подходов к хранению и обработке больших данных на примере реляционной базы данных PostgreSQL и документо ориентированной базы данных MongoDB.
## Индивидуальные задания
### Задание №12. Сортировка
#### Для PostgreSQL
1. Создать таблицу logs (100 000 записей) с полем timestamp.
2. Выполнить запрос SELECT * FROM logs ORDER BY timestamp DESC LIMIT 1000.
#### Для MongoDB
1. Создать коллекцию logs (100 000 записей).
2. Выполнить запрос find().sort("timestamp", -1).limit(1000).
#### Анализ в Jupyter Notebook
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
### Шаг 2. Создание функций для проверки подключения к базам данных
```
def check_mongo_connection(client):
    """Проверка подключения к MongoDB"""
    try:
        client.server_info()
        print("✅ Успешное подключение к MongoDB")
        return True
    except Exception as e:
        print(f"❌ Ошибка подключения к MongoDB: {e}")
        return False

def check_postgres_connection(conn_params):
    """Проверка подключения к PostgreSQL"""
    try:
        conn = psycopg2.connect(**conn_params)
        print("✅ Успешное подключение к PostgreSQL")
        return conn
    except Exception as e:
        print(f"❌ Ошибка подключения к PostgreSQL: {e}")
        return None

def measure_time(func, *args, **kwargs):
    """Измерение времени выполнения функции"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time
```
### Шаг 3. Генерация данных
Генерация данных о логах:
```
np.random.seed(42)
 
# Параметры 
n_logs = 100000  # 100,000 записей

print(f"Генерация {n_logs:,} записей логов...")
 
# Генерация данных логов
logs_data = []
start_date = datetime(2024, 1, 1)
 
# Уровни логирования
log_levels = ['INFO', 'ERROR', 'WARNING', 'DEBUG']
 
for i in range(n_logs):
    # Случайная дата в пределах года
    days_offset = np.random.randint(0, 365)
    timestamp = start_date + timedelta(days=days_offset)
    
    logs_data.append({
        'log_id': i + 1,
        'timestamp': timestamp,
        'log_level': np.random.choice(log_levels),
        'message': f'Log entry number {i + 1}'
    })
 
# Создание DataFrame
logs_df = pd.DataFrame(logs_data)
 
print(f"Создано {len(logs_df):,} записей логов")
 
# Вывод первых записей
print("\nПример данных:")
print(logs_df.head())
```
Результат выполнения:

   <img width="800" height="420" alt="image" src="images/Снимок%20экрана%202025-10-19%20181146.png" />

Сохранение данных в csv-файлы:
```
logs_df.to_csv('log.csv', index=False)

print("✅ Данные сохранены в CSV файл:")
print("- log.csv")

# Анализ данных
print(f"\n📊 Анализ данных:")

print(f"📈 Распределение по уровням логирования:")
print(logs_df['log_level'].value_counts())

# Количество логов в день
avg_logs_per_day = len(logs_df) / logs_df['timestamp'].dt.date.nunique()
print(f"📊 Среднее количество логов в день: {avg_logs_per_day:.1f}")
```
Результат выполнения:

   <img width="800" height="320" alt="image" src="images/Снимок%20экрана%202025-10-19%20181159.png" />
