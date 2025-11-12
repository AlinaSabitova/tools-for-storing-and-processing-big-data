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
#### Установка библиотек:
```
!pip install pandas numpy pymongo psycopg2-binary sqlalchemy matplotlib seaborn
```
Результат выполнения:

   <img width="660" height="400" alt="image" src="images/Снимок%20экрана%202025-10-19%20172632.png" />

#### Импорт библиотек:
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
#### Генерация данных о логах:
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

#### Сохранение данных в csv-файлы:
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

### Шаг 4. Реализация в PostgreSQL
#### Подключение к PostgreSQL и загрузка данных:
```
pg_conn_params = {
    "dbname": "studpg",
    "user": "postgres",
    "password": "changeme",
    "host": "postgresql",  # Имя сервиса в docker-compose
    "port": "5432"
}

pg_conn = check_postgres_connection(pg_conn_params)
if pg_conn:
    try:
        # Создание таблиц
        with pg_conn.cursor() as cur:
            # Удаление существующих таблиц
            cur.execute("DROP TABLE IF EXISTS logs CASCADE")
            
            # Создание таблицы 
            cur.execute("""
                CREATE TABLE logs (
                    log_id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    log_level VARCHAR(10),
                    message TEXT
                )
            """)
        
        print("✅ Созданы таблицы")
        
        # Загрузка данных
        print("📥 Загрузка данных в PostgreSQL...")
        
        # Загрузка логов
        with pg_conn.cursor() as cur:
            for _, row in logs_df.iterrows():
                cur.execute("""
                    INSERT INTO logs (log_id, timestamp, log_level, message)
                    VALUES (%s, %s, %s, %s)
                """, (row['log_id'], row['timestamp'], row['log_level'], row['message']))
                pg_conn.commit()
        print(f"✅ Загружено {len(logs_df):,} логов")

    except Exception as e:
        print(f"❌ Ошибка при работе с PostgreSQL: {e}")
    finally:
        pg_conn.close()
else:
    print("❌ Пропуск операций с PostgreSQL из-за ошибки подключения")
```
Результат выполнения:

   <img width="800" height="300" alt="image" src="images/Снимок%20экрана%202025-10-19%20181904.png" />

#### Выполнение запроса SELECT * FROM logs ORDER BY timestamp DESC LIMIT 1000.
```
def execute_logs_sorting():
    """Выполнение запроса сортировки логов"""
    
    start_time = time.time()
    
    pg_conn = psycopg2.connect(**pg_conn_params)
    
    try:
        with pg_conn.cursor() as cur:
            query = """
            SELECT * FROM logs 
            ORDER BY timestamp DESC 
            LIMIT 1000
            """
            
            cur.execute(query)
            results = cur.fetchall()
            execution_time = time.time() - start_time
            
            # Вывод результатов
            print(f"✅ Запрос выполнен за {execution_time:.4f} секунд")
            print(f"📊 Получено {len(results)} записей")
            
            print("\nПервые 10 записей:")
            print("log_id | timestamp           | log_level | message")
            print("-" * 60)
            for row in results[:10]:
                log_id, timestamp, log_level, message = row
                print(f"{log_id:6} | {timestamp} | {log_level:8} | {message}")
                
            return results, execution_time
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ Ошибка в PostgreSQL запросе: {e}")
        return [], execution_time
    finally:
        pg_conn.close()
print("🔍 Выполнение запроса сортировки логов")
results, query_time = execute_logs_sorting()
```
Результат выполнения:

   <img width="800" height="420" alt="image" src="images/Снимок%20экрана%202025-10-19%20182038.png" />

### Шаг 4. Реализация в MongoDB
#### Подключение к MongoDB и загрузка данных:
```
try:
    mongo_client = MongoClient('mongodb://mongouser:mongopass@mongodb:27017/')
    if check_mongo_connection(mongo_client):
        print("✅ Подключение через Docker сервис 'mongodb'")
    else:
        raise Exception("Не удалось подключиться через Docker сервис")
except:
    try:
        # Если не работает через Docker, попробуем localhost
        mongo_client = MongoClient('mongodb://mongouser:mongopass@localhost:27017/')
        if check_mongo_connection(mongo_client):
            print("✅ Подключение через localhost")
        else:
            raise Exception("Не удалось подключиться через localhost")
    except:
        print("❌ Не удалось подключиться к MongoDB")
        print("Проверьте, что MongoDB запущен: docker compose ps")
        mongo_client = None

if mongo_client:
    mongo_db = mongo_client['studmongo']
    
    # Очистка существующих коллекций
    mongo_db.logs.drop()
    
    # Загрузка данных в MongoDB
    print("📥 Загрузка данных в MongoDB...")
    
    # Загрузка данных
    logs_collection = mongo_db['logs']
    logs_records = logs_df.to_dict('records')
    logs_collection.insert_many(logs_records)
    print(f"✅ Загружено {len(logs_records):,} логов")

else:
    print("❌ Пропуск операций с MongoDB из-за ошибки подключения")
```
Результат выполнения:

   <img width="800" height="420" alt="image" src="images/Снимок%20экрана%202025-10-19%20182929.png" />

#### Выполнение запроса find().sort("timestamp", -1).limit(1000)
```
def execute_mongodb_sorting():
    """Выполнение запроса сортировки логов"""
    
    start_time = time.time()
    
    try:
        # Проверяем, что подключение к MongoDB активно
        if not mongo_client:
            print("❌ Нет подключения к MongoDB")
            return [], time.time() - start_time
            
        mongo_db = mongo_client['studmongo']
        logs_collection = mongo_db['logs']
        
        # Выполняем запрос: find().sort("timestamp", -1).limit(1000)
        results = list(logs_collection.find()
                      .sort("timestamp", -1)
                      .limit(1000))
        
        execution_time = time.time() - start_time
        
        # Вывод результатов
        print(f"✅ Запрос выполнен за {execution_time:.4f} секунд")
        print(f"📊 Получено {len(results)} документов")
        
        # Выводим первые 10 документов для примера
        print("\nПервые 10 документов:")
        print("log_id | timestamp           | log_level | message")
        print("-" * 60)
        for doc in results[:10]:
            print(f"{doc['log_id']:6} | {doc['timestamp']} | {doc['log_level']:8} | {doc['message']}")
                
        return results, execution_time
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ Ошибка в MongoDB запросе: {e}")
        return [], execution_time
 
print(f"\n Выполнение запроса сортировки логов в MongoDB")
 
mongodb_results, mongodb_time = execute_mongodb_sorting()
```
Результат выполнения:

   <img width="800" height="420" alt="image" src="images/Снимок%20экрана%202025-10-19%20182938.png" />

### Шаг 5. Сравнение производительности операций сортировки
#### Сравнение производительности без индексов
```
print("📊 Сравнение производительности операций сортировки")
 
# Создание таблицы сравнения
table1 = pd.DataFrame({
    'PostgreSQL': [f"{query_time:.4f} сек"],
    'MongoDB': [f"{mongodb_time:.4f} сек"]
}, index=['Время выполнения запроса'])
 
print("\n Время выполнения запроса (без индексов)")
print(table1)
 
# График
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
dbs = ['PostgreSQL', 'MongoDB']
times = [query_time, mongodb_time]
colors = ['#1f77b4', '#ff7f0e']
 
bars = plt.bar(dbs, times, color=colors, alpha=0.7)
plt.title('Время выполнения запроса\n(без индексов)', fontsize=14, fontweight='bold')
plt.ylabel('Время (секунды)')
plt.grid(axis='y', alpha=0.3)
 
# Добавляем значения на столбцы
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}с',
             ha='center', va='bottom')
 
plt.tight_layout()
plt.show()

# Анализ результатов
print("\n📈 Анализ результатов:")
print(f"• Разница в производительности: {max(query_time, mongodb_time) / min(query_time, mongodb_time):.2f}x")
print(f"• Быстрее: {'PostgreSQL' if query_time < mongodb_time else 'MongoDB'}")
```
Результат выполнения:

   <img width="800" height="600" alt="image" src="images/Снимок%20экрана%202025-10-28%20204745.png" />

#### Добавление индексов и повторное сравнение производительности
```
print("\n🔍 Создание индексов в PostgreSQL")
pg_conn = check_postgres_connection(pg_conn_params)
if pg_conn:
    try:
        with pg_conn.cursor() as cur:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)")
        pg_conn.commit()
        print("✅ Индексы созданы в PostgreSQL")
    except Exception as e:
        print(f"❌ Ошибка создания индексов в PostgreSQL: {e}")
    finally:
        pg_conn.close()
 
# Создание индексов в MongoDB
print("🔍 Создание индексов в MongoDB")
try:
    if mongo_client:
        mongo_db = mongo_client['studmongo']
        mongo_db.logs.create_index([("timestamp", -1)])
        print("✅ Индекс создан в MongoDB")
except Exception as e:
    print(f"❌ Ошибка создания индекса в MongoDB: {e}")
 
# Повторное тестирование производительности с индексами
print("\n🔍 Тестирование производительности с индексами")
 
# PostgreSQL с индексами
print("\n📊 PostgreSQL с индексами:")
pg_results_indexed, pg_time_indexed = execute_logs_sorting()

# MongoDB с индексами  
print("\n📊 MongoDB с индексами:")
mongo_results_indexed, mongo_time_indexed = execute_mongodb_sorting()
 
# Сравнение с индексами и без
table2_data = {
    'PostgreSQL': [f"{query_time:.4f} сек", f"{pg_time_indexed:.4f} сек"],
    'MongoDB': [f"{mongodb_time:.4f} сек", f"{mongo_time_indexed:.4f} сек"]
}
table2 = pd.DataFrame(table2_data, index=['Без индексов', 'С индексами'])
 
print("\n Сравнение производительности")
print(table2)
 
# График сравнения с индексами и без с использованием Seaborn
plt.figure(figsize=(15, 6))

# Подготовка данных для Seaborn
data = []
data.append({'Database': 'PostgreSQL', 'Indexes': 'Без индексов', 'Time': query_time})
data.append({'Database': 'MongoDB', 'Indexes': 'Без индексов', 'Time': mongodb_time})
data.append({'Database': 'PostgreSQL', 'Indexes': 'С индексами', 'Time': pg_time_indexed})
data.append({'Database': 'MongoDB', 'Indexes': 'С индексами', 'Time': mongo_time_indexed})

df = pd.DataFrame(data)

# Создание группированного барплота
plt.subplot(1, 2, 1)
ax = sns.barplot(data=df, x='Database', y='Time', hue='Indexes', 
                 palette={'Без индексов': '#ff6b6b', 'С индексами': '#51cf66'},
                 alpha=0.8, gap=0.1)

plt.xlabel('База данных')
plt.ylabel('Время (секунды)')
plt.title('Сравнение производительности', fontsize=14, fontweight='bold')
plt.legend(title='Тип запроса')
plt.grid(axis='y', alpha=0.3)

# Добавляем значения на столбцы
for container in ax.containers:
    ax.bar_label(container, fmt='%.4fс', fontsize=8, padding=2)

plt.tight_layout()
plt.show()
```
Результат выполнения:

   <img width="800" height="1000" alt="image" src="images/Снимок%20экрана%202025-10-28%20205312.png" />

   <img width="700" height="500" alt="image" src="images/Снимок%20экрана%202025-10-28%20205325.png" />

# Выводы
```
# без индексов
if query_time < mongodb_time:
    faster_no_index = "PostgreSQL"
else:
    faster_no_index = "MongoDB"
 
# с индексами
if pg_time_indexed < mongo_time_indexed:
    faster_with_index = "PostgreSQL"
else:
    faster_with_index = "MongoDB"
 
# Расчет ускорения в процентах
pg_speedup = ((query_time - pg_time_indexed) / query_time) * 100
mongo_speedup = ((mongodb_time - mongo_time_indexed) / mongodb_time) * 100
 
print("📊 Анализ производительности баз данных\n")
print(f"Без индексов производительность выше у: {faster_no_index}")
print(f"С индексами производительность выше у: {faster_with_index}")
print(f"Ускорение, достигнутое в PostgreSQL: {pg_speedup:.1f}%")
print(f"Ускорение, достигнутое в MongoDB: {mongo_speedup:.1f}%")

# Итоговые выводы
print(f"\n🎯 Итоговые выводы:\n")
print(f"  • Индексация значительно повысила производительность в обеих СУБД")
print(f"  • Без индексов PostgreSQL показывает лучшие результаты")
print(f"  • Но с индексацией MongoDB демонстрирует более высокие результаты")
print(f"  • PostgreSQL эффективнее для сложных реляционных запросов, а MongoDB — для быстрых документ-ориентированных выборок")
```
Результат выполнения:

   <img width="800" height="400" alt="image" src="images/Снимок%20экрана%202025-10-28%20205301.png" />
