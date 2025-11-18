# Лабораторная работа №5.1
## Цель работы
Получить практические навыки развертывания одноузлового кластера Hadoop, освоить базовые операции с распределенной файловой системой HDFS, выполнить загрузку и простейшую обработку данных, а также научиться выгружать результаты для последующего анализа и визуализации во внешней среде.
## Индивидуальные задания
### Задание №12. Зарплаты Data Scientist
Средняя зарплата по уровню опыта
## Шаг 1. Подготовка окружения
#### 1. Клонируем репозиторий:

```
git clone <repository_url>
```

<img width="974" height="159" alt="image" src="https://github.com/user-attachments/assets/4190ba66-4459-4f37-b8ef-ec0e6f5bf7a9" />

2. Проверим, что все файлы загрузились:
```
ls -la hadoop/
ls -la scripts/
ls -la notebooks/
```

<img width="974" height="235" alt="image" src="https://github.com/user-attachments/assets/8f442725-6e09-4d2a-928a-25e02a96fa7b" />

<img width="974" height="149" alt="image" src="https://github.com/user-attachments/assets/231c66ac-2423-49c1-861e-29440ef5317d" />

<img width="974" height="149" alt="image" src="https://github.com/user-attachments/assets/f5a6df0b-2bb3-4d7f-bc06-3590b5925b2d" />

3. Запустим докер:
```
docker compose up -d
```

<img width="974" height="323" alt="image" src="https://github.com/user-attachments/assets/dba17e51-ddea-4797-8c1d-fc1f52c74282" />

<img width="974" height="298" alt="image" src="https://github.com/user-attachments/assets/08e8808d-4711-483d-9f9b-bc209bbef9e5" />

4. Посмотрим логи:
```
docker compose logs -f hadoop
```

<img width="974" height="312" alt="image" src="https://github.com/user-attachments/assets/06707d57-bc95-44e7-bcc4-9254ad4dd06a" />

5. Откроем терминал внутри контейнера:
```
docker compose exec hadoop bash
```

<img width="974" height="84" alt="image" src="https://github.com/user-attachments/assets/9bece2e3-8652-42e1-8372-d1f4218337d5" />

6. Проверим компоненты:

<img width="959" height="278" alt="image" src="https://github.com/user-attachments/assets/a9cc938c-1fbd-4e3c-a6ae-2cd23b9fb9c5" />

## Шаг2. Работа с HDFS
1. Создаем директории и проверяем, что они были созданы:
```
# Создать директории для входных и выходных данных
hdfs dfs -mkdir -p /user/hadoop/input
hdfs dfs -mkdir -p /user/hadoop/output

# Проверить созданные директории
hdfs dfs -ls /user/hadoop/
```

<img width="974" height="171" alt="image" src="https://github.com/user-attachments/assets/e39823f6-586c-4137-b103-56cc9adda57a" />

3. Загрузим данные вручную ввиду большого объема файла:

<img width="974" height="354" alt="image" src="https://github.com/user-attachments/assets/d1cc71fb-2075-4280-972e-b90e14ee7c25" />

4. Проверим загрузку и посмотрим размер файла

```
# Проверить загрузку
hdfs dfs -ls -h /user/hadoop/input/

# Просмотреть размер файла
hdfs dfs -du -h /user/hadoop/input/
```

<img width="974" height="112" alt="image" src="https://github.com/user-attachments/assets/01a5f478-83c2-4ee1-a6e3-b6f9097604d6" />

5. Просмотрим первые строки файла
```
hdfs dfs -cat /user/hadoop/input/database.csv | head -20
```

<img width="974" height="323" alt="image" src="https://github.com/user-attachments/assets/6a40ba56-675b-4744-9589-dbfdd1d08fe8" />

6. Посмотрим статистику в HDFS
```
hdfs dfsadmin -report
```

<img width="974" height="401" alt="image" src="https://github.com/user-attachments/assets/8ea65d31-39fd-400d-9271-961786555e8d" />

<img width="974" height="402" alt="image" src="https://github.com/user-attachments/assets/ab40b1bc-85cf-4d46-b102-9a9921155f1e" />

7. Откроем yarn:

<img width="974" height="294" alt="image" src="https://github.com/user-attachments/assets/7c2a19ad-6a79-4f6c-82f2-3a2c8d9d4a08" />

## Шаг3. Анализ в Pandas
1. Заранее подготовим [скрипт](/lw_5_1/scripts/analyze_pandas.py)

2. Запустим анализ:
```
python3 analyze_pandas.py
```

<img width="974" height="416" alt="image" src="https://github.com/user-attachments/assets/931a1ded-b76c-415e-a9a0-8fdc81b5357b" />

<img width="974" height="131" alt="image" src="https://github.com/user-attachments/assets/cd47c9d4-6413-4a01-8632-820eef5d5e59" />

3. Проверим, что файл с анализом сохранился локально:
```
cat results/salary_by_experience.csv
```

<img width="974" height="143" alt="image" src="https://github.com/user-attachments/assets/215e29e4-9c7a-4ff7-bf7a-62e1bc1c67d7" />

## Шаг3. Анализ в Spark
1. Заранее подготовим [скрипт](/lw_5_1/scripts/analyze_spark.py)

2. Запустим его:
```
python3 analyze_spark.py
```

<img width="974" height="454" alt="image" src="https://github.com/user-attachments/assets/32b5e31c-1882-4535-82a2-259dc25ee0ae" />

<img width="974" height="183" alt="image" src="https://github.com/user-attachments/assets/259d7b73-5de4-4989-9390-3b0618b4e572" />

3. Проверим, что результаты сохранились в HDFS
```
hdfs dfs -ls /user/hadoop/output
hdfs dfs -cat /user/hadoop/output/magnitude_by_type/part-00000 | head -20
```

<img width="974" height="69" alt="image" src="https://github.com/user-attachments/assets/ed5dc02c-972f-46fa-b309-6ce4d8d8844d" />

<img width="974" height="113" alt="image" src="https://github.com/user-attachments/assets/2a048b62-6793-4608-8884-a837fa39b227" />

Откроем также HDFS в браузере:

<img width="974" height="357" alt="image" src="https://github.com/user-attachments/assets/ac3d90fc-55f5-4f27-88ea-fb63fce991d7" />

Видим, что файлы действительно были сохранены

## Шаг4. Анализ в Jupyter Notebook

1. Запустим Jupyter Notebook
   
```
bash scripts/start_jupyter.sh
```

<img width="974" height="232" alt="image" src="https://github.com/user-attachments/assets/2303d8c0-cf52-4a61-bd77-3dcce600d70a" />

2. Загрузим данные из HDFS
3. 
Подключимся:
```
pip install pandas numpy matplotlib seaborn
```

<img width="974" height="352" alt="image" src="https://github.com/user-attachments/assets/8edb589e-d7d2-42b0-bf8a-14974509db9c" />

Установим нужные библиотеки:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import os

# Настройка отображения
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
%matplotlib inline

# Увеличение размера графиков
plt.rcParams['figure.figsize'] = (12, 8)
```
Скачаем файл из HDFS:
```
import os
import subprocess
import pandas as pd

# Загрузка данных из HDFS
print("Загрузка данных из HDFS...")

# Путь к данным в HDFS (правильный путь)
hdfs_path = "/user/hadoop/input/salary_data.csv"
local_path = "/opt/salary_data.csv"

# Скачиваем файл из HDFS
try:
    # Команда для скачивания из HDFS
    hdfs_download_cmd = f"hdfs dfs -get {hdfs_path} {local_path}"
    print(f"Выполнение команды: {hdfs_download_cmd}")
    
    # Задаем JAVA_HOME, так как Jupyter может его не видеть
    env = dict(os.environ, **{'JAVA_HOME': '/usr/lib/jvm/java-11-openjdk-amd64'})
    result = subprocess.run(hdfs_download_cmd, shell=True, capture_output=True, text=True, cwd="/opt", env=env)
    
    if result.returncode == 0:
        print(f"Данные успешно загружены из HDFS: {hdfs_path}")
    else:
        print(f"Ошибка при загрузке из HDFS: {result.stderr}")
        print("Попытка найти файл локально...")
        local_path = "/opt/data/salary_data.csv"

    # Проверяем наличие файла
    if not os.path.exists(local_path):
        print(f"Файл не найден в {local_path}. Используем альтернативный путь...")
        # Исправлена синтаксическая ошибка
        local_path = "salary_data.csv"
except Exception as e:
    print(f"Ошибка при выполнении subprocess: {e}")
    print("Попытка использовать локальный файл...")
    local_path = "/opt/data/salary_data.csv"

# Проверяем наличие файла перед загрузкой
if not os.path.exists(local_path):
    print("Файл не найден. Пробуем последний вариант...")
    local_path = "salary_data.csv"

# Финальная проверка и загрузка данных
if os.path.exists(local_path):
    df = pd.read_csv(local_path, low_memory=False)
    print(f"Размер датасета: {df.shape}")
    print(f"Данные успешно загружены из {local_path}")
    print(df.head())
else:
    print(f"ОШИБКА: Файл database.csv не найден!")
    print(f"Искали по следующим путям:")
    print(f"  - /opt/database.csv (из HDFS)")
    print(f"  - /opt/data/database.csv (локальный)")
    print(f"  - database.csv (в текущей директории)")
    df = pd.DataFrame()  # Создаем пустой DataFrame чтобы не было ошибки
```

<img width="974" height="385" alt="image" src="https://github.com/user-attachments/assets/b125b9b6-9810-4867-88dd-403509d3f74c" />

Очистим данные перед анализом:
```
# Очистка данных
df_clean = df.copy()
df_clean = df_clean[df_clean['salary'].notna()]
df_clean['experience_level'] = df_clean['experience_level'].fillna('Unknown')

print(f"Количество строк: {len(df_clean)}")
print(f"Уровни опыта: {df_clean['experience_level'].unique()}")
```

<img width="974" height="52" alt="image" src="https://github.com/user-attachments/assets/12ade80b-fbc1-4e9a-b8e7-73e3465c7af0" />

2. Анализ зарплат по уровню опыта

```
# Группировка по уровню опыта и вычисление средней зарплаты
salary_by_experience = df_clean.groupby('experience_level')['salary_in_usd'].agg(['mean', 'count']).reset_index()
salary_by_experience.columns = ['Experience_Level', 'Mean_Salary', 'Count']
salary_by_experience['Mean_Salary'] = salary_by_experience['Mean_Salary'].round(2)
salary_by_experience = salary_by_experience.sort_values('Mean_Salary', ascending=False)
 
print("Средняя зарплата по уровням опыта:")
print(salary_by_experience)
```

<img width="974" height="109" alt="image" src="https://github.com/user-attachments/assets/7b528cb1-708c-4c79-9a2a-3d89cc3e5d72" />

```
# Результат
max_experience = salary_by_experience.iloc[0]
print(f"Уровень опыта с максимальной средней зарплатой: {max_experience['Experience_Level']}")
print(f"Средняя зарплата для данного уровня опыта: {max_experience['Mean_Salary']:.2f}$")
print(f"Количество специалистов данного уровня опыта: {int(max_experience['Count'])}")
```

<img width="974" height="69" alt="image" src="https://github.com/user-attachments/assets/8c0beca8-a744-434c-9606-a9bdcceb5c2d" />

3. Визуализация
   
Установим HDFS:

<img width="974" height="511" alt="image" src="https://github.com/user-attachments/assets/a28f0128-9833-4756-8321-17c46f7d0b29" />

Построим график сравнения средней зарплаты по уровню опыта:
```
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from hdfs import InsecureClient
 
# --- 1. Подготовка данных и сортировка ---
df_sorted = salary_by_experience.sort_values('Mean_Salary', ascending=False)
 
# --- 2. Создание графика ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 7))
 
ax = sns.barplot(
    x='Mean_Salary',
    y='Experience_Level',
    hue='Experience_Level',
    data=df_sorted,
    palette='Blues_r',
    legend=False
)
 
# "Приближаем" ось X для акцентирования разницы
min_val = df_sorted['Mean_Salary'].min()
plt.xlim(left=min_val - 10000)

# Добавляем метки со значениями
for bar in ax.patches:
    ax.text(
        bar.get_width() + 1000,
        bar.get_y() + bar.get_height() / 2,
        f'${bar.get_width():,.0f}',
        va='center', ha='left',
        fontsize=12, color='black',
        weight='bold'
    )
# Настройка заголовков и осей
plt.title('Сравнение средней зарплаты по уровням опыта', fontsize=16, pad=20)
plt.xlabel('Средняя зарплата', fontsize=12)
plt.ylabel('Уровень опыта', fontsize=12)
sns.despine(left=True, bottom=True)
plt.tight_layout()
 
# --- 3. Сохранение графика в буфер памяти ---
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=300)
# Показываем график в ноутбуке
plt.show()

buffer.seek(0)
 
# --- 4. Подключение к HDFS и перезапись файла ---
hdfs_path = '/user/hadoop/results/salary_by_experience.png'

# Подключаемся к HDFS
client = InsecureClient('http://hadoop:9870', user='root')
 
# Записываем данные из буфера в HDFS с флагом перезаписи
with client.write(hdfs_path, overwrite=True) as writer:
    writer.write(buffer.getvalue())
 
print(f"График успешно перезаписан в HDFS по пути: {hdfs_path}")
```

<img width="974" height="577" alt="image" src="https://github.com/user-attachments/assets/809df481-9c31-407d-a38d-0b422c9b214d" />

Построим график "Топ-3 самые высокооплачиваемые должности":
```
# --- 1. Подготовка данных ---
top_jobs = df_clean.groupby('job_title')['salary_in_usd'].mean().nlargest(3)
top_jobs_df = top_jobs.reset_index()
top_jobs_df.columns = ['Job_Title', 'Mean_Salary_USD']
top_jobs_df['Mean_Salary_USD'] = top_jobs_df['Mean_Salary_USD'].round(2)
top_jobs_sorted = top_jobs_df.sort_values('Mean_Salary_USD', ascending=False)
 
print("Топ-3 зарплаты:")
print(top_jobs_sorted)
 
# --- 2. Создание графика ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
 
ax = sns.barplot(
    x='Job_Title',
    y='Mean_Salary_USD',
    hue='Job_Title',
    data=top_jobs_sorted,
    palette='Set2',
    legend=False
)
 
plt.ylim(bottom=217000, top=218200)
 
# Добавляем метки со значениями
for i, bar in enumerate(ax.patches):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50, 
        f'${bar.get_height():,.0f}',
        va='bottom', ha='center',
        fontsize=12, color='black',
        weight='bold'
    )

# Настройка заголовков и осей
plt.title('Топ-3 самых высокооплачиваемых должностей', fontsize=16, pad=20)
plt.xlabel('Должность', fontsize=12)
plt.ylabel('Средняя зарплата (USD)', fontsize=12)
plt.xticks(rotation=0)
 
plt.grid(True, axis='y', alpha=0.3, linestyle='--')
 
plt.yticks(np.arange(217200, 218250, 100))
 
sns.despine(left=True, bottom=True)
plt.tight_layout()
 
# --- 3. Сохранение графика в буфер памяти ---
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=300)
plt.show()
buffer.seek(0)
 
# --- 4. Подключение к HDFS и перезапись файла ---
hdfs_path = '/user/hadoop/results/top_3_jobs.png'
 
# Подключаемся к HDFS
client = InsecureClient('http://hadoop:9870', user='root')
 
# Записываем данные из буфера в HDFS с флагом перезаписи
with client.write(hdfs_path, overwrite=True) as writer:
    writer.write(buffer.getvalue())
 
print(f"График успешно перезаписан в HDFS по пути: {hdfs_path}")
```

<img width="974" height="611" alt="image" src="https://github.com/user-attachments/assets/59236682-da0b-4249-a6c3-d67334526c9f" />

Построим график, отражающий динамику средней зарплаты по годам:
```
# --- 1. Подготовка данных ---
salary_by_year = df_clean.groupby('work_year')['salary_in_usd'].mean().reset_index()
salary_by_year.columns = ['Year', 'Mean_Salary_USD']
salary_by_year['Mean_Salary_USD'] = salary_by_year['Mean_Salary_USD'].round(2)
salary_by_year_sorted = salary_by_year.sort_values('Year', ascending=True)
 
print("Средние зарплаты по годам:")
print(salary_by_year_sorted)
 
# --- 2. Создание линейчатой диаграммы ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))
 
# Создаем линейчатую диаграмму
plt.plot(salary_by_year_sorted['Year'], salary_by_year_sorted['Mean_Salary_USD'], 
         marker='o', linewidth=3, markersize=8, color='#2E86AB')
 
# Шкалирование для узкого диапазона 155-156 тысяч
min_salary = salary_by_year_sorted['Mean_Salary_USD'].min()
max_salary = salary_by_year_sorted['Mean_Salary_USD'].max()
 
# Очень узкий диапазон для подчеркивания разницы
plt.ylim(155400, 156300)
 
# Добавляем метки со значениями
for i, (year, salary) in enumerate(zip(salary_by_year_sorted['Year'], salary_by_year_sorted['Mean_Salary_USD'])):
    plt.text(year, salary + 150, f'${salary:,.2f}', 
             ha='center', va='bottom', fontsize=12, weight='bold', color='black')
 
# Настройка заголовков и осей
plt.title('Динамика средних зарплат по годам', fontsize=16, pad=20)
plt.xlabel('Год', fontsize=12)
plt.ylabel('Средняя зарплата (USD)', fontsize=12)
 
# Улучшаем сетку для лучшего восприятия разницы
plt.grid(True, alpha=0.3, linestyle='--')
 
# Частые деления на оси Y для точного сравнения
plt.xticks(salary_by_year_sorted['Year'])
plt.yticks(np.arange(155500, 156500, 500))
 
sns.despine(left=True, bottom=True)
plt.tight_layout()
 
# --- 3. Сохранение графика в буфер памяти ---
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=300)
# Показываем график в ноутбуке
plt.show()
# "Перематываем" буфер в начало для чтения
buffer.seek(0)
 
# --- 4. Подключение к HDFS и перезапись файла ---
hdfs_path = '/user/hadoop/results/salary_by_year.png'
 
# Подключаемся к HDFS
client = InsecureClient('http://hadoop:9870', user='root')
 
# Записываем данные из буфера в HDFS с флагом перезаписи
with client.write(hdfs_path, overwrite=True) as writer:
    writer.write(buffer.getvalue())
 
print(f"График успешно перезаписан в HDFS по пути: {hdfs_path}")
```

<img width="974" height="595" alt="image" src="https://github.com/user-attachments/assets/6f0967dc-0168-46c2-abed-f30f05647274" />

## Шаг 5. Простые скрипты на bash

Напишим простые запросы для подсчета строк в файле, выводу последних десяти строк и подсчета уникальных уровней опыта:

<img width="955" height="332" alt="Снимок экрана 2025-11-18 133556" src="https://github.com/user-attachments/assets/ea2fd189-3c85-44f4-bf2b-43daa51a62ff" />
