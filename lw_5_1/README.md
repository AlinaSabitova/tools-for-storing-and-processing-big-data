# Лабораторная работа №5.1
## Цель работы
Получить практические навыки развертывания одноузлового кластера Hadoop, освоить базовые операции с распределенной файловой системой HDFS, выполнить загрузку и простейшую обработку данных, а также научиться выгружать результаты для последующего анализа и визуализации во внешней среде.
## Индивидуальные задания
### Задание №12. Зарплаты Data Scientist
Средняя зарплата по уровню опыта
## Шаг1. Подготовка окружения
1. Клонируем репозиторий:

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

Проверим также, запустив HDFS:

<img width="974" height="354" alt="image" src="https://github.com/user-attachments/assets/1ca02ae6-6fe0-45be-a7ab-788f96209ed6" />

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
Заранее подготовим скрипт:
```
"""
Анализ зарплат IT-специалистов с использованием Pandas
Задача: найти среднюю зарплату по уровню опыта
"""
import pandas as pd
import sys
import os
import subprocess

def check_hdfs_file_exists(hdfs_path):
    """Проверить существование файла в HDFS"""
    try:
        result = subprocess.run(
            f"hdfs dfs -test -e {hdfs_path}",
            shell=True,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False

def copy_to_hdfs(local_path, hdfs_path):
    """Скопировать файл из локальной файловой системы в HDFS"""
    try:
        # Создаем директорию в HDFS если не существует
        hdfs_dir = os.path.dirname(hdfs_path)
        subprocess.run(
            f"hdfs dfs -mkdir -p {hdfs_dir}",
            shell=True,
            capture_output=True
        )

        # Копируем файл
        result = subprocess.run(
            f"hdfs dfs -put -f {local_path} {hdfs_path}",
            shell=True,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False

def load_data_from_hdfs(hdfs_path):
    """Загрузить данные из HDFS"""
    try:
        # Создаем временный локальный файл
        temp_file = "/tmp/salary_data_temp.csv"

        # Копируем файл из HDFS в локальную файловую систему
        copy_command = f"hdfs dfs -get {hdfs_path} {temp_file}"
        result = subprocess.run(copy_command, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            return None

        # Загружаем данные из временного файла
        df = pd.read_csv(temp_file, low_memory=False)

        # Удаляем временный файл
        os.remove(temp_file)

        return df

    except Exception:
        # Пытаемся удалить временный файл в случае ошибки
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except:
            pass
        return None

def load_data(filepath):
    """Загрузить данные из CSV файла (HDFS или локальный)"""
    try:
        # Загружаем из HDFS
        if filepath.startswith('/user/hadoop/'):
            return load_data_from_hdfs(filepath)
        else:
            # Локальный файл
            if os.path.exists(filepath):
                return pd.read_csv(filepath, low_memory=False)
            else:
                return None
    except Exception:
        return None

def clean_data(df):
    """Очистка и подготовка данных"""
    print("\n=== Очистка данных ===")
    print(f"Исходное количество строк: {len(df)}")

    # Удалить строки без зарплаты
    df = df[df['salary_in_usd'].notna()]

    # Заполнить пустые значения в experience_level
    df['experience_level'] = df['experience_level'].fillna('Unknown')

    # Расшифровка кодов уровней опыта
    experience_map = {
        'EN': 'Entry-level',
        'MI': 'Mid-level',
        'SE': 'Senior-level',
        'EX': 'Executive-level',
        'Unknown': 'Unknown'
    }
    df['experience_level_name'] = df['experience_level'].map(experience_map)

    print(f"Количество строк после очистки: {len(df)}")
    print(f"Уникальных уровней опыта: {df['experience_level'].nunique()}")

    return df

def analyze_salary_by_experience(df):
    """Анализ зарплаты по уровням опыта"""
    print("\n=== Анализ зарплаты по уровням опыта ===")

    # Группировка по уровню опыта и вычисление статистики по зарплате (без медианы)
    result = df.groupby(['experience_level', 'experience_level_name'])['salary_in_usd'].agg([
        'mean', 'count', 'std', 'min', 'max'
    ]).reset_index()

    result.columns = [
        'Experience_Code', 'Experience_Level', 'Mean_Salary_USD',
        'Count', 'Std_Deviation', 'Min_Salary', 'Max_Salary'
    ]

    # Сортировка по средней зарплате
    result = result.sort_values('Mean_Salary_USD', ascending=False)

    return result

def find_salary_statistics(df):
    """Статистика зарплат по уровням опыта"""
    result = analyze_salary_by_experience(df)

    print("\n=== Результаты анализа зарплат ===")
    print("\nУровни опыта по средней зарплате:")
    print(result.to_string(index=False, float_format='%.2f'))

    max_salary_level = result.iloc[0]
    min_salary_level = result.iloc[-1]

    print(f"\nСамая высокооплачиваемая категория: '{max_salary_level['Experience_Level']}' ({max_salary_level['Experience_Code']})")
    print(f"Средний доход по категории: ${max_salary_level['Mean_Salary_USD']:,.2f} USD")
    print(f"Количество специалистов данной категории: {int(max_salary_level['Count'])}")
    print(f"Диапазон зарплат в данной категории: ${max_salary_level['Min_Salary']:,.2f} - ${max_salary_level['Max_Salary']:,.2f} USD")

    print(f"\nНаименее оплачиваемая категория: '{min_salary_level['Experience_Level']}' ({min_salary_level['Experience_Code']})")
    print(f"Средний доход по категории: ${min_salary_level['Mean_Salary_USD']:,.2f} USD")
    print(f"Количество специалистов данной категории: {int(min_salary_level['Count'])}")
    print(f"Диапазон зарплат в данной категории: ${min_salary_level['Min_Salary']:,.2f} - ${min_salary_level['Max_Salary']:,.2f} USD")

    return result

def additional_analysis(df):
    """Дополнительный анализ данных"""
    print("\n=== Дополнительная статистика ===")

    # Общая статистика по зарплатам
    print(f"\nОбщая статистика зарплат:")
    print(f"Средняя зарплата: ${df['salary_in_usd'].mean():,.2f}")
    print(f"Минимальная зарплата: ${df['salary_in_usd'].min():,.2f}")
    print(f"Максимальная зарплата: ${df['salary_in_usd'].max():,.2f}")

    # Распределение по уровням опыта
    print(f"\nРаспределение по уровням опыта:")
    experience_counts = df['experience_level_name'].value_counts()
    for level, count in experience_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {level}: {count} специалистов ({percentage:.1f}%)")

def main():
    # Используем точный путь
    data_file = '/user/hadoop/input/salary_data.csv'

    print("=== Анализ зарплат в Data Science ===")

    # Проверим файл в HDFS
    if not check_hdfs_file_exists(data_file):
        print(f"Ошибка: Файл не найден в HDFS: {data_file}")
        sys.exit(1)

    # Загрузка данных
    df = load_data(data_file)

    if df is None:
        print("Ошибка: Не удалось загрузить данные из HDFS")
        sys.exit(1)

    print(f"Успешно загружено строк: {len(df)}")

    # Очистка данных
    df_clean = clean_data(df)

    # Основной анализ
    result = find_salary_statistics(df_clean)

    # Дополнительный анализ
    additional_analysis(df_clean)

    # Сохранить результаты локально
    local_output_file = 'results/salary_by_experience.csv'
    os.makedirs('results', exist_ok=True)
    result.to_csv(local_output_file, index=False)
    print(f"\nРезультаты сохранены локально в: {local_output_file}")

    # Сохранить результаты в HDFS
    hdfs_output_file = '/user/hadoop/output/salary_by_experience.csv'
    if copy_to_hdfs(local_output_file, hdfs_output_file):
        print(f"Результаты также сохранены в HDFS: {hdfs_output_file}")
    else:
        print("Не удалось сохранить результаты в HDFS")

    return result

if __name__ == '__main__':
    main()
```
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
1. Заранее подготовим скрипт:
```
"""
Анализ зарплат в Data Science с использованием PySpark
Задача: найти среднюю зарплату по уровню опыта
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, stddev, min as spark_min, max as spark_max
from pyspark.sql.functions import when
import sys
import subprocess
import os

def check_local_file_exists(local_path):
    """Проверить существование файла в локальной файловой системе"""
    return os.path.exists(local_path)

def copy_to_hdfs(local_path, hdfs_path):
    """Скопировать файл из локальной файловой системы в HDFS"""
    try:
        # Создаем директорию в HDFS если не существует
        hdfs_dir = os.path.dirname(hdfs_path)
        subprocess.run(
            f"hdfs dfs -mkdir -p {hdfs_dir}",
            shell=True,
            capture_output=True
        )

        # Копируем файл
        result = subprocess.run(
            f"hdfs dfs -put -f {local_path} {hdfs_path}",
            shell=True,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False

def create_spark_session():
    """Создать Spark сессию"""
    spark = SparkSession.builder \
        .appName("Salary Analysis") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()
    return spark

def load_data(spark, filepath):
    """Загрузить данные из локального файла"""
    try:
        # Явно указываем, что это локальный файл с префиксом file://
        local_file_path = f"file://{filepath}"
        df = spark.read.csv(local_file_path, header=True, inferSchema=True)
        return df
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

def clean_and_prepare(df):
    """Очистка и подготовка данных"""
    print("\n=== Очистка данных ===")
    print(f"Исходное количество строк: {df.count()}")

    # Удалить строки с null в зарплате
    df = df.filter(col('salary_in_usd').isNotNull())

    # Заполнить null в experience_level
    df = df.na.fill('Unknown', subset=['experience_level'])

    # Создать колонку с полными названиями уровней опыта
    df = df.withColumn(
        'experience_level_name',
        when(col('experience_level') == 'EN', 'Entry-level')
        .when(col('experience_level') == 'MI', 'Mid-level')
        .when(col('experience_level') == 'SE', 'Senior-level')
        .when(col('experience_level') == 'EX', 'Executive-level')
        .otherwise('Unknown')
    )

    print(f"Количество строк после очистки: {df.count()}")
    distinct_levels = df.select('experience_level').distinct().count()
    print(f"Уникальных уровней опыта: {distinct_levels}")

    return df

def analyze_salary_by_experience(df):
    """Анализ средней зарплаты по уровням опыта"""
    print("\n=== Анализ средней зарплаты по уровням опыта ===")

    # Группировка и агрегация
    result = df.groupBy('experience_level', 'experience_level_name') \
        .agg(
            avg('salary_in_usd').alias('Mean_Salary_USD'),
            count('*').alias('Count'),
            stddev('salary_in_usd').alias('Std_Deviation'),
            spark_min('salary_in_usd').alias('Min_Salary'),
            spark_max('salary_in_usd').alias('Max_Salary')
        ) \
        .orderBy(col('Mean_Salary_USD').desc())

    return result

def additional_analysis(df):
    """Дополнительный анализ данных"""
    print("\n=== Дополнительная статистика ===")

    # Общая статистика по зарплатам
    salary_stats = df.agg(
        avg('salary_in_usd').alias('avg_salary'),
        spark_min('salary_in_usd').alias('min_salary'),
        spark_max('salary_in_usd').alias('max_salary')
    ).collect()[0]

    print(f"Общая статистика зарплат в USD:")
    print(f"Средняя зарплата: ${salary_stats['avg_salary']:,.2f}")
    print(f"Минимальная зарплата: ${salary_stats['min_salary']:,.2f}")
    print(f"Максимальная зарплата: ${salary_stats['max_salary']:,.2f}")

    # Распределение по уровням опыта
    print(f"\nРаспределение по уровням опыта:")
    experience_counts = df.groupBy('experience_level_name').agg(count('*').alias('count')).collect()
    total_count = df.count()

    for row in experience_counts:
        percentage = (row['count'] / total_count) * 100
        print(f"  {row['experience_level_name']}: {row['count']} записей ({percentage:.1f}%)")

def save_spark_results(result, local_output_path, hdfs_output_dir):
    """Сохранить результаты Spark и скопировать в HDFS"""
    try:
        # Создаем локальную директорию
        os.makedirs("results", exist_ok=True)

        # Сохраняем как единый CSV файл через Pandas
        pandas_df = result.toPandas()
        local_csv_file = f"{local_output_path}.csv"
        pandas_df.to_csv(local_csv_file, index=False)
        print(f"Результаты сохранены локально в: {local_csv_file}")

        # Копируем в HDFS
        hdfs_csv_file = f"{hdfs_output_dir}/salary_by_experience_spark.csv"
        if copy_to_hdfs(local_csv_file, hdfs_csv_file):
            print(f"Результаты также сохранены в HDFS: {hdfs_csv_file}")
            return True
        else:
            print("Не удалось сохранить результаты в HDFS")
            return False

    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")
        return False

def main():
    # Локальный путь к данным
    local_path = "/opt/data/salary_data.csv"

    # Создать Spark сессию
    spark = create_spark_session()

    print("=== Анализ зарплат в Data Science с использованием PySpark ===")

    # Проверим файл локально
    if not check_local_file_exists(local_path):
        print(f"Ошибка: Файл не найден локально: {local_path}")
        spark.stop()
        sys.exit(1)

    print(f"Файл найден локально, размер: {os.path.getsize(local_path)} байт")

    # Загрузить данные из локального файла
    df = load_data(spark, local_path)

    if df is None:
        print("Ошибка: Не удалось загрузить данные")
        spark.stop()
        sys.exit(1)

    print(f"Успешно загружено строк: {df.count()}")

    # Очистка данных
    df_clean = clean_and_prepare(df)

    # Анализ
    result = analyze_salary_by_experience(df_clean)

    # Показать результаты
    print("\n=== Результаты анализа зарплат ===")
    print("\nУровни опыта по средней зарплате:")
    result.show(truncate=False)

    # Найти уровень с максимальной и минимальной зарплатой
    result_list = result.collect()
    max_salary_row = result_list[0]
    min_salary_row = result_list[-1]

    print(f"\nСамая высокооплачиваемая категория: '{max_salary_row['experience_level_name']}' ({max_salary_row['experience_level']})")
    print(f"Средний доход по категории: ${max_salary_row['Mean_Salary_USD']:,.2f} USD")
    print(f"Количество специалистов данной категории: {max_salary_row['Count']}")
    print(f"Диапазон зарплат в данной категории: ${max_salary_row['Min_Salary']:,.2f} - ${max_salary_row['Max_Salary']:,.2f} USD")

    print(f"\nНаименее оплачиваемая категория: '{min_salary_row['experience_level_name']}' ({min_salary_row['experience_level']})")
    print(f"Средний доход по категории: ${min_salary_row['Mean_Salary_USD']:,.2f} USD")
    print(f"Количество специалистов данной категории: {min_salary_row['Count']}")
    print(f"Диапазон зарплат в данной категории: ${min_salary_row['Min_Salary']:,.2f} - ${min_salary_row['Max_Salary']:,.2f} USD")

    # Дополнительный анализ
    additional_analysis(df_clean)

    # Сохранить результаты
    local_output_path = "results/salary_by_experience_spark"
    hdfs_output_dir = "/user/hadoop/output"

    save_spark_results(result, local_output_path, hdfs_output_dir)

    spark.stop()

if __name__ == '__main__':
    main()
```

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

2. 
