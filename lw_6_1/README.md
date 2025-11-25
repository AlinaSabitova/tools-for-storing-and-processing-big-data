# Анализ и визуализация больших данных. Машинное обучение на больших данных с использованием Apache Spark MLlib
## Цель и задачи работы:
- Познакомиться с понятием «большие данные» и способами их обработки;
  
- Познакомиться с инструментом Apache Spark и возможностями, которые он предоставляет для обработки больших данных.

 - Получить навыки выполнения разведочного анализа данных использованием pyspark.

## Выполнение работы в Google Colab
### 1. Инициализация платформы Spark и загрузка данных в фрейм данных Spark

Импорт модулей, не связанных с PySpark:

<img width="1833" height="734" alt="image" src="https://github.com/user-attachments/assets/b3abf352-8514-4e08-ba5e-55c49de7d6ba" />

Подключение к гугл диску:

<img width="1799" height="647" alt="image" src="https://github.com/user-attachments/assets/009ae1ae-cc33-4452-99c9-4e66f073f73f" />

Импорт модулей, связанных с PySpark:

<img width="1843" height="736" alt="image" src="https://github.com/user-attachments/assets/4483fd1f-fb64-44dc-bebe-7198dbc3f3a9" />

<img width="1806" height="249" alt="image" src="https://github.com/user-attachments/assets/9c50368f-7613-482d-85a5-1b2076d62b94" />

### 2. Обзор набора данных
Обзор данных и столбцов:

<img width="1829" height="573" alt="image" src="https://github.com/user-attachments/assets/cba8ae9d-c6a8-4156-b8a9-5cfea7e2e20d" />

<img width="1003" height="511" alt="image" src="https://github.com/user-attachments/assets/bad5b770-5db1-4d17-b649-d6eb4a196acd" />

Описание фрейма данных:

<img width="1823" height="709" alt="image" src="https://github.com/user-attachments/assets/0c4cd42c-193e-4356-9a8f-076d5e836c79" />

<img width="1764" height="211" alt="image" src="https://github.com/user-attachments/assets/22449da0-92ae-4654-9fee-dd0b4e684a28" />

### 3. Обнаружение пропущенных значений и аномальных нулей.
Обзор столбцов:

<img width="1802" height="626" alt="image" src="https://github.com/user-attachments/assets/d1b6e10b-04c6-413a-acd9-c3f37d308361" />

Проверка столбцов на наличие NaN значений:

<img width="1259" height="462" alt="image" src="https://github.com/user-attachments/assets/d3436fef-f72f-4dd7-9862-11fa77b6332a" />

Получение общей сводки данных: 

<img width="1279" height="580" alt="image" src="https://github.com/user-attachments/assets/1fc8eb28-d7cb-4ca0-bd67-dc7d6dd2c4ed" />

<img width="1216" height="117" alt="image" src="https://github.com/user-attachments/assets/8e9e6b80-f7f5-4f67-a6e6-7e21a7d57c9d" />

Вывод статистики по тренировкам, число записей которых превышает 50:

<img width="1286" height="248" alt="image" src="https://github.com/user-attachments/assets/145062a9-995b-4764-9d8f-11e86cb137f5" />

### 4. Ленивая оценка Pyspark
Топ-5 тренировок по количеству пользователей:

<img width="1766" height="521" alt="image" src="https://github.com/user-attachments/assets/f8a2cead-f447-4f96-acb7-911829089da7" />

### 5. Исследовательский анализ данных
Создание таблицы и графиков для 5 видов спорта, которыми занимается наибольшее количество пользователей:

```
highest_sport_users_df_renamed = highest_sport_users_df.copy()

# Calculate the percentage for the top 5 sports
highest_sport_users_df_renamed['percentage'] = highest_sport_users_df_renamed['Users count'] / total_sports_users * 100

# Creating the 'others' group
others = {
    'sport': 'others',
    'Users count': total_sports_users - sum(highest_sport_users_df_renamed['Users count']),
    'percentage': 100 - sum(highest_sport_users_df_renamed['percentage'])
}

# Convert 'others' into a DataFrame
others_df = pd.DataFrame([others])

# Use pd.concat() to append 'others' to the original DataFrame
highest_sport_users_df_renamed = pd.concat([highest_sport_users_df_renamed, others_df], ignore_index=True)

print('Топ-5 видов спорта, в которых участвует больше всего пользователей:')
print(highest_sport_users_df_renamed)

# Plotting
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=plt.figaspect(0.35))

# Bar plot
axs[0].bar(x=highest_sport_users_df_renamed['sport'], height=highest_sport_users_df_renamed['Users count'])
axs[0].set_title('Вид спортивных увлечений', fontsize='small')
axs[0].set_xlabel('Вид спорта', fontsize='small')
axs[0].set_ylabel('Количество пользователей', fontsize='small')
axs[0].set_xticklabels(highest_sport_users_df_renamed['sport'], rotation='vertical', fontsize='small')

# Pie chart
explode = (0.1, 0.1, 0.3, 0.3, 0.3, 0.1)
axs[1].pie(
    x=highest_sport_users_df_renamed['percentage'],
    labels=highest_sport_users_df_renamed['sport'],
    autopct='%1.1f%%', shadow=True, explode=explode, startangle=90, radius=1
)
axs[1].set_title('Вид спортивных увлечений', fontsize='small')

# Adding text to the figure
fig.text(0.5, 1.02, 'Топ-5 видов спорта, в которых участвует больше всего пользователей', ha='center', va='top', transform=fig.transFigure)

# Show the plot
plt.show()
```

Результат: 

<img width="1692" height="690" alt="image" src="https://github.com/user-attachments/assets/1077ec6b-375c-443e-8e7d-47f0aea4c55d" />

<img width="1484" height="723" alt="image" src="https://github.com/user-attachments/assets/99477f5f-e86d-4bc1-a353-009c88226234" />

Анализ зависимости от пола пользователей:

<img width="1389" height="362" alt="image" src="https://github.com/user-attachments/assets/59eff075-16c9-46a5-87b8-aee3c828f3d3" />

### 6. UNSTACK PYSPARK DATAFRAME

Вычисления процентов участия в тренировках мужчин и женщин для всех видов спорта:

```
total_activities = ranked_sport_users_df.count()
print(f'Всего: {total_activities} активности в зависимости от пола:')
# Добавление информации о занятиях в зависимости от пола
activities_by_gender = df.groupBy('sport', 'gender').count().toPandas()
# Визуализация
fig = plt.figure(figsize=(12, 25))
grid_size = (1,1);
ax = plt.subplot2grid(grid_size, (0,0), colspan=1, rowspan=1)
plot = activities_by_gender.groupby(['sport', 'gender']).agg(np.mean).groupby(level=0).apply(
    lambda x: 100 * x / x.sum()).unstack().plot(kind='barh', stacked=True, width=1  ## APPLY UNSTACK TO RESHAPE DATA
                , edgecolor='black', ax=ax, title='Список всех занятий в зависимости от пола')
ylabel = plt.ylabel('Sport (Activity)');
xlabel = plt.xlabel('Процент участия по гендерному признаку');
legend = plt.legend(
    sorted(activities_by_gender['gender'].unique()), loc='center left', bbox_to_anchor=(1.0, 0.5)
)
param_update = plt.rcParams.update({'font.size': 16});
ax = plt.gca()
formatter = ax.xaxis.set_major_formatter(mtick.PercentFormatter());
a = fig.tight_layout()
plt.show()
```

Результат:

<img width="1715" height="695" alt="image" src="https://github.com/user-attachments/assets/e86c463c-70ac-4c74-b0ed-a40b3e428b50" />

<img width="1593" height="690" alt="image" src="https://github.com/user-attachments/assets/9566c4d6-9c1d-41f4-820d-889e63f95184" />

Найдем топ-5 видов спорта по числу тренировок:

```
# Pivot the data by gender and sport
activities_by_gender_df = activities_by_gender.pivot_table(
    index="sport", columns="gender", values='count', fill_value=0) \
    .reset_index().rename_axis(None, axis=1)

# Add total count and percentage
activities_by_gender_df['total'] = activities_by_gender_df['male'] + activities_by_gender_df['female'] + activities_by_gender_df['unknown']
activities_by_gender_df['percentage'] = activities_by_gender_df['total'] / sum(activities_by_gender_df['total']) * 100

# Get top 5 activities by percentage
top_activities_by_gender_df = activities_by_gender_df.sort_values(by='percentage', ascending=False).head(5)

# Create the 'others' group
others = {'sport': 'others'}
for column in ['female', 'male', 'unknown', 'total', 'percentage']:
    value = sum(activities_by_gender_df[column]) - sum(top_activities_by_gender_df[column])
    others.update({column: value})

# Convert 'others' to a DataFrame
others_df = pd.DataFrame([others])

# Use pd.concat to add 'others' row to the DataFrame
top_activities_by_gender_df = pd.concat([top_activities_by_gender_df, others_df], ignore_index=True)

# Sort by percentage again
top_activities_by_gender_df = top_activities_by_gender_df.sort_values(by='percentage', ascending=False)

# Display the final DataFrame
print(top_activities_by_gender_df)

# Plotting
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=plt.figaspect(0.35))

# Bar plot for total count of activities
axs[0].bar(x=top_activities_by_gender_df['sport'], height=top_activities_by_gender_df['total'])
axs[0].set_title('Количество тренировок', fontsize='small')
axs[0].set_xlabel('Спорт', fontsize='small')
axs[0].set_ylabel('Количество тренировок (раз)', fontsize='small')
axs[0].set_xticklabels(top_activities_by_gender_df['sport'], rotation='vertical', fontsize='small')

# Pie chart for percentage distribution
explode = (0.1, 0.1, 0.3, 0.3, 0.3, 0.3)
axs[1].pie(
    x=top_activities_by_gender_df['percentage'],
    labels=top_activities_by_gender_df['sport'],
    autopct='%1.1f%%', shadow=True, explode=explode, radius=1
)
axs[1].set_title('Соотношение тренировок', fontsize='small')

# Add text to the figure
fig.text(0.5, 1.02, 'Топ-5 видов спорта', ha='center', va='top', transform=fig.transFigure)

# Show the plot
plt.show()
```

Результат:

<img width="1727" height="670" alt="image" src="https://github.com/user-attachments/assets/75540204-2a15-489b-a9e3-66af3c03fd44" />

<img width="1726" height="736" alt="image" src="https://github.com/user-attachments/assets/35fdcbc8-9a85-4840-9e61-29fc690f446b" />

Узнаем, сколько людей занималось более чем одним видом спорта:

<img width="1245" height="420" alt="image" src="https://github.com/user-attachments/assets/bd3c6d24-bab0-41ad-a8e6-702c2026f5e8" />

Построим диаграмму:

<img width="533" height="654" alt="image" src="https://github.com/user-attachments/assets/5af45d21-fe87-4ebe-8514-b88c9b6c403c" />

Проанализируем количество рекордов за тренировку по всем видам спорта:

```
print('\nГрафик распределения тренировок по видам деятельности:')
plot_size_x, plot_size_y = 5, 5
figsize_x, figsize_y = plot_size_x * 4 + 3, plot_size_y * 13 + 1
figsize=(figsize_x, figsize_y)
fig = plt.figure(figsize=figsize) #
grid_size = (13,4)
ax = plt.subplot2grid(grid_size, (0,0), colspan=1, rowspan=1)
#fig, ax = plt.subplots()
PerWorkoutRecordCount_dist = df.select('PerWorkoutRecordCount', 'sport').toPandas().hist(
    column='PerWorkoutRecordCount', bins=10, sharex = False, grid=True
    , xlabelsize='small', ylabelsize='small', by='sport', ax = ax
    , layout = grid_size, figsize=figsize
    )
a = fig.tight_layout()
title = fig.text(0.5, 1, 'Распределение количества рекордов за тренировку по видам спорта', ha='center'
         , fontsize='small', transform=fig.transFigure);
xlabel = fig.text(
    0.5, 0.01, '# рекорды/тренировки', va='bottom', ha='center', transform=fig.transFigure
)
ylabel = fig.text(0.01, 0.5, 'Частота (количество)', va='center', rotation='vertical');
```

Результат:

<img width="1673" height="720" alt="image" src="https://github.com/user-attachments/assets/7722bd31-734c-4eb1-81c1-9e8efa3f4bc6" />

<img width="1684" height="707" alt="image" src="https://github.com/user-attachments/assets/d7070768-6b4b-47c7-8c7c-095b7f432c46" />

Посмотрим, сколько пользователей выполнило более 10 тренировок:

<img width="1758" height="553" alt="image" src="https://github.com/user-attachments/assets/c56d3df3-acbe-4eb3-bafa-ee3a1b00138b" />

### 7. Pyspark UDF

Создадим 4 вспомогательные функции для столбца 'timestamp', как описано выше, а затем преобразуем их в UDF:

```
# Преобразование столбца метки времени в Datetime.Datetime, чтобы позже использовать его для функции .withColumn.
def to_time(timestamp_list):
    # преобразовать в дату и время и минус 7 часов из-за разницы во временном окне Endomondo с временем utc в качестве описания набора данных
    return [datetime.fromtimestamp(t) - timedelta(hours=7) for t in timestamp_list]

# Регистрация вспомогательной функции to_time в структуре UDF pyspark
udf_to_time = udf(to_time, ArrayType(elementType=TimestampType()))

# Вспомогательная функция для получения продолжительности (в минутах) списка значений даты и времени, которая будет позже использована для функции withColumn.
def get_duration(datetime_list):
    time_dif = max(datetime_list) - min(datetime_list)
    return time_dif.seconds/60

# Регистрация вспомогательной функции get_duration как пользовательской функции в pyspark.
udf_get_duration = udf(get_duration, FloatType())

# Вспомогательная функция для получения времени начала тренировки из списка даты и времени, которая будет позже использована для функции withColumn.
def get_start_time(datetime_list):
    return min(datetime_list)

# Регистрация вспомогательной функции get_start_time как пользовательской функции в pyspark
udf_get_start_time = udf(get_start_time, TimestampType())

# Вспомогательная функция для получения списка интервалов во время тренировки
def get_interval(datetime_list):
    if len(datetime_list) == 1:
        return [0]
    else:
        interval_list = []
        for i in range(0, len(datetime_list)-1):
            interval = (datetime_list[i+1] - datetime_list[i]).seconds
            interval_list.append(interval)
        return interval_list

# Регистрация вспомогательной функции get_interval как пользовательской функции в pyspark.
udf_get_interval = udf(get_interval, ArrayType(elementType=IntegerType()))

# Создание нового столбца date_time для преобразования метки времени в формат даты и времени Python для последующего использования.
df = df.withColumn('date_time',
    udf_to_time('timestamp'))

# Создание столбца «workout_start_time», чтобы получить время начала каждой тренировки/строки:
df = df.withColumn('workout_start_time', hour(udf_get_start_time('date_time')))

# Создание столбца продолжительности из только что созданного столбца date_time, используя функцию udf udf_get_duration, определенную выше.
df = df.withColumn('duration', udf_get_duration('date_time'))

# Создание столбца интервала из столбца date_time, используя функцию udf udf_get_interval, определенную выше.
df = df.withColumn('interval', udf_get_interval('date_time'))

print('Новые столбцы (''date_time'', ''workout_start_time'' in hour\
, ''duration'' in minutes & ''interval'' in seconds)\n, first 5 rows:')
df.select('timestamp','date_time', 'workout_start_time', 'duration', 'interval').limit(5).toPandas()
```

<img width="1663" height="305" alt="image" src="https://github.com/user-attachments/assets/c2812e82-0d03-46b4-8395-f1c4ba4d0a1b" />

Теперь посмотрим на продолжительность каждой тренировки (в минутах):

<img width="1596" height="227" alt="image" src="https://github.com/user-attachments/assets/dd3ba18f-1d38-4625-b488-bdf8ab2d39bc" />

Построим графики продолжительности тренировок:

```

print('\nПостроение графика распределения продолжительности по видам спорта:')
plot_size_x, plot_size_y = 5, 5
figsize_x, figsize_y = plot_size_x * 4 +3, plot_size_y * 13 + 1
figsize = (figsize_x, figsize_y)
fig = plt.figure(figsize=figsize) #
grid_size = (13,4)
ax = plt.subplot2grid(grid_size, (0,0), colspan=1, rowspan=1)

duration_dist = df.select('duration', 'sport').toPandas().hist(
    column='duration', by='sport', bins=15, sharex = False, grid=True
    , xlabelsize='small', ylabelsize='small' , ax = ax
    , layout = grid_size, figsize=figsize
    )
a = fig.tight_layout()
title = fig.text(0.5, 1, 'Распределение продолжительности тренировок по видам спорта'
             , ha='center', va='center', transform=fig.transFigure
            )
xlabel = fig.text(0.5, 0.01, 'Продолжительность тренировки (минуты)'
             , ha='center', va='center', transform=fig.transFigure)
ylabel = fig.text(0.01, 0.5, 'Частота (количество)', va='center', rotation='vertical');
```

<img width="1676" height="714" alt="image" src="https://github.com/user-attachments/assets/885d620a-ce39-46f8-a8b9-02f99052f341" />

<img width="1682" height="717" alt="image" src="https://github.com/user-attachments/assets/5f9b6930-0eca-43ab-a707-e554f93c300a" />

### 8.Преобразование объектов строк в устойчивый распределенный набор данных Spark (RDD)

Создадим вспомогательные функции для просмотра интервалов между записями:

```
# Вспомогательная функция для расчета статистики имени столбца из кортежа x (спорт, список записей столбца) tuple x of (sport, records list of the column)
#, статистика для расчета также предоставляется в качестве входных данных
def calculate_stats(x,column_name, stat_list):
    sport, records_list = x
    stat_dict = {'sport': sport}
    if 'min' in stat_list:
        min_stat = min(records_list)
        stat_dict.update({'min ' + column_name : min_stat})
    if 'max' in stat_list:
        max_stat = max(records_list)
        stat_dict.update({'max ' + column_name: max_stat})
    if 'mean' in stat_list:
        average_stat = stats.mean(records_list)
        stat_dict.update({'mean ' + column_name: average_stat})
    if 'stdev' in stat_list:
        std_stat = stats.stdev(records_list)
        stat_dict.update({'stdev ' + column_name: std_stat})
    if '50th percentile' in stat_list:
        median_stat = stats.median(records_list)
        stat_dict.update({'50th percentile ' + column_name: median_stat})
    if '25th percentile' in stat_list:
        percentile_25th_stat = np.percentile(records_list, 25)
        stat_dict.update({'25th percentile ' + column_name: percentile_25th_stat})
    if '75th percentile' in stat_list:
        percentile_75th_stat = np.percentile(records_list, 75)
        stat_dict.update({'75th percentile ' + column_name: percentile_75th_stat})
    if '95th percentile' in stat_list:
        percentile_95th_stat = np.percentile(records_list, 95)
        stat_dict.update({'95th percentile ' + column_name: percentile_95th_stat})
    return stat_dict

def to_list(a):
    return a

def extend(a, b):
    a.extend(b)
    return a

def retrieve_array_column_stat_df(df, column_name, stat_list):
    # Преобразование спорт и «column_name» в RDD, чтобы легко рассчитать статистику интервалов по видам спорта.
    sport_record_rdd = df.select('sport', column_name).rdd \
    .map(tuple).combineByKey(to_list, extend, extend).persist()

    # Вычислить статистику входного столбца, вызвав функцию Calcul_stats, определенную выше.
    record_statistic_df = pd.DataFrame(sport_record_rdd.map(
        lambda x: calculate_stats(x, column_name,stat_list)).collect()
                                      )
    # Установка правильного порядка столбцов данных.
    columns_order = ['sport'] + [stat + ' ' + column_name for stat in stat_list]
    # Изменение порядка столбцов
    return record_statistic_df[columns_order]

stat_list = ['min', '25th percentile', 'mean', '50th percentile',
                     '75th percentile', '95th percentile', 'max', 'stdev']
interval_statistic_df = retrieve_array_column_stat_df(df, column_name='interval', stat_list=stat_list)
print('\nПросмотр статистики интервалов в секундах (по виду спорта)' )
interval_statistic_df
```

<img width="1741" height="706" alt="image" src="https://github.com/user-attachments/assets/07b3a592-9ae3-457e-84f1-0a4ec05dd212" />

Теперь отобразим эти числа в виде столбцов и линейных диаграмм:

```
import numpy as np
import matplotlib.pyplot as plt

print('\nОбобщенная статистика интервальных видов спорта:')

bar_columns = ['25th percentile interval', '50th percentile interval', '75th percentile interval', '95th percentile interval']
line_columns1 = ['min interval', 'mean interval']
line_columns2 = ['max interval', 'stdev interval']

interval_statistic_df = interval_statistic_df.sort_values(by='95th percentile interval', ascending=False)

figsize = (13, 59)
fig, axs = plt.subplots(nrows=7, figsize=figsize)

d = axs[0].set_title('Интервальная статистика по видам спорта', fontsize=18)

for i in range(7):
    interval_statistic_sub_df = interval_statistic_df.iloc[i*7:i*7+7]

    # Plot the bar chart for the quantiles
    plot1 = interval_statistic_sub_df[['sport'] + bar_columns].groupby(['sport']).agg(np.mean).plot(
        kind='bar', stacked=True, grid=False, alpha=0.5, edgecolor='black', ax=axs[i]
    )

    # Plot the line chart for min and mean intervals
    plot2 = interval_statistic_sub_df[['sport'] + line_columns1].plot(x='sport', ax=axs[i], marker='o')

    # Create a secondary y-axis for max and stdev intervals
    ax2 = axs[i].twinx()
    plot3 = interval_statistic_sub_df[['sport'] + line_columns2].plot(x='sport', ax=ax2, marker='o', color=['m', 'g'])

    # Legends for the plots
    a = axs[i].legend(loc='center left', fontsize=16, bbox_to_anchor=(1.2, 0.5), frameon=False)
    a = ax2.legend(labels=['max interval (right)', 'stdev interval (right)'], loc="center left", fontsize=16, bbox_to_anchor=(1.2, 0.11), frameon=False)

    # Formatting for x-ticks and labels
    b = axs[i].set_xticklabels(interval_statistic_sub_df['sport'], rotation='horizontal', fontsize='small')
    c = axs[i].set_xlabel('Спорт (Активность)', fontsize='small')
    d = axs[i].set_ylabel('Статистика квантилей + мин/среднее\n(секунд)', fontsize=16)
    e = ax2.set_ylabel('Max/stdev Statistics\n(second)', fontsize=16)

    # Set font size for y-axis ticks
    for tick in axs[i].yaxis.get_major_ticks():
        tick.label1.set_fontsize(16)  # Use label1 for primary y-axis labels
    ax2.tick_params(axis='y', labelsize=16)

    # Make sure x-tick labels are visible for all subplots
    b = plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=True)

# Adjust the layout for better visualization
plt.subplots_adjust(hspace=0.2)

# Show the plot
plt.show()
```

<img width="1715" height="355" alt="image" src="https://github.com/user-attachments/assets/9fb2ec1c-9442-422e-a050-e075913be23b" />

<img width="1685" height="676" alt="image" src="https://github.com/user-attachments/assets/50f61ed0-129b-4fe6-ab9b-58bf73dc047d" />

<img width="1715" height="680" alt="image" src="https://github.com/user-attachments/assets/5be6bb31-18be-4964-a7f5-642b8c892fc6" />

Теперь получим графики распределения времени начала тренировки по видам спорта с разбивкой по полу:

```
activities = start_time_df['sport'].unique()
plot_size_x, plot_size_y = 5, 5
figsize_x, figsize_y = (plot_size_x + 0.5) * 4 +3, (plot_size_y + 1) * 13 + 1


nrows, ncols = 13, 4
a = fig.subplots_adjust(hspace = 1, wspace = 1)
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize_x, figsize_y))
print('\nГрафик распределения времени начала тренировки по видам спорта с разбивкой по полу:')
a = plt.setp(axs, xticks=[0, 4, 8, 12, 16, 20])
for index, sport in enumerate(activities):
    row_index, col_index = divmod(index, ncols)
    male_start_time_list = start_time_df[(start_time_df.sport == sport) &
                                            (start_time_df.gender == 'male')]['workout_start_time']
    female_start_time_list = start_time_df[(start_time_df.sport == sport) &
                                            (start_time_df.gender == 'female')]['workout_start_time']
    unknown_start_time_list = start_time_df[(start_time_df.sport == sport) &
                                            (start_time_df.gender == 'unknown')]['workout_start_time']
    if len(male_start_time_list) > 0:
        male_dist = axs[row_index, col_index].hist(male_start_time_list,
                                      bins = 12, alpha=0.5, label='male', range=(0, 23))
    if len(female_start_time_list) > 0:
        female_dist = axs[row_index, col_index].hist(female_start_time_list,
                                      bins = 12, alpha=0.5, label='female', range=(0, 23))
    if len(unknown_start_time_list) > 0:
        unknown_dist = axs[row_index, col_index].hist(unknown_start_time_list,
                                      bins = 12, alpha=0.5, label = 'unknown', range=(0, 23))
    b= axs[row_index, col_index].set_title('Activitiy: ' + sport, fontsize='small')
    a = axs[row_index, col_index].legend(loc="upper left", fontsize='small')
    a = plt.setp(axs[row_index, col_index].get_xticklabels(), fontsize='small')

for i in range(1,4):
    x = axs[12, i].set_visible(False)
a = fig.tight_layout()
z = fig.text(0.5, 1, 'Распределение времени начала тренировки (часы) по видам спорта'
             , ha='center', va='top', transform=fig.transFigure)
y = fig.text(0.5, 0.01, 'Начало тренировки час в день (час)'
             , ha='center', va='bottom', transform=fig.transFigure)
z = fig.text(0.02, 0.5, 'Частота (количество)', va='center', rotation='vertical');
```

Результат:

<img width="1696" height="733" alt="image" src="https://github.com/user-attachments/assets/a32796e7-35c9-4b19-b0d4-be447c0a27ed" />

Нормализуем время для всех тренировок, рассчитав продолжительность (в секундах) каждой записи временной метки из первой записи тренировки. Затем отображаем частоту сердечных сокращений в зависимости от этого нормализованного времени, группируя по видам спорта.

```
# Лямбда-функция для объединения списка списков в один большой список
flattern = lambda l: set([item for sublist in l for item in sublist])

normalized_datetime_list = []
for index,data_row in pd_df.iterrows():
    min_date_time = min(data_row['date_time'])
    normalized_datetime_list.append(
        [(date_time - min_date_time).seconds for date_time in data_row['date_time']]
    )

pd_df['normalized_date_time'] = normalized_datetime_list

print('New normalized datetime (first 7 rows):')
pd_df.head(7)[['userId', 'sport', 'date_time','normalized_date_time']]

print('\nПостроение необработанного пульс (выборка) по нормированному времени:')

sport_list = pd_df['sport'].unique()
# Динамическое определение длины фигуры зависит от длины спортивного списка.
fig, axs = plt.subplots(len(sport_list), figsize=(15, 6*len(sport_list)))
subplot_adj = fig.subplots_adjust(hspace = 0.6)
plot_setp = plt.setp(axs, yticks=range(0,250,20))

for sport_index, sport in enumerate(sport_list):
    workout = pd_df[pd_df.sport == sport]
    max_time = max(flattern(workout.normalized_date_time))
    for workout_index, data_row in workout.iterrows():
        label = 'user: ' + str(data_row['userId']) + ' - gender: ' + data_row['gender']
        plot_i = axs[sport_index].plot(
            data_row['normalized_date_time'], data_row['heart_rate'], label=label
        )
    title_i = axs[sport_index].set_title('Activitiy: ' + sport, fontsize='small')
    xlabel_i = axs[sport_index].set_xlabel('Time (sec)', fontsize='small')
    xsticklabels_i = axs[sport_index].set_xticklabels(
        range(0, max_time, 500),rotation = 'vertical', fontsize=9
    )
    ysticklabels_i = axs[sport_index].set_yticklabels(range(0,250,20),fontsize='small')
    legend_i = axs[sport_index].legend(
        loc='center left', bbox_to_anchor=(1.0, 0.5), prop={'size': 9}
    )

x_label = fig.text(0.04, 0.5, 'Частота сердечных сокращений (уд/мин)', va='center', rotation='vertical')
chart_title = fig.text(0.5, 1.3, 'Необработанная частота пульса (выборка) по нормализованному времени',
            ha='center', va='center', fontsize='small', transform=axs[0].transAxes)
```

Результат:

<img width="1687" height="718" alt="image" src="https://github.com/user-attachments/assets/c9df56b6-2e32-4bba-a17a-a0da05c057f7" />

Теперь проанализируем перемещения во время тренировок:

<img width="1429" height="542" alt="image" src="https://github.com/user-attachments/assets/5da2103a-7c71-4631-a42c-d818de4d7f52" />

```
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.ticker as mtick
import numpy as np  # To use numpy functions for standard deviation

def get_fixed_mins_maxs(mins, maxs):
    deltas = (maxs - mins) / 12.
    mins = mins + deltas / 4.
    maxs = maxs - deltas / 4.
    return [mins, maxs]

workout_count = pd_df_small.shape[0]
ncols = 3
nrows = math.ceil(workout_count / ncols)

fig = plt.figure(figsize=(8 * (ncols + 0.5), 8 * nrows))
fig.subplots_adjust(hspace=0.2, wspace=0.5)

print('Построение траекторий тренировки в виде 3D для каждой тренировки:')
for row_index, row in pd_df_small.iterrows():
    min_long = min(row['longitude']) - np.std(row['longitude'])  # Use numpy.std()
    max_long = max(row['longitude']) + np.std(row['longitude'])  # Use numpy.std()
    minmax_long = get_fixed_mins_maxs(min_long, max_long)

    min_lat = min(row['latitude']) - np.std(row['latitude'])  # Use numpy.std()
    max_lat = max(row['latitude']) + np.std(row['latitude'])  # Use numpy.std()
    minmax_lat = get_fixed_mins_maxs(min_lat, max_lat)

    min_alt = min(row['altitude']) - np.std(row['altitude'])  # Use numpy.std()
    max_alt = max(row['altitude']) + np.std(row['altitude'])  # Use numpy.std()
    minmax_alt = get_fixed_mins_maxs(min_alt, max_alt)

    ax = fig.add_subplot(nrows, ncols, row_index + 1, projection='3d')

    title = 'Активность: ' + row['sport'] + ' - Пол: ' + row['gender'] + \
            '\nРекорды: ' + str(int(row['PerWorkoutRecordCount'])) + \
            ' - Длительность: ' + str(int(row['duration'])) + ' минуты'
    ax.set_title(title, fontsize=16)

    # Scatter plot for points
    scatter = ax.scatter(row['longitude'], row['latitude'], row['altitude'], c='r', marker='o')

    # Plot the workout path in 3D
    plot = ax.plot3D(row['longitude'], row['latitude'], row['altitude'], c='gray', label='Workout path')

    # Labels for the axes
    ax.set_xlabel('Долгота (градусы)', fontsize=16)
    ax.set_ylabel('Широта (градусы)', fontsize=16)
    ax.set_zlabel('Высота (м)', fontsize=16, rotation=0)

    # Set font size for ticks (accessing `label1` for the primary ticks)
    for t in ax.xaxis.get_major_ticks():
        t.label1.set_fontsize(16)
    for t in ax.yaxis.get_major_ticks():
        t.label1.set_fontsize(16)
    for t in ax.zaxis.get_major_ticks():
        t.label1.set_fontsize(16)

    # Add legends
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    # Set limits for each axis
    ax.set_xlim(minmax_long)
    ax.set_ylim(minmax_lat)
    if minmax_alt[0] != minmax_alt[1]:
        ax.set_zlim(minmax_alt)

    # Set aspect ratio for the plot (adjust these values as needed)
    ax.set_box_aspect([4, 2, 0.5])  # aspect ratio for x, y, z axes

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

# Add a global title
fig.text(0.5, 1.02, "Маршрут тренировки (долгота/широта/высота)", ha='center', va='top', fontsize=18)

plt.tight_layout()
plt.show()
```

Результат:

<img width="1675" height="708" alt="image" src="https://github.com/user-attachments/assets/1a79a978-a7ba-43d8-a426-19c42c35f482" />

# Индивидуальное задание
## Вариант №12
### 1.Какие виды спорта, согласно анализу в разделах 5 и 6, являются самыми популярными и по числу пользователей, и по числу тренировок? Есть ли виды спорта, популярные в одной метрике, но не в другой?

<img width="1760" height="221" alt="image" src="https://github.com/user-attachments/assets/d6f2cc6a-8a77-4073-b81b-de713bef9e43" />

### 2.Интерпретируйте статистику по duration (раздел 7, df.select('duration').toPandas().describe().T). Какова медианная продолжительность тренировки? Что означает большая разница между 75-м перцентилем и максимумом?

<img width="1393" height="430" alt="image" src="https://github.com/user-attachments/assets/f5fbd7d3-6176-4bfe-a0c3-ef640013932f" />

### 3.Объясните бизнес-ценность анализа интервалов записи данных (interval, раздел 8). Почему компании Endomondo может быть важно знать, как часто их трекеры записывают данные?

<img width="1539" height="371" alt="image" src="https://github.com/user-attachments/assets/3ca314e7-1c88-4ea8-81b1-df7a1a673ed7" />

### 4.Напишите код PySpark, чтобы вычислить среднюю продолжительность (duration) тренировок для пользователей мужского (male) и женского (female) пола отдельно.

```
from pyspark.sql import functions as F

# Вычисляем среднюю продолжительность тренировок по полу с округлением
avg_duration_by_gender = df \
    .filter(df.gender.isin(['male', 'female'])) \
    .groupBy('gender') \
    .agg(F.round(F.avg('duration'), 1).alias('avg_duration_minutes')) \
    .orderBy('gender')

# Показываем результаты
print("Средняя продолжительность тренировок по полу:")
avg_duration_by_gender.show()
```

<img width="1256" height="477" alt="image" src="https://github.com/user-attachments/assets/1f159888-dd0e-46dd-894d-fe26b642c68b" />

Нарисуем график:
```
import seaborn as sns

# Устанавливаем стиль seaborn
sns.set_style("whitegrid")

duration_pandas = avg_duration_by_gender.toPandas()

# Создание графика
ax = sns.barplot(x='gender', y='avg_duration_minutes', data=duration_pandas,
                 palette=['lightpink', 'lightblue'],
                 edgecolor='black',
                 linewidth=1)

# Настройка внешнего вида
min_val = duration_pandas['avg_duration_minutes'].min()
max_val = duration_pandas['avg_duration_minutes'].max()
ax.set_ylim(min_val - 5, max_val + 2)

ax.set_title('Средняя продолжительность тренировок по полу', fontsize=14, pad=20)
ax.set_ylabel('Продолжительность (минуты)', fontsize=12)
ax.set_xlabel('Пол', fontsize=12)

# Добавление значений на столбцы
for i, v in enumerate(duration_pandas['avg_duration_minutes']):
    ax.text(i, v + 0.3, f'{v} мин', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Улучшение сетки
ax.grid(axis='y', alpha=0.3)

# Убираем рамки
sns.despine(left=True, bottom=True)

# Переименовываем подписи на оси X
gender_labels = {'female': 'Женщины', 'male': 'Мужчины'}
ax.set_xticklabels([gender_labels.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()])

# Показываем график
plt.show()
```

<img width="670" height="501" alt="image" src="https://github.com/user-attachments/assets/9ec5c717-f5b3-43f0-9210-10a568926079" />

### 5.Предложите задачу бинарной классификации в Spark MLlib: предсказать, будет ли тренировка пользователя дольше медианной продолжительности для данного вида спорта. Какие признаки (пол пользователя, время старта, день недели - если бы был) могут быть использованы?

<img width="1473" height="705" alt="image" src="https://github.com/user-attachments/assets/466f4794-f88e-4f41-9b1c-d12c047e8c50" />
