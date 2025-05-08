import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import optuna
data = pd.read_csv('public_data.csv')
data['date'] = pd.to_datetime(data['date'])

# ставим начальный индекс на 1
data.index = range(1,len(data)+1)
final_dfs = []

#функция вывода графика
def plot_country_volumes(df: pd.DataFrame, y: str):
    fig = px.line(df, x='date', y=y, labels={'date': 'Date'})
    fig.update_layout(template="simple_white", font=dict(size=18), title_text='CO2',
                      width=1750, title_x=0.5, height=400)

    return fig.show()

def objective(trial, series):
    # Определяем параметры для оптимизации (пример)
    p = trial.suggest_int('p', 0, 3)
    d = trial.suggest_int('d', 0, 1)
    q = trial.suggest_int('q', 0, 3)
    P = trial.suggest_int('P', 0, 2)
    D = trial.suggest_int('D', 0, 1)
    Q = trial.suggest_int('Q', 0, 2)

    model = SARIMAX(series[:-30], order=(p, d, q), seasonal_order=(P, D, Q, 12), enforce_stationarity=False, enforce_invertibility=False) #Важно!
    model_fit = model.fit(disp=False) # disp=False для подавления вывода

    forecast = model_fit.forecast(steps=30)
    mse = mean_squared_error(series[-30:], forecast)  # Оцениваем на отложенной выборке

    return mse

# Перебираем каждую колонку, кроме даты
result = 0
for i in data.columns[1:]:
    study = optuna.create_study(direction='minimize')

    # Запускаем оптимизацию.  Укажите n_trials (количество попыток)
    study.optimize(lambda trial: objective(trial, data[i]), n_trials=10)  # Изменил lambda

    # Получаем лучшие гиперпараметры
    best_params = study.best_params
    print(f"Лучшие параметры для {i}: {best_params}")
    best_model = SARIMAX(data[i][:-30],
                         order=(best_params['p'], best_params['d'], best_params['q']),
                         seasonal_order=(best_params['P'], best_params['D'], best_params['Q'], 12),
                         enforce_stationarity=False, enforce_invertibility=False)  # Важно!
    best_model_fit = best_model.fit(disp=False)
    # Прогноз на основе обученной модели
    forecast = best_model_fit.forecast(steps=30)

    # Рассчитываем MSE и MAE
    mse = mean_squared_error(data[i][-30:], forecast)
    mae = mean_absolute_error(data[i][-30:], forecast)

    print(f'MSE: {mse}')
    print(f'MAE: {mae}')

    # Делаем прогноз на 31 день
    forecast_future = best_model_fit.forecast(steps=31)


    # Создаем новый DataFrame для будущих значений
    future_dates = pd.date_range(start='2023-05-01', periods=31, freq='D')
    forecast_df = pd.DataFrame({'date': future_dates, i: forecast_future})

    # Присоединяем прогноз к списку с прогнозами
    final_dfs.append(forecast_df)
    res = final_dfs[0]
    print(final_dfs)
    # Присоединяем датафреймы с прогнозами к исходеым данным
    for i in range(len(final_dfs)-1):
        combined_df = pd.merge(res, final_dfs[i+1], on = 'date')
        res = combined_df
    # Выставляем начальный индекс прогнозов на 1583
    res = res.reset_index(drop=True)
    res.index += 1583
    result = pd.concat([data, res])

    #вывод графиков с прогнозами выброса CO2 каждой страны
for i in data.columns[1:]:
    plot_country_volumes(result, i)

    # Создание итогового csv файла
    # result.to_csv('output.csv', index=False)




        # Вывод датафрейма с прогнозированными значениями
        # print(forecast_df)
#     model = SARIMAX(data[i][:-30], order=(1, 0, 0), seasonal_order=(2, 0, 0, 12))
#     model_fit = model.fit()
#
#     # Вывод подробной информации
#     # print(model_fit.summary())
#
#     # Вывод гарфиков стран без прогноза
#     # plot_country_volumes(data, i)
#
#     # Прогноз на основе обученной модели
#     forecast = model_fit.forecast(steps=30)
#
#     # Рассчитываем MSE и MAE
#     mse = mean_squared_error(data[i][-30:], forecast)
#     mae = mean_absolute_error(data[i][-30:], forecast)
#
#     print(f'MSE: {mse}')
#     print(f'MAE: {mae}')
#
#     # Делаем прогноз на 31 день
#     forecast_future = model_fit.forecast(steps=31)
#
#
#     # Создаем новый DataFrame для будущих значений
#     future_dates = pd.date_range(start='2023-05-01', periods=31, freq='D')
#     forecast_df = pd.DataFrame({'date': future_dates, i: forecast_future})
#
#     # Присоединяем прогноз к списку с прогнозами
#     final_dfs.append(forecast_df)
#
#     # Вывод датафрейма с прогнозированными значениями
#     # print(forecast_df)
#
# # print(final_dfs)
# res = final_dfs[0]
#
# # Присоединяем датафреймы с прогнозами к исходеым данным
# for i in range(len(data.columns)-2):
#     combined_df = pd.merge(res, final_dfs[i+1], on = 'date')
#     res = combined_df
# # Выставляем начальный индекс прогнозов на 1583
# res = res.reset_index(drop=True)
# res.index += 1583
# result = pd.concat([data, res])
#
# #вывод графиков с прогнозами выброса CO2 каждой страны
# for i in data.columns[1:]:
#     plot_country_volumes(result, i)

#Создание итогового csv файла
# result.to_csv('output.csv', index=False)




# data = pd.read_csv('public_data.csv')
# data['date'] = pd.to_datetime(data['date'])
#
# # ставим начальный индекс на 1
# data.index = range(1,len(data)+1)
# final_dfs = []
#
# #функция вывода графика
# def plot_country_volumes(df: pd.DataFrame, y: str):
#     fig = px.line(df, x='date', y=y, labels={'date': 'Date'})
#     fig.update_layout(template="simple_white", font=dict(size=18), title_text='CO2',
#                       width=1750, title_x=0.5, height=400)
#
#     return fig.show()
#
#
# # Перебираем каждую колонку, кроме даты
# for i in data.columns[1:]:
#     model = SARIMAX(data[i][:-30], order=(1, 1, 1), seasonal_order=(1, 0, 0, 365))
#     model_fit = model.fit()
#
#     # Вывод подробной информации
#     # print(model_fit.summary())
#
#     # Вывод гарфиков стран без прогноза
#     # plot_country_volumes(data, i)
#
#     # Прогноз на основе обученной модели
#     forecast = model_fit.forecast(steps=30)
#
#     # Рассчитываем MSE и MAE
#     mse = mean_squared_error(data[i][-30:], forecast)
#     mae = mean_absolute_error(data[i][-30:], forecast)
#
#     print(f'MSE: {mse}')
#     print(f'MAE: {mae}')
#
#     # Делаем прогноз на 31 день
#     forecast_future = model_fit.forecast(steps=31)
#
#
#     # Создаем новый DataFrame для будущих значений
#     future_dates = pd.date_range(start='2023-05-01', periods=31, freq='D')
#     forecast_df = pd.DataFrame({'date': future_dates, i: forecast_future})
#
#     # Присоединяем прогноз к списку с прогнозами
#     final_dfs.append(forecast_df)
#
#     # Вывод датафрейма с прогнозированными значениями
#     # print(forecast_df)
#
# # print(final_dfs)
# res = final_dfs[0]
#
# # Присоединяем датафреймы с прогнозами к исходеым данным
# for i in range(len(data.columns)-2):
#     combined_df = pd.merge(res, final_dfs[i+1], on = 'date')
#     res = combined_df
# # Выставляем начальный индекс прогнозов на 1583
# res = res.reset_index(drop=True)
# res.index += 1583
# result = pd.concat([data, res])
#
# #вывод графиков с прогнозами выброса CO2 каждой страны
# for i in data.columns[1:]:
#     plot_country_volumes(result, i)
#
# #Создание итогового csv файла
# # result.to_csv('output.csv', index=False)