import pandas as pd
import plotly.express as px
from scipy.stats import boxcox
import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima.model import ARIMA
from scipy.special import inv_boxcox
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def plot_country_volumes(df: pd.DataFrame, y: str):
    fig = px.line(df, x='date', y=y, labels={'date': 'Date'})
    fig.update_layout(template="simple_white", font=dict(size=18), title_text='CO2',
                      width=1650, title_x=0.5, height=400)

    return fig.show()

data = pd.read_csv('public_data.csv')
data['date'] = pd.to_datetime(data['date'])
for i in data.columns[1:]:


    # plot_country_volumes(df=data, y=i)


    data[i+'_Boxcox'], lam = boxcox(data[i])

    # plot_Italy_volumes(df=data, y='Italy_Boxcox')



    # Difference the data
    data[i+'_diff'] = data[i+'_Boxcox'].diff()
    data.dropna(inplace=True)

    # Plot acf and pacf
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5), dpi=80)
    plot_acf(data[i+'_diff'])
    plot_pacf(data[i+'_diff'], method='ywm')
    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    # plt.show()

    # Split train and test
    train1 = data.iloc[:-int(len(data) * 0.2)]
    test1 = data.iloc[-int(len(data) * 0.2):]
    test = data[-31:]
    train = data.iloc[:]
    # test = data.iloc[:int(len(data))]
    # train = data.copy()
    # train = train.sort_values(by='date')


    last_date = train['date'].iloc[-1]
    forecast_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 32)]

    # Build ARIMA model and inverse the boxcox
    model = ARIMA(train[i+'_Boxcox'], order=(28, 1, 28)).fit()
    # boxcox_forecasts = model.forecast(steps=31)
    boxcox_forecasts = model.forecast(len(test))
    forecasts = inv_boxcox(boxcox_forecasts, lam)

    forecast_df = pd.DataFrame({'date': forecast_dates, 'forecasts': forecasts})

    def plot_forecasts1(forecasts: list[float], title: str) -> None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train['date'], y=train[i], name='Train'))
        fig.add_trace(go.Scatter(x=test['date'], y=test[i], name='Test'))
        fig.add_trace(go.Scatter(x=test['date'], y=forecasts, name='Forecast'))
        fig.update_layout(template="simple_white", font=dict(size=18), title_text=title,
                          width=1750, title_x=0.5, height=400, xaxis_title='Date',
                          yaxis_title=i+'Volume')

        return fig.show()

    def plot_forecasts(train: pd.DataFrame, forecast_df: pd.DataFrame, title: str) -> None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train['date'], y=train[i], name='Train'))
        combined_dates = [train['date'].iloc[-1]] + forecast_df['date'].tolist()
        combined_forecasts = [train[i].iloc[-1]] + forecast_df['forecasts'].tolist()
        fig.add_trace(go.Scatter(x=combined_dates, y=combined_forecasts, name='Forecast'))  # Отображаем даты прогноза
        fig.update_layout(template="simple_white", font=dict(size=18), title_text=title,
                          width=1750, title_x=0.5, height=400, xaxis_title='Date',
                          yaxis_title=i+'Volume')


        return fig.show()


    # Plot the forecasts
    plot_forecasts1(forecasts, 'ARIMA')
    # plot_forecasts(train, forecast_df, 'ARIMA')
    # mape = mean_absolute_percentage_error(test1, forecast_df)
    # print(f"MAPE: {mape}")






# def predict_co2_emissions_arima(csv_filepath, output_filepath="predicted_co2_arima.csv", order=(5,1,0), forecast_days=31):
#     """
#     Прогнозирует выбросы CO2 для каждой страны в CSV файле на forecast_days вперед,
#     используя рекурсивное прогнозирование с ARIMA и экспортирует результаты в новый CSV файл.
#
#     Args:
#         csv_filepath (str): Путь к входному CSV файлу.
#         output_filepath (str, optional): Путь к выходному CSV файлу.
#                                           Defaults to "predicted_co2_arima.csv".
#         order (tuple, optional): Порядок (p, d, q) модели ARIMA. Defaults to (5,1,0).
#         forecast_days (int, optional): Количество дней для прогнозирования. Defaults to 31.
#     """
#
#     try:
#         # 1. Чтение данных из CSV
#         df = pd.read_csv("public_data.csv", index_col=0)
#         df.index = pd.to_datetime(df.index)
#
#         # 2. Создание DataFrame для хранения прогнозов
#         future_dates = pd.date_range(start=df.index[-1] + datetime.timedelta(days=1), periods=forecast_days)
#         future_df = pd.DataFrame(index=future_dates)
#
#         # 3. Прогнозирование для каждой страны
#         for country in df.columns:
#             print(f"Прогнозирование для страны: {country}")
#
#             # a. Подготовка данных
#             historical_data = df[country].dropna().values  # Удаляем пропущенные значения
#             predictions = []
#
#             # b. Рекурсивное прогнозирование
#             history = list(historical_data)  # Используем список для простоты добавления новых значений
#             for _ in range(forecast_days):
#                 # Обучение модели ARIMA
#                 model = ARIMA(history, order=order) #Обучаем модель на имеющихся данных
#                 model_fit = model.fit()
#
#                 # Прогнозирование следующего значения
#                 output = model_fit.forecast() #прогнозируем одно значение
#                 yhat = output[0] #берем прогноз
#
#                 # Добавление прогноза в список прогнозов и "историю"
#                 predictions.append(yhat)
#                 history.append(yhat)  # Добавляем прогноз к "истории" для следующей итерации
#
#             # c. Запись прогнозов в future_df
#             future_df[country] = predictions
#
#         # 4. Объединение исторических данных и прогнозов
#         combined_df = pd.concat([df, future_df])
#
#         # 5. Экспорт в CSV
#         combined_df.to_csv(output_filepath)
#         print(f"Прогнозы сохранены в файл: {output_filepath}")
#
#     except FileNotFoundError:
#         print(f"Ошибка: Файл не найден: {csv_filepath}")
#     except Exception as e:
#         print(f"Произошла ошибка: {e}")
#
#
# # Пример использования:
# csv_file = "co2_emissions.csv"  # Замените на имя вашего файла
# output_file = "co2_emissions_predicted_arima.csv"
# predict_co2_emissions_arima(csv_file, output_file, order=(5,1,0), forecast_days=31)



