import pandas as pd
import plotly.express as px
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
warnings.filterwarnings("ignore", category=ConvergenceWarning)
data = pd.read_csv('public_data.csv')
data['date'] = pd.to_datetime(data['date'])

# Ставим начальный индекс на 1
data.index = range(1,len(data)+1)
final_dfs = []
maes = []
mses = []
# Функция вывода графика
def plot_country_volumes(df: pd.DataFrame, y: str):
    fig = px.line(df, x='date', y=y, labels={'date': 'Date'})
    fig.update_layout(template="simple_white", font=dict(size=18), title_text='CO2',
                      width=1580, title_x=0.5, height=400)

    return fig.show()
# Перебираем каждую колонку, кроме date
for i in data.columns[1:]:
    # Создаём и обучаем модель для проверки качества прогноза
    model = ExponentialSmoothing(data[i][:-30], seasonal_periods=365, seasonal='add')
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)

    # метрики
    mse = mean_squared_error(data[i][-30:], forecast)
    mae = mean_absolute_error(data[i][-30:], forecast)

    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    maes.append(mae)
    mses.append(mse)

    # Создаём и обучаем ещё одну модель для прогнозирования на 31 день вперёд
    future_model = ExponentialSmoothing(data[i], seasonal_periods=365, seasonal='add')
    future_model_fit = future_model.fit()
    forecast_future = future_model_fit.forecast(steps=31)

    # Создаем новый DataFrame для будущих значений
    future_dates = pd.date_range(start='2023-05-01', periods=31, freq='D')
    forecast_df = pd.DataFrame({'date': future_dates, i: forecast_future})

    # Присоединяем прогноз к списку с прогнозами
    final_dfs.append(forecast_df)
    res = final_dfs[0]
    # print(final_dfs)
    # Присоединяем датафреймы с прогнозами к исходным данным
    for i in range(len(final_dfs)-1):
        combined_df = pd.merge(res, final_dfs[i+1], on = 'date')
        res = combined_df
    # Выставляем начальный индекс прогнозов на 1583
    res = res.reset_index(drop=True)
    res.index += 1583
    result = pd.concat([data, res])

    #вывод графиков с прогнозами выброса CO2 каждой страны
# for i in data.columns[1:]:
#     plot_country_volumes(result, i)
# Создание итогового csv файла
# result.to_csv('output.csv', index=False)

print("--------------------------------------------------------")
combined_mae = np.mean(maes)
print(f"Combined MAE (Simple Average): {combined_mae}")
combined_mse = np.mean(mses)
print(f"Combined MSE (Simple Average): {combined_mse}")
