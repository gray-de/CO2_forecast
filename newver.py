import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go

data = pd.read_csv('public_data.csv')
data['date'] = pd.to_datetime(data['date'])
data.index = range(1,len(data)+1)
final_dfs = []
# print(data)
def plot_country_volumes(df: pd.DataFrame, y: str):
    fig = px.line(df, x='date', y=y, labels={'date': 'Date'})
    fig.update_layout(template="simple_white", font=dict(size=18), title_text='CO2',
                      width=1750, title_x=0.5, height=400)

    return fig.show()

# plot_country_volumes(data, "Brazil")

for i in data.columns[1:]:
    model = SARIMAX(data[i][:-30], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()
    # print(model_fit.summary())

    # Прогноз на основе обученной модели
    forecast = model_fit.forecast(steps=30)

    # Рассчитываем MSE и MAE
    mse = mean_squared_error(data[i][-30:], forecast)
    mae = mean_absolute_error(data[i][-30:], forecast)

    print(f'MSE: {mse}')
    print(f'MAE: {mae}')

    forecast_future = model_fit.get_forecast(steps=31)


    # Создаем новый DataFrame для будущих значений
    future_dates = pd.date_range(start='2023-04-30', periods=31, freq='D') + pd.DateOffset(days=1)
    forecast_df = pd.DataFrame({'date': future_dates, i: forecast_future.predicted_mean})
    # print(forecast_df)


    # Присоединяем прогноз к исходному DataFrame
    # new_df = pd.concat([data, forecast_df], join = "inner")
    final_dfs.append(forecast_df)
    print(forecast_df)
    # plot_country_volumes(new_df, i)
# print(final_dfs)
res = final_dfs[0]
for i in range(len(data.columns)-2):
    combined_df = pd.merge(res, final_dfs[i+1], on = 'date')
    res = combined_df
print(res)