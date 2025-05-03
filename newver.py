import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go

data = pd.read_csv('public_data.csv')
data['date'] = pd.to_datetime(data['date'])
data.index = range(1,len(data)+1)
# print(data)
def plot_country_volumes(df: pd.DataFrame, y: str):
    fig = px.line(df, x='date', y=y, labels={'date': 'Date'})
    fig.update_layout(template="simple_white", font=dict(size=18), title_text='CO2',
                      width=1750, title_x=0.5, height=400)

    return fig.show()

# plot_country_volumes(data, "Brazil")
model = SARIMAX(data['Brazil'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()
# print(model_fit.summary())

# Прогноз на основе обученной модели
forecast = model_fit.forecast(steps=30)

# Рассчитываем MSE и MAE
mse = mean_squared_error(data['Brazil'][-30:], forecast)
mae = mean_absolute_error(data['Brazil'][-30:], forecast)

print(f'MSE: {mse}')
print(f'MAE: {mae}')

forecast_future = model_fit.get_forecast(steps=31)


# Создаем новый DataFrame для будущих значений
future_dates = pd.date_range(start='2023-04-30', periods=31, freq='D') + pd.DateOffset(days=1)
forecast_df = pd.DataFrame({'date': future_dates, 'Brazil': forecast_future.predicted_mean})
# print(forecast_df)


# Присоединяем прогноз к исходному DataFrame
new_df = pd.concat([data, forecast_df])
print(new_df)
plot_country_volumes(new_df, 'Brazil')