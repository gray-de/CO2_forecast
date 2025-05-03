import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import grid_search_forecaster
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

data = pd.read_csv('public_data.csv')
data['date'] = pd.to_datetime(data['date']) # Приводим дату в тип pandas

# Задаем обучающие данные
# Обычно предсказание нужно с определенного момента, которое уточняется с заказчиком
# Тут используем обычную hold-out валидацию
train = data.iloc[:-int(len(data) * 0.2)]
test = data.iloc[-int(len(data) * 0.2):]
# Отрисуем данные
def plot_country_volumes(df: pd.DataFrame, y: str):
    fig = px.line(df, x='date', y=y, labels={'date': 'Date'})
    fig.update_layout(template="simple_white", font=dict(size=18), title_text='CO2',
                      width=1650, title_x=0.5, height=400)

    return fig.show()
linear_forecaster = ForecasterAutoreg(
    regressor=LinearRegression(),
    lags=12
)

# Обучаем модель
linear_forecaster.fit(train)

# Строим прогноз
predictions = linear_forecaster.predict(len(train))

# Печатаем метрики
print(f"MAPE = {mean_absolute_percentage_error(train, predictions)}")
print(f"MAE = {mean_absolute_error(train, predictions)}")
print(f"MSE = {mean_squared_error(train, predictions)}")