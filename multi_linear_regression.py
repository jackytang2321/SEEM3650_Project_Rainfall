import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

rainfall = pd.read_csv('annual_mean_daily_rainfall.csv')
sea_temp = pd.read_csv('annual_mean_sea_temperature.csv')
carbon = pd.read_csv('annual_carbon_emissions.csv')

data = {'x1': sea_temp.iloc[:25, 1].values,
        'x2': carbon.iloc[:25, 1].values,
        'y': rainfall.iloc[:25, 1].values}
df = pd.DataFrame(data)
x = np.array(df[['x1', 'x2']])
y = np.array(df['y'])

# KFold
LR = LinearRegression(fit_intercept=False, )
kcv = KFold(random_state=60, shuffle=True)
MSE = []
for train_index, test_index in kcv.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    LR.fit(x_train, y_train)
    print(f'{y_test} and {np.round(LR.predict(x_test), 2)}')
    MSE.append(mean_squared_error(y_test, np.round(LR.predict(x_test), 2)))
print(f'Multi-Linear Regression MSE: {np.mean(MSE):.5f}')

# Prediction
hist_y = rainfall.iloc[25:, 1].values
new_x1 = sea_temp.iloc[25:, 1].values
new_x2 = carbon.iloc[25:, 1].values
new_x = np.vstack([new_x1, new_x2]).T
pred_y = []
for i in range(5):
    pred = LR.predict(new_x[i, :].reshape(1, -1))
    pred_y.append(round(pred[0], 2))
pred_MSE = mean_squared_error(hist_y, pred_y)
print(f'Predicted Y (2018-2022) = {pred_y}')
print(f'History Data of Y (2018-2022) = {hist_y}')
print(f'Predicted Y (From LR) MSE: {pred_MSE:.5f}')
