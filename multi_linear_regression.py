import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

rainfall = pd.read_csv('annual_mean_daily_rainfall.csv')
sea_temp = pd.read_csv('annual_mean_sea_temperature.csv')
carbon = pd.read_csv('annual_carbon_emissions.csv')
y = rainfall.iloc[:25, 1].values
x1 = sea_temp.iloc[:25, 1].values
x2 = carbon.iloc[:25, 1].values
x = np.vstack([x1, x2]).T

# KFold
LR = LinearRegression(fit_intercept=False)
kcv = KFold(n_splits=5, random_state=60, shuffle=True)
MSE = []
for train_index, test_index in kcv.split(x):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    LR.fit(X_train, y_train)
    MSE.append(mean_squared_error(y_test, LR.predict(X_test)))
print(f'Multi-Linear Regression MSE: {np.mean(MSE):.5f}' + \
      f', STD: {np.std(MSE):.5f}')

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
