from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

rainfall = pd.read_csv('annual_mean_rainfall.csv')
sea_temp = pd.read_csv('annual_mean_sea_temperature.csv')
carbon = pd.read_csv('annual_carbon_emissions.csv')
pressure = pd.read_csv('annual_mean_pressure.csv')
humidity = pd.read_csv('annual_mean_relative_humidity.csv')
temperature = pd.read_csv('annual_mean_temperature.csv')

data = {'y': rainfall.iloc[:25, 1].values,
        'x1': sea_temp.iloc[:25, 1].values,
        'x2': carbon.iloc[:25, 1].values,
        'x3': pressure.iloc[:25, 1].values,
        'x4': humidity.iloc[:25, 1].values,
        'x5': temperature.iloc[:25, 1].values}
df = pd.DataFrame(data)
x = np.array(df[['x1', 'x2', 'x3', 'x4', 'x5']])
y = np.array(df['y'])

alpha_ridge = [1e-15, 1e-10, 1e-5, 1e-2, 1, 5, 15, 50]

# KFold
for i in range(len(alpha_ridge)):
    RR = Ridge(alpha=alpha_ridge[i])
    kcv = KFold(n_splits=5, random_state=60, shuffle=True)
    RMSE = []
    for train_index, test_index in kcv.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        RR.fit(x_train, y_train)
        RMSE.append(sqrt(mean_squared_error(y_test, np.round(RR.predict(x_test), 2))))
    print(f'For alpha = {alpha_ridge[i]}:')
    print(f'Ridge Regression RMSE: {np.mean(RMSE):.5f}')
    
    # Prediction
    RR.fit(x, y)
    new_data = {'hist_y': rainfall.iloc[25:, 1].values,
                'new_x1': sea_temp.iloc[25:, 1].values,
                'new_x2': carbon.iloc[25:, 1].values,
                'new_x3': pressure.iloc[25:, 1].values,
                'new_x4': humidity.iloc[25:, 1].values,
                'new_x5': temperature.iloc[25:, 1].values}
    new_df = pd.DataFrame(new_data)
    new_x = np.array(new_df[['new_x1', 'new_x2', 'new_x3', 'new_x4', 'new_x5']])
    hist_y = np.array(new_df['hist_y'])
    pred_y = []
    for i in range(5):
        pred = RR.predict(new_x[i, :].reshape(1, -1))
        pred_y.append(round(pred[0], 2))
    pred_RMSE = sqrt(mean_squared_error(hist_y, pred_y))
    print(f'History Data of Y (2018-2022) = {hist_y}')
    print(f'Predicted Y (2018-2022) = {pred_y}')
    print(f'Predicted Y (From RR) RMSE: {pred_RMSE:.5f}')
    print('')
