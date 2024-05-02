from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

rainfall = pd.read_csv('annual_mean_rainfall.csv')
carbon = pd.read_csv('annual_carbon_emissions.csv')
pressure = pd.read_csv('annual_mean_pressure.csv')
humidity = pd.read_csv('annual_mean_relative_humidity.csv')
temperature = pd.read_csv('annual_mean_temperature.csv')

data = {'y': rainfall.iloc[:63, 1].values,
        'x1': carbon.iloc[:63, 1].values,
        'x2': pressure.iloc[:63, 1].values,
        'x3': humidity.iloc[:63, 1].values,
        'x4': temperature.iloc[:63, 1].values}
df = pd.DataFrame(data)
x = np.array(df[['x1', 'x2', 'x3', 'x4']])
y = np.array(df['y'])

alpha_lasso = [1e-15, 1e-10, 1e-5, 1e-2, 1, 5, 15, 50]

# KFold
for i in range(len(alpha_lasso)):
    LS = Lasso(alpha=alpha_lasso[i])
    kcv = KFold(n_splits=3, random_state=60, shuffle=True)
    RMSE = []
    R2 = []
    for train_index, test_index in kcv.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        LS.fit(x_train, y_train)
        R2.append(r2_score(y_test, np.round(LS.predict(x_test), 2)))
        RMSE.append(sqrt(mean_squared_error(y_test, np.round(LS.predict(x_test), 2))))
    print(f'For alpha = {alpha_lasso[i]}:')
    print(f'Lasso Regression R-squared: {np.mean(R2):.5f}')
    print(f'Lasso Regression RMSE: {np.mean(RMSE):.5f}')
    
    # Prediction
    LS.fit(x, y)
    new_data = {'hist_y': rainfall.iloc[63:, 1].values,
                'new_x1': carbon.iloc[63:, 1].values,
                'new_x2': pressure.iloc[63:, 1].values,
                'new_x3': humidity.iloc[63:, 1].values,
                'new_x4': temperature.iloc[63:, 1].values}
    new_df = pd.DataFrame(new_data)
    new_x = np.array(new_df[['new_x1', 'new_x2', 'new_x3', 'new_x4']])
    hist_y = np.array(new_df['hist_y'])
    pred_y = []
    for i in range(10):
        pred = LS.predict(new_x[i, :].reshape(1, -1))
        pred_y.append(round(pred[0], 2))
    pred_R2 = r2_score(hist_y, pred_y)
    pred_RMSE = sqrt(mean_squared_error(hist_y, pred_y))
    print(f'History Data of Y (2013-2022) = {hist_y}')
    print(f'Predicted Y (2013-2022) = {pred_y}')
    print(f'Predicted Y (From Lasso Regression) R-squared: {pred_R2:.5f}')
    print(f'Predicted Y (From Lasso Regression) RMSE: {pred_RMSE:.5f}')
    print('')