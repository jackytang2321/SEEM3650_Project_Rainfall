import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

rainfall = pd.read_csv('annual_mean_daily_rainfall.csv')
sea_temp = pd.read_csv('annual_mean_sea_temperature.csv')
carbon = pd.read_csv('annual_carbon_emissions.csv')
y = rainfall.iloc[:25,1].values
x1 = sea_temp.iloc[:25,1].values
x2 = carbon.iloc[:25,1].values

# KFold
x = np.vstack([x1, x2]).T
LR = LinearRegression(fit_intercept=False)
kcv = KFold(n_splits=5, random_state=60, shuffle=True)
MSE = []
for train_index, test_index in kcv.split(x):
    # split the data
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #fit the model
    LR.fit(X_train, y_train)
    MSE.append(mean_squared_error(y_test, LR.predict(X_test)))
print(f'Multi-Linear Regression MSE: {np.mean(MSE):.5f}' + \
      f', STD: {np.std(MSE):.5f}')

# Prediction
hist_y = rainfall.iloc[25:,1].values
new_x1 = sea_temp.iloc[25:,1].values
new_x2 = carbon.iloc[25:,1].values
new_x = np.vstack([new_x1, new_x2]).T
pred_y = []
for i in range(5):
    pred = LR.predict(new_x[i, :].reshape(1, -1))
    pred_y.append(round(pred[0], 2))
pred_MSE = mean_squared_error(hist_y, pred_y)
print(f'Predicted Y (2018-2022) = {pred_y}')
print(f'History Data of Y (2018-2022) = {hist_y}')
print(f'Predicted Y (From LR) MSE: {pred_MSE:.5f}')

# Create 3d-graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y)
plane_coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
x1_plane, x2_plane = np.meshgrid(x1, x2)
y_plane = plane_coef[0] * x1_plane + plane_coef[1] * x2_plane
ax.plot_surface(x1_plane, x2_plane, y_plane, alpha=0.5)
ax.set_xlabel('Sea Temperature (Celsius)')
ax.set_ylabel('Total Greenhouse Gas emissions\n(kilotonnes CO2-e)')
ax.set_zlabel('Annual Mean Daily Rainfall (mm)')
plt.title('Multiple Linear Regression')
plt.show()
