import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.model_selection import KFold

rainfall = pd.read_csv('annual_mean_daily_rainfall.csv')
sea_temp = pd.read_csv('annual_mean_sea_temperature.csv')
carbon = pd.read_csv('annual_carbon_emissions.csv')

y = rainfall.iloc[:,1].values
x1 = sea_temp.iloc[:,1].values
x2 = carbon.iloc[:,1].values

# x = np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=1)

# Create the plot and Add the data points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y)

# Fit a plane using np.linalg.lstsq
x = np.vstack([x1, x2]).T
plane_coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

# Linear Regression
LR = LinearRegression(fit_intercept=False)
LR.fit(x, y)
print(LR.coef_)

# Create planes
x1_plane, x2_plane = np.meshgrid(x1, x2)
y_plane = plane_coef[0] * x1_plane + plane_coef[1] * x2_plane

# Add the regression plane
ax.plot_surface(x1_plane, x2_plane, y_plane, alpha=0.5)

# Add labels and title
ax.set_xlabel('Sea Temperature (Celsius)')
ax.set_ylabel('Total Greenhouse Gas emissions (kilotonnes CO2-e)')
ax.set_zlabel('Annual Mean Daily Rainfall (mm)')
plt.title('Multiple Linear Regression')
plt.show()
