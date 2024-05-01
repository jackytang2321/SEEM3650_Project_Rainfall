import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data
rainfall = pd.read_csv('annual_mean_daily_rainfall.csv')
sea_temp = pd.read_csv('annual_mean_sea_temperature.csv')
carbon = pd.read_csv('annual_carbon_emissions.csv')
y = rainfall.iloc[:25, 1].values
x1 = sea_temp.iloc[:25, 1].values
x2 = carbon.iloc[:25, 1].values

# Create 3d-graph
x = np.vstack([x1, x2]).T
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