import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.model_selection import KFold

rainfall = pd.read_csv('daily_HK_RF_ALL_AVG.csv')
humidity = pd.read_csv('Daily_Mean_Relative_Humidity.csv')
pressure = pd.read_csv('daily_MSLP.csv')

x1 = humidity.iloc[:,1].values
x2 = pressure.iloc[:,1].values
y = rainfall.iloc[:,1].values

# x = np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=1)

# Create the plot and Add the data points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y)

# Fit a plane using np.linalg.lstsq
x = np.vstack([x1, x2, np.ones_like(x1)]).T
plane_coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

# Linear Regression
LR = LinearRegression(fit_intercept=False)
LR.fit(x, y)
print(LR.coef_)

# Create a meshgrid for the plane
x1_plane, x2_plane = np.meshgrid(x1, x2)
y_plane = plane_coef[0] * x1_plane + plane_coef[1] * x2_plane + plane_coef[2]

# Add the regression plane
ax.plot_surface(x1_plane, x2_plane, y_plane, alpha=0.5)

# Add labels and title
ax.set_xlabel('Mean Relative Humidity (%)')
ax.set_ylabel('Mean Pressure (hPa)')
ax.set_zlabel('Mean Rainfall (mm)')
plt.title('Multiple Linear Regression')
plt.show()