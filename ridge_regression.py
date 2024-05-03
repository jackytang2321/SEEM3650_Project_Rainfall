import numpy as np
import pandas as pd
from numpy import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

training_data = pd.read_csv('training_data.csv')
data = {'rainfall': training_data.iloc[:, 1].values,
        'carbon': training_data.iloc[:, 2].values,
        'pressure': training_data.iloc[:, 3].values,
        'humidity': training_data.iloc[:, 4].values,
        'temperature': training_data.iloc[:, 5].values}
df = pd.DataFrame(data)
x = df[['carbon', 'pressure', 'humidity', 'temperature']]
y = df[['rainfall']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/9, random_state=21)

# Find optimal alpha for ridge regression
alphas_test = np.linspace(1e-15, 200)
rcv = RidgeCV(alphas=alphas_test, store_cv_values=True, fit_intercept=False)
rcv.fit(x_train, y_train)
optimal_index = rcv.cv_values_.mean(axis=0).argmin()
optimal_alpha = round(alphas_test[optimal_index], 2)

# Ridge Regression
print(f'Results of Ridge Regression with alpha={optimal_alpha}:')
RR = Ridge(alpha=optimal_alpha, fit_intercept=False)
RR.fit(x_train, y_train)
y_pred = RR.predict(x_test)
r2 = r2_score(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f'R-squared Score: {r2:.4}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
