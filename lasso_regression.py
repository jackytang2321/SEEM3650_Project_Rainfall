import numpy as np
import pandas as pd
from numpy import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

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

# Find optimal alpha for lasso regression
alphas_test = np.linspace(1e-15, 200)
lcv = LassoCV(alphas=alphas_test, fit_intercept=False)
lcv.fit(x_train, np.array(y_train).ravel())
optimal_alpha = round(lcv.alpha_, 2)

# Lasso Regression
print(f'Results of Lasso Regression with alpha={optimal_alpha}:')
LS = Lasso(alpha=optimal_alpha, fit_intercept=False)
LS.fit(x_train, y_train)
y_pred = LS.predict(x_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f'R-squared Score: {r2:.4}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
