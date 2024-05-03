import pandas as pd
from numpy import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# Linear Regression
print('Result of Linear Regression:')
LR = LinearRegression(fit_intercept=False)
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f'R-squared Score: {r2:.4}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
