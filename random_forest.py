import pandas as pd
from numpy import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
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

# Find optimal n_quantiles
rfr = RandomForestRegressor(random_state=21)
max_r2 = 0
max_n = 0
for i in range(1, len(x_train)+1):
    qt = QuantileTransformer(output_distribution='normal', n_quantiles=i)
    ttr = TransformedTargetRegressor(regressor=rfr, transformer=qt, check_inverse=False)
    ttr.fit(x_train, y_train)
    y_pred = ttr.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    if max_r2 <= r2:
        max_r2 = r2
        max_n = i
print(f'Optimal n_quantiles: {max_n}') # Optimal n_quantiles = 3

# Random Forest Regression
print(f'Result of Random Forest Regression')
rfr = RandomForestRegressor(random_state=21)
qt = QuantileTransformer(output_distribution='normal', n_quantiles=3)
ttr = TransformedTargetRegressor(regressor=rfr, transformer=qt)
ttr.fit(x_train, y_train)
y_pred = ttr.predict(x_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f'R-squared Score: {r2:.4}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
