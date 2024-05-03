import numpy as np
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

# We would have 6 different scenarios for the dataset from 2013-2022
# For all 6 scenarios, we will manually change the Carbon Emission only.
# Although Temperature have strong correlation with Carbon Emission, but the changes on it is very low. We could even ingore that.
# Final Solution Package:
prediction_data = pd.read_csv('prediction_data.csv')

# Real Scenario
print('Real Scenario:')
pred_data1 = {'rainfall': prediction_data.iloc[1:11, 1].values,
        'carbon': prediction_data.iloc[1:11, 2].values,
        'pressure': prediction_data.iloc[1:11, 3].values,
        'humidity': prediction_data.iloc[1:11, 4].values,
        'temperature': prediction_data.iloc[1:11, 5].values}
pred_df = pd.DataFrame(pred_data1)
x = pred_df[['carbon', 'pressure', 'humidity', 'temperature']]
y = pred_df[['rainfall']]
y_base_pred = LR.predict(x)
r2 = r2_score(y, y_base_pred)
mae = mean_absolute_error(y, y_base_pred)
rmse = sqrt(mean_squared_error(y, y_base_pred))
print(f'R-squared Score: {r2:.4}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'Real Mean Rainfall: {np.mean(y)}')
print(f'Predicted Mean Rainfall: {np.mean(y_base_pred)}')
print('')

# Worst Scenario (Carbon Emission +25%)
print('Worst Scenario (Carbon Emission +25%):')
pred_data2 = {'rainfall': prediction_data.iloc[12:22, 1].values,
        'carbon': prediction_data.iloc[12:22, 2].values,
        'pressure': prediction_data.iloc[12:22, 3].values,
        'humidity': prediction_data.iloc[12:22, 4].values,
        'temperature': prediction_data.iloc[12:22, 5].values}
pred_df = pd.DataFrame(pred_data2)
x = pred_df[['carbon', 'pressure', 'humidity', 'temperature']]
y = pred_df[['rainfall']]
y_pred = LR.predict(x)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = sqrt(mean_squared_error(y, y_pred))
print(f'R-squared Score: {r2:.4}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'Real Mean Rainfall: {np.mean(y)}')
print(f'Predicted Mean Rainfall: {np.mean(y_pred)}')
diff = (np.mean(y_pred) - np.mean(y_base_pred))/np.mean(y_base_pred) * 100
print(f'Under this scenario, Predicted rainfall change by {diff:.2f}%')
print('')

# Worse Scenario (Carbon Emission +15%)
print('Worse Scenario (Carbon Emission +15%):')
pred_data3 = {'rainfall': prediction_data.iloc[23:33, 1].values,
        'carbon': prediction_data.iloc[23:33, 2].values,
        'pressure': prediction_data.iloc[23:33, 3].values,
        'humidity': prediction_data.iloc[23:33, 4].values,
        'temperature': prediction_data.iloc[23:33, 5].values}
pred_df = pd.DataFrame(pred_data3)
x = pred_df[['carbon', 'pressure', 'humidity', 'temperature']]
y = pred_df[['rainfall']]
y_pred = LR.predict(x)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = sqrt(mean_squared_error(y, y_pred))
print(f'R-squared Score: {r2:.4}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'Real Mean Rainfall: {np.mean(y)}')
print(f'Predicted Mean Rainfall: {np.mean(y_pred)}')
diff = (np.mean(y_pred) - np.mean(y_base_pred))/np.mean(y_base_pred) * 100
print(f'Under this scenario, Predicted rainfall change by {diff:.2f}%')
print('')

# Better Scenario (Carbon Emission -15%)
print('Better Scenario (Carbon Emission -15%):')
pred_data4 = {'rainfall': prediction_data.iloc[34:44, 1].values,
        'carbon': prediction_data.iloc[34:44, 2].values,
        'pressure': prediction_data.iloc[34:44, 3].values,
        'humidity': prediction_data.iloc[34:44, 4].values,
        'temperature': prediction_data.iloc[34:44, 5].values}
pred_df = pd.DataFrame(pred_data4)
x = pred_df[['carbon', 'pressure', 'humidity', 'temperature']]
y = pred_df[['rainfall']]
y_pred = LR.predict(x)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = sqrt(mean_squared_error(y, y_pred))
print(f'R-squared Score: {r2:.4}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'Real Mean Rainfall: {np.mean(y)}')
print(f'Predicted Mean Rainfall: {np.mean(y_pred)}')
diff = (np.mean(y_pred) - np.mean(y_base_pred))/np.mean(y_base_pred) * 100
print(f'Under this scenario, Predicted rainfall change by {diff:.2f}%')
print('')

# Climate Action Plan 2030+ Scenario 1 (Carbon Emission -25%)
print('Climate Action Plan 2030+ Scenario 1 (Carbon Emission -25%):')
pred_data5 = {'rainfall': prediction_data.iloc[45:55, 1].values,
        'carbon': prediction_data.iloc[45:55, 2].values,
        'pressure': prediction_data.iloc[45:55, 3].values,
        'humidity': prediction_data.iloc[45:55, 4].values,
        'temperature': prediction_data.iloc[45:55, 5].values}
pred_df = pd.DataFrame(pred_data5)
x = pred_df[['carbon', 'pressure', 'humidity', 'temperature']]
y = pred_df[['rainfall']]
y_pred = LR.predict(x)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = sqrt(mean_squared_error(y, y_pred))
print(f'R-squared Score: {r2:.4}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'Real Mean Rainfall: {np.mean(y)}')
print(f'Predicted Mean Rainfall: {np.mean(y_pred)}')
diff = (np.mean(y_pred) - np.mean(y_base_pred))/np.mean(y_base_pred) * 100
print(f'Under this scenario, Predicted rainfall change by {diff:.2f}%')
print('')

# Climate Action Plan 2030+ Scenario 2 (Carbon Emission -35%)
print('Climate Action Plan 2030+ Scenario 1 (Carbon Emission -25%):')
pred_data6 = {'rainfall': prediction_data.iloc[56:66, 1].values,
        'carbon': prediction_data.iloc[56:66, 2].values,
        'pressure': prediction_data.iloc[56:66, 3].values,
        'humidity': prediction_data.iloc[56:66, 4].values,
        'temperature': prediction_data.iloc[56:66, 5].values}
pred_df = pd.DataFrame(pred_data6)
x = pred_df[['carbon', 'pressure', 'humidity', 'temperature']]
y = pred_df[['rainfall']]
y_pred = LR.predict(x)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = sqrt(mean_squared_error(y, y_pred))
print(f'R-squared Score: {r2:.4}')
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'Real Mean Rainfall: {np.mean(y)}')
print(f'Predicted Mean Rainfall: {np.mean(y_pred)}')
diff = (np.mean(y_pred) - np.mean(y_base_pred))/np.mean(y_base_pred) * 100
print(f'Under this scenario, Predicted rainfall change by {diff:.2f}%')
print('')
