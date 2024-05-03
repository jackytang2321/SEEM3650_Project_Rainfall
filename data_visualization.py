import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

training_data = pd.read_csv('training_data.csv')
data = {'rainfall': training_data.iloc[:, 1].values,
        'carbon': training_data.iloc[:, 2].values,
        'pressure': training_data.iloc[:, 3].values,
        'humidity': training_data.iloc[:, 4].values,
        'temperature': training_data.iloc[:, 5].values}
df = pd.DataFrame(data)

# Visualization of Training Data
sns.pairplot(data=df, diag_kind='kde', kind='reg')
plt.savefig('pairplot of training data.png')
plt.show()
sns.heatmap(df[['rainfall', 'carbon', 'pressure', 'humidity', 'temperature']].corr(), cmap='Blues', annot=True, fmt='.4f')
plt.savefig('heatmap of training data.png')
plt.show()