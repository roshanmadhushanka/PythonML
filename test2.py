# MinMaxScaler
import pandas as pd
import numpy as np

from sklearn import preprocessing
from dataprocessor import ProcessData

# replace mean with min
# df_norm = (df - df.mean()) / (df.max() - df.min())
# df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

df = ProcessData.trainData()
response_column = 'RUL'
training_columns = list(df.columns)
training_columns.remove(response_column)
training_columns.remove('UnitNumber')
training_columns.remove('Time')

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df[df.columns[:-1]])
df_normalized = pd.DataFrame(x_scaled, columns=df.columns[:-1])
print df_normalized

