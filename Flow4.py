from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_squared_error, mean_absolute_error
from sklearn_pandas import DataFrameMapper
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import os

# Loading data
training_frame = pd.read_csv("Training.csv")
testing_frame = pd.read_csv("Testing.csv")

# Feature selection
training_columns = [u'Setting1', u'Setting2', u'Setting3', u'Sensor1', u'Sensor2', u'Sensor3', u'Sensor4', u'Sensor5',
                    u'Sensor6', u'Sensor7', u'Sensor8', u'Sensor9', u'Sensor10', u'Sensor11', u'Sensor12', u'Sensor13',
                    u'Sensor14', u'Sensor15', u'Sensor16', u'Sensor17', u'Sensor18', u'Sensor19', u'Sensor20',
                    u'Sensor21', u'Sensor1_ma_5', u'Sensor2_ma_5', u'Sensor3_ma_5', u'Sensor4_ma_5', u'Sensor5_ma_5',
                    u'Sensor6_ma_5', u'Sensor7_ma_5', u'Sensor8_ma_5', u'Sensor9_ma_5', u'Sensor10_ma_5',
                    u'Sensor11_ma_5', u'Sensor12_ma_5', u'Sensor13_ma_5', u'Sensor14_ma_5', u'Sensor15_ma_5',
                    u'Sensor16_ma_5', u'Sensor17_ma_5', u'Sensor18_ma_5', u'Sensor19_ma_5', u'Sensor20_ma_5',
                    u'Sensor21_ma_5', u'Sensor1_mm_5', u'Sensor2_mm_5', u'Sensor3_mm_5', u'Sensor4_mm_5',
                    u'Sensor5_mm_5', u'Sensor6_mm_5', u'Sensor7_mm_5', u'Sensor8_mm_5', u'Sensor9_mm_5',
                    u'Sensor10_mm_5', u'Sensor11_mm_5', u'Sensor12_mm_5', u'Sensor13_mm_5', u'Sensor14_mm_5',
                    u'Sensor15_mm_5', u'Sensor16_mm_5', u'Sensor17_mm_5', u'Sensor18_mm_5', u'Sensor19_mm_5',
                    u'Sensor20_mm_5', u'Sensor21_mm_5', u'Sensor1_sd_10', u'Sensor2_sd_10', u'Sensor3_sd_10',
                    u'Sensor4_sd_10', u'Sensor5_sd_10', u'Sensor6_sd_10', u'Sensor7_sd_10', u'Sensor8_sd_10',
                    u'Sensor9_sd_10', u'Sensor10_sd_10', u'Sensor11_sd_10', u'Sensor12_sd_10', u'Sensor13_sd_10',
                    u'Sensor14_sd_10', u'Sensor15_sd_10', u'Sensor16_sd_10', u'Sensor17_sd_10', u'Sensor18_sd_10',
                    u'Sensor19_sd_10', u'Sensor20_sd_10', u'Sensor21_sd_10', u'Sensor1_entropy_250',
                    u'Sensor2_entropy_250', u'Sensor3_entropy_250', u'Sensor4_entropy_250', u'Sensor5_entropy_250',
                    u'Sensor6_entropy_250', u'Sensor7_entropy_250', u'Sensor8_entropy_250', u'Sensor9_entropy_250',
                    u'Sensor10_entropy_250', u'Sensor11_entropy_250', u'Sensor12_entropy_250', u'Sensor13_entropy_250',
                    u'Sensor14_entropy_250', u'Sensor15_entropy_250', u'Sensor16_entropy_250', u'Sensor17_entropy_250',
                    u'Sensor18_entropy_250', u'Sensor19_entropy_250', u'Sensor20_entropy_250', u'Sensor21_entropy_250']

response_column = u'RUL'

# Parsing data
training_data = training_frame[training_columns]
target_data = training_frame[response_column]
testing_data = testing_frame[training_columns]
ground_truth_data = testing_frame[response_column]

# Setting up mapper
df_mapper = DataFrameMapper([(training_columns, None), (response_column, None)])

# Train data - pandas to sklearn
data = df_mapper.fit_transform(training_frame)
# train
x = data[:, 0:108]
# response
y = data[:, 108]

# Test data - pandas to sklearn
test = df_mapper.fit_transform(testing_frame)
# test
tX = test[:, 0:108]
# ground truth
tY = test[:, 108]

# Setting up algorithm
rf = RandomForestRegressor(max_depth=20)

# Train model
rf.fit(X=x, y=y)

# Get prediction results
result = []
for row in tX:
    if len(row) == 1:
        row = row.reshape(-1, 1)
    elif len(row) > 1:
        row = row.reshape(1, -1)
    result.append(rf.predict(row)[0])

# Analyze performance
print "Performance"
print "-----------"
print "Root Mean Squared Error", mean_squared_error(tY, np.array(result)) ** 0.5
print "Mean Absolute Error", mean_absolute_error(tY, np.array(result))

# Dump pickle files
joblib.dump(df_mapper, "mapper.pkl", compress = 3)
joblib.dump(rf, "estimator.pkl", compress = 3)

# Build pmml
os.system("java -jar converter-executable-1.1-SNAPSHOT.jar --pkl-input estimator.pkl --pmml-output estimator.pmml")



'''
Scikit learn model to pmml
https://github.com/jpmml/sklearn2pmml
https://pypi.python.org/pypi/sklearn-pmml/0.1.0#downloads
'''





