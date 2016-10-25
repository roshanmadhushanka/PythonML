# Recursive Feature Elimination

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn_pandas import DataFrameMapper

from dataprocessor import ProcessData

training_frame = ProcessData.trainData(moving_k_closest_average=True, standard_deviation=True)
testing_frame = ProcessData.testData(moving_k_closest_average=True, standard_deviation=True)

# Training data columns
del training_frame['UnitNumber']
del training_frame['Time']

# Testing columns
del testing_frame['UnitNumber']
del testing_frame['Time']


training_columns = list(training_frame.columns)
training_columns.remove('RUL')
response_column = 'RUL'

# Set mapper
df_mapper = DataFrameMapper([(training_columns, None), (response_column, None)])

# Train data - pandas to sklearn
data = df_mapper.fit_transform(training_frame)

# Test data - pandas to sklearn
test = df_mapper.fit_transform(testing_frame)

features = df_mapper.features[0][0]
response = df_mapper.features[1][0]


# [row : column]
column_count = len(data[0, :])

# Sklearn data
x_train = data[:, 0:column_count-1]
y_train = data[:, column_count-1]
x_test = test[:, 0:column_count-1]
y_test = test[:, column_count-1]

# Setting up algorithm
rf = RandomForestRegressor(max_depth=10, n_estimators=10)

rfe = RFE(rf, 20)
rfe = rfe.fit(x_train, y_train)

# summarize the selection of the attributes
support = rfe.ranking_
ranking = rfe.ranking_

for i in xrange(len(support)):
    if support[i] == True:
        print features[i]
print(rfe.support_)
print(rfe.ranking_)
