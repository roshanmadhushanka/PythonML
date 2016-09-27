# Bayesian Ridge Regression

from sklearn import linear_model
from sklearn.metrics import  mean_squared_error, mean_absolute_error
from sklearn_pandas import DataFrameMapper
from sklearn.externals import joblib
from dataprocessor import ProcessData
import numpy as np
import os

# Define response variable
response_column = "RUL"

# Process data
training_frame = ProcessData.trainData(moving_average=True, standard_deviation=True, moving_median=True, probability_distribution=True)
testing_frame = ProcessData.testData(moving_average=True, standard_deviation=True, moving_median=True, probability_distribution=True)

# Select training columns
training_columns = list(training_frame.columns)
training_columns.remove(response_column) # Remove RUL
training_columns.remove("UnitNumber")    # Remove UnitNumber
training_columns.remove("Time")          # Remove Time

# Set mapper
df_mapper = DataFrameMapper([(training_columns, None), (response_column, None)])

# Train data - pandas to sklearn
data = df_mapper.fit_transform(training_frame)

# Test data - pandas to sklearn
test = df_mapper.fit_transform(testing_frame)

# [row : column]
column_count = len(data[0, :])

# train
x = data[:, 0:column_count-1]
# response
y = data[:, column_count-1]
# test
tX = test[:, 0:column_count-1]
# ground truth
tY = test[:, column_count-1]

# Setting up algorithm
rg = linear_model.BayesianRidge()

# Train model
rg.fit(X=x, y=y)

# Get prediction results
result = []
for row in tX:
    if len(row) == 1:
        row = row.reshape(-1, 1)
    elif len(row) > 1:
        row = row.reshape(1, -1)
    result.append(rg.predict(row)[0])

# Analyze performance
print "Performance"
print "-----------"
print "Root Mean Squared Error", mean_squared_error(tY, np.array(result)) ** 0.5
print "Mean Absolute Error", mean_absolute_error(tY, np.array(result))

# Dump pickle files
joblib.dump(df_mapper, "mapper.pkl", compress = 3)
joblib.dump(rg, "estimator.pkl", compress = 3)

# Build pmml
os.system("java -jar converter-executable-1.1-SNAPSHOT.jar --pkl-input estimator.pkl --pmml-output estimator.pmml")


