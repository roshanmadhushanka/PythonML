# Siddhi Test

from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import  mean_squared_error, mean_absolute_error
from sklearn_pandas import DataFrameMapper
from sklearn.externals import joblib
from dataprocessor import ProcessData
import numpy as np
import os

# Define file names
mapper_pkl = "mapper.pkl"
estimator_pkl = "estimator.pkl"
estimator_pmml = "model.pmml"

# Define response variable
response_column = "RUL"

# Process data
training_frame = ProcessData.trainData(moving_k_closest_average=True, standard_deviation=True, moving_entropy=True)
testing_frame = ProcessData.testData(moving_k_closest_average=True, standard_deviation=True, moving_entropy=True)

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
rf = RandomForestRegressor(max_depth=10, n_estimators=10)

# Train model
rf.fit(X=x, y=y)

# Get prediction results
result = rf.predict(tX)

print "Result"
print "------"
print result

# Analyze performance
print "Performance"
print "-----------"
print "Root Mean Squared Error", mean_squared_error(tY, np.array(result)) ** 0.5
print "Mean Absolute Error", mean_absolute_error(tY, np.array(result))

# Dump pickle files
print df_mapper.features
print rf.get_params()

joblib.dump(df_mapper, mapper_pkl, compress = 3)
joblib.dump(rf, estimator_pkl, compress = 3)

# Build pmml
command = "java -jar converter-executable-1.1-SNAPSHOT.jar --pkl-mapper-input mapper.pkl --pkl-estimator-input estimator.pkl --pmml-output mapper-estimator.pmml"
os.system(command)


