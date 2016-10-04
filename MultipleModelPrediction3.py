# h2o testing
import h2o
import numpy as np
from h2o.estimators import H2ODeepLearningEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn_pandas import DataFrameMapper
from dataprocessor import ProcessData
import math

h2o.init()

_nmodels = 10
_lim = 2

# define response variable
response_column = 'RUL'

# Pandas frame
data_frame = ProcessData.trainData()
test_frame = ProcessData.testData()

# Create h2o frame
h2o_data = h2o.H2OFrame(data_frame)
h2o_data.set_names(list(data_frame.columns))

h2o_test = h2o.H2OFrame(test_frame)
h2o_test.set_names(list(test_frame.columns))

# split frame
data = h2o_data.split_frame(ratios=(0.9, 0.09))

# split data
train_data = data[0]
validate_data = data[1]
test_data = h2o_test

# Feature selection
training_columns = list(data_frame.columns)
training_columns.remove(response_column)
training_columns.remove("UnitNumber")
training_columns.remove("Time")

# ground truth
ground_truth = np.array(test_frame['RUL'])
validate_rsponse = np.array(validate_data['RUL'])

# model array
model_arry = range(_nmodels)

# Building model
print "Building models"
print "---------------"
for i in range(_nmodels):
    model_arry[i] = H2ODeepLearningEstimator(hidden=[200, 200], score_each_iteration=True, variable_importances=True)


print "Training models"
print "---------------"
for i in range(_nmodels):
    model_arry[i].train(x=training_columns, y=response_column, training_frame=train_data)


predicted_arr = range(_nmodels)
for i in range(_nmodels):
    predicted_arr[i] = model_arry[i].predict(validate_data)

error_arr = np.zeros(shape=_nmodels)
for i in range(_nmodels):
    error_arr[i] = mean_squared_error()

predicted_vals = np.zeros(shape=100)
for i in range(len(test[:,0])):
    tmp = []
    for j in range(_nmodels):
        tmp.append(predicted_arr[j][i, 0])
    tmp.sort()
    print tmp
    predicted_vals[i] = (sum(tmp[_lim:-_lim]) / float(len(tmp[_lim:-_lim])))


print "Root Mean Squared Error :", math.sqrt(mean_squared_error(tY, predicted_vals))
print "Mean Absolute Error     :", mean_absolute_error(tY, predicted_vals)


