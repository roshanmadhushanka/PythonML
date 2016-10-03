# h2o testing
import h2o
import numpy as np
from h2o.estimators import H2ODeepLearningEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn_pandas import DataFrameMapper
from dataprocessor import ProcessData
import math

_nmodels = 10
_lim = 2

# initialize server
h2o.init()

# define response variable
response_column = 'RUL'

# load pre-processed data frames
training_frame = ProcessData.trainData(standard_deviation=True, moving_k_closest_average=True)
testing_frame = ProcessData.testData(standard_deviation=True, moving_k_closest_average=True)

# create h2o frames
train = h2o.H2OFrame(training_frame)
test = h2o.H2OFrame(testing_frame)
train.set_names(list(training_frame.columns))
test.set_names(list(testing_frame.columns))

# Feature selection
training_columns = list(training_frame.columns)
training_columns.remove(response_column)
training_columns.remove("UnitNumber")
training_columns.remove("Time")

# Set mapper
df_mapper = DataFrameMapper([(training_columns, None), (response_column, None)])

# Test data - pandas to sklearn
test_tmp = df_mapper.fit_transform(testing_frame)

# [row : column]
column_count = len(test_tmp[0, :])

# ground truth
tY = np.array(testing_frame['RUL'])

# model array
model_arry = range(_nmodels)

# Building model
print "Building models"
print "---------------"
for i in range(_nmodels):
    model_arry[i] = H2ODeepLearningEstimator(hidden=[200, 200], score_each_iteration=False, variable_importances=True)


print "Training models"
print "---------------"
for i in range(_nmodels):
    model_arry[i].train(x=training_columns, y=response_column, training_frame=train)


predicted_arr = range(_nmodels)
for i in range(_nmodels):
    predicted_arr[i] = model_arry[i].predict(test)


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
