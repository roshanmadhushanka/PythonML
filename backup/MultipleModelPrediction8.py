# RMSE 30.27
'''
Multiple models predict the value
Based on that predicted values create a histogram of k number of bins
Select the most predicted value
'''

# h2o testing
import math

import h2o
import numpy as np
from h2o.estimators import H2ODeepLearningEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dataprocessor import ProcessData

_nmodels = 10  # SD, MKA, PROB
_nbins = 5
_validation_ratio = 0.8

# _nmodels = 20
# _lim = 7

# initialize server
h2o.init()

# get processed data
pTrain = ProcessData.trainData(moving_k_closest_average=True, standard_deviation=True, probability_distribution=True)
pTest = ProcessData.testData(moving_k_closest_average=True, standard_deviation=True, probability_from_file=True)

# convert to h2o frames
hTrain = h2o.H2OFrame(pTrain)
hTest = h2o.H2OFrame(pTest)
hTrain.set_names(list(pTrain.columns))
hTest.set_names(list(pTest.columns))

# select column names
response_column = 'RUL'

training_columns = list(pTrain.columns)
training_columns.remove(response_column)
training_columns.remove("UnitNumber")
training_columns.remove("Time")

# split frames
train, validate = hTrain.split_frame([_validation_ratio])
test = hTest
ground_truth = np.array(pTest['RUL'])

# model array
model_arry = range(_nmodels)

# Building model
print "Building models"
print "---------------"
for i in range(_nmodels):
    model_arry[i] = H2ODeepLearningEstimator(hidden=[200, 200], score_each_iteration=True, variable_importances=True, epochs=100)

print "Training models"
print "---------------"
for i in range(_nmodels):
    print "Train : " + str(i + 1) + "/" + str(_nmodels)
    model_arry[i].train(x=training_columns, y=response_column, training_frame=train, validation_frame=validate)

print "Error in models"
print "---------------"
for i in range(_nmodels):
    print model_arry[i].mse(model_arry[i].model_performance(test_data=test))

predicted_arr = range(_nmodels)
for i in range(_nmodels):
    predicted_arr[i] = model_arry[i].predict(test)

predicted_vals = np.zeros(shape=100)
for i in range(len(test[:, 0])):
    tmp = []
    for j in range(_nmodels):
        tmp.append(predicted_arr[j][i, 0])
    p, x = np.histogram(tmp, bins=_nbins)
    p = list(p)
    x = list(x)
    k = p.index(max(p)) # Index of most occurring value
    lst = [y for y in tmp if (y >= x[k] and y <= x[k+1])]
    print "P",p
    print "X",x
    print "tmp", tmp
    print "lst", lst
    print "val", sum(lst) / float(len(lst))
    print "------------------------------------------------------------------------------------------------------------"
    #predicted_vals[i] = x[p.index(max(p))+1]
    predicted_vals[i] = sum(lst) / float(len(lst))

print "Root Mean Squared Error :", math.sqrt(mean_squared_error(ground_truth, predicted_vals))
print "Mean Absolute Error     :", mean_absolute_error(ground_truth, predicted_vals)

# frame.split_frame([0.7])