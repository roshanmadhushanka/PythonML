'''
Multiple Models in
SciKit Learn - RFR
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.metrics import  mean_squared_error, mean_absolute_error
from sklearn_pandas import DataFrameMapper
from sklearn.externals import joblib
from dataprocessor import ProcessData
import numpy as np
import math

# parameters
_nmodels = 10
_nbins = 5

# define response variable
response_column = 'RUL'

# load pre-processed data frames
training_frame = ProcessData.trainData()
testing_frame = ProcessData.testData()

# Feature selection
training_columns = list(training_frame.columns)
training_columns.remove(response_column)
training_columns.remove("UnitNumber")
training_columns.remove("Time")

# Set mapper
df_mapper = DataFrameMapper([(training_columns, None), (response_column, None)])

# Train data - pandas to sklearn
data = df_mapper.fit_transform(training_frame)

# Test data - pandas to sklearn
test = df_mapper.fit_transform(testing_frame)

# [row : column]
column_count = len(data[0, :])

# train
trainX = data[:, 0:column_count-1]
# response
trainY = data[:, column_count-1]
# test
testX = test[:, 0:column_count-1]
# ground truth
testY = test[:, column_count-1]

model_arry = range(_nmodels)

# Building model
print "Building models"
print "---------------"
for i in range(_nmodels):
    model_arry[i] = RandomForestRegressor(max_depth=20, n_estimators=50)

print "Training models"
print "---------------"
for i in range(_nmodels):
    print "Train : " + str(i + 1) + "/" + str(_nmodels)
    model_arry[i].fit(X=trainX, y=trainY)


predicted_arr = range(_nmodels)
for i in range(_nmodels):
    predicted_arr[i] = model_arry[i].predict(testX)

predicted_vals = np.zeros(shape=100)
for i in range(len(test[:, 0])):
    tmp = []
    for j in range(_nmodels):
        tmp.append(predicted_arr[j][i])
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

print "Root Mean Squared Error :", math.sqrt(mean_squared_error(testY, predicted_vals))
print "Mean Absolute Error     :", mean_absolute_error(testY, predicted_vals)
