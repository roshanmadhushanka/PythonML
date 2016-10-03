import h2o
from h2o.estimators import H2ODeepLearningEstimator, H2OGeneralizedLinearEstimator, H2OGradientBoostingEstimator, H2ORandomForestEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn_pandas import DataFrameMapper
import numpy as np

from dataprocessor import ProcessData

# setting up h2o server
h2o.init()

# define response variable
response_column = 'RUL'

# Begin : Deep Learning
# ----------------------------------------------------------------------------------------------------------------------
# MKA, SD, PROB
_model_name_1 = "Deep Learning"
print "Model : " + _model_name_1
print "-------------------------"

# load pre-processed data frames
training_frame = ProcessData.trainData(moving_k_closest_average=True, standard_deviation=True, probability_distribution=True)
testing_frame = ProcessData.testData(standard_deviation=True, moving_k_closest_average=True, probability_from_file=True)

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

# Build model
model1 = H2ODeepLearningEstimator()

# Train model
model1.train(x=training_columns, y=response_column, training_frame=train)

# End : Deep Learning
# ----------------------------------------------------------------------------------------------------------------------


# Begin : Random Forest Regression
# ----------------------------------------------------------------------------------------------------------------------
# MKA, SD, PROB
_model_name_2 = "Random Forest Regression"
print "Model : " + _model_name_2
print "-------------------------"

# load pre-processed data frames
training_frame = ProcessData.trainData(moving_k_closest_average=True, standard_deviation=True, probability_distribution=True)
testing_frame = ProcessData.testData(standard_deviation=True, moving_k_closest_average=True, probability_from_file=True)

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

# Build model
model2 = H2ORandomForestEstimator()

# Train model
model2.train(x=training_columns, y=response_column, training_frame=train)

# End : Random Forest Regression
# ----------------------------------------------------------------------------------------------------------------------


# Begin : Gradient Boosting
# ----------------------------------------------------------------------------------------------------------------------
# MKA, SD, PROB
_model_name_3 = "Gradient Boosting"
print "Model : " + _model_name_3
print "-------------------------"

# load pre-processed data frames
training_frame = ProcessData.trainData(moving_k_closest_average=True, standard_deviation=True, probability_distribution=True)
testing_frame = ProcessData.testData(standard_deviation=True, moving_k_closest_average=True, probability_from_file=True)

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

# Build model
model3 = H2OGradientBoostingEstimator()

# Train model
model3.train(x=training_columns, y=response_column, training_frame=train)

# End : Gradient Boosting
# ----------------------------------------------------------------------------------------------------------------------


# Begin : Generalized Linear Modeling
# ----------------------------------------------------------------------------------------------------------------------
# MA, SD, PROB
_model_name_4 = "Generalized Linear Modeling"
print "Model : " + _model_name_4
print "-------------------------"

# load pre-processed data frames
training_frame = ProcessData.trainData(moving_average=True, standard_deviation=True, probability_distribution=True)
testing_frame = ProcessData.testData(moving_average=True, standard_deviation=True, probability_from_file=True)

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

# Build model
model4 = H2OGeneralizedLinearEstimator()

# Train model
model4.train(x=training_columns, y=response_column, training_frame=train)

# End : Generalized Linear Modeling
# ----------------------------------------------------------------------------------------------------------------------

# Prediction
# ----------------------------------------------------------------------------------------------------------------------
print "Begin Prdiction"
print "---------------"

# ground truth
tY = np.array(testing_frame['RUL'])

predicted1 = model1.predict(test)
print "Complete : " + _model_name_1

predicted2 = model2.predict(test)
print "Complete : " + _model_name_2

predicted3 = model3.predict(test)
print "Complete : " + _model_name_3

predicted4 = model4.predict(test)
print "Complete : " + _model_name_4

# ----------------------------------------------------------------------------------------------------------------------

print ""

# Evaluating results
# ----------------------------------------------------------------------------------------------------------------------
print "Evaluating results"
print "------------------"

predicted_vals = np.zeros(shape=100)
for i in range(len(test[:,0])):
    tmp = []
    tmp.append(predicted1[i, 0])
    tmp.append(predicted2[i, 0])
    tmp.append(predicted3[i, 0])
    tmp.append(predicted4[i, 0])
    tmp.sort()
    print tmp
    predicted_vals[i] = (sum(tmp[1:-1]) / float(len(tmp[1:-1])))


print "Mean Squared Error  :", mean_squared_error(tY, predicted_vals)
print "Mean Absolute Error :", mean_absolute_error(tY, predicted_vals)
