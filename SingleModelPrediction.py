# RMSE 40.4527818561
import h2o
import math
import numpy as np
from h2o.estimators import H2ORandomForestEstimator
from dataprocessor import ProcessData
from h2o.estimators import H2ODeepLearningEstimator

_validation_ratio = 0.8
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

# Define model
model = H2ODeepLearningEstimator(hidden=[500, 500], score_each_iteration=True, variable_importances=True, epochs=100)

# Train model
model.train(x=training_columns, y=response_column, training_frame=train, validation_frame=validate)

# Model performance
print model.model_performance(test_data=test)

