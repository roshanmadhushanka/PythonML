
#RMSE 43.9605178314
import h2o
import math
import numpy as np
from sklearn.metrics import mean_squared_error

from dataprocessor import ProcessData
from h2o.estimators import H2ODeepLearningEstimator

_validation_ratio = 0.8

# initialize server
h2o.init()

# define response variable
response_column = 'RUL'

# load pre-processed data frames
training_frame = ProcessData.trainData(moving_k_closest_average=True, standard_deviation=True, probability_distribution=True)
testing_frame = ProcessData.testData(moving_k_closest_average=True, standard_deviation=True, probability_from_file=True)

# create h2o frames
data = h2o.H2OFrame(training_frame)
test = h2o.H2OFrame(testing_frame)
data.set_names(list(training_frame.columns))
test.set_names(list(testing_frame.columns))

train, validate = data.split_frame([_validation_ratio])

# Feature selection
training_columns = list(training_frame.columns)
training_columns.remove(response_column)
training_columns.remove("UnitNumber")
training_columns.remove("Time")

for layer in range(100, 1001, 10):
    print "Layer", layer
    # Define model
    model = H2ODeepLearningEstimator(hidden=[layer, layer], score_each_iteration=True, variable_importances=True, epochs=100)

    # Train model
    model.train(x=training_columns, y=response_column, training_frame=train, validation_frame=validate)

    # Evaluate model
    print model.model_performance(test_data=test)
