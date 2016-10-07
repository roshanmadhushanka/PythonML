import h2o
import math
import numpy as np
from dataprocessor import ProcessData
from h2o.estimators import H2ODeepLearningEstimator

# initialize server
h2o.init()

# define response variable
response_column = 'RUL'

# load pre-processed data frames
training_frame = ProcessData.trainData(moving_k_closest_average=True, standard_deviation=True, probability_distribution=True)
testing_frame = ProcessData.testData(moving_k_closest_average=True, standard_deviation=True, probability_from_file=True)

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

# Define model
model = H2ODeepLearningEstimator(hidden=[500, 500], score_each_iteration=True, variable_importances=True, epochs=100)

# Train model
model.train(x=training_columns, y=response_column, training_frame=train)

# Evaluate model
mean_squared_error = model.mse(model.model_performance(test_data=test))

# Output
print "Root Mean Squared Error", math.sqrt(mean_squared_error)

# Print predictions
predict = list(model.predict(test_data=test))
print "Predict"
print predict

testY = np.array(testing_frame['RUL'])

print "Root Mean Squared Error :", math.sqrt(mean_squared_error(testY, np.array(list(predict))))


