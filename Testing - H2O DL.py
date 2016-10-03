# h2o testing
import h2o
from h2o.estimators import H2ODeepLearningEstimator
from dataprocessor import ProcessData

# initialize server
from featureeng.Math import moving_probability, probabilty_distribution

h2o.init()

# define response variable
response_column = 'RUL'

# load pre-processed data frames
training_frame = ProcessData.trainData(standard_deviation=True, moving_k_closest_average=True, moving_median_centered_average=True, moving_threshold_average=True)
testing_frame = ProcessData.testData(standard_deviation=True, moving_k_closest_average=True, moving_median_centered_average=True, moving_threshold_average=True)

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

# Building mode
model = H2ODeepLearningEstimator(hidden=[200, 200], score_each_iteration=False, variable_importances=True)
model.show()

# Training model
model.train(x=training_columns, y=response_column, training_frame=train)

# Performance testing
performance = model.model_performance(test_data=test)
print "\nPerformance data"
print "----------------------------------------------------------------------------------------------------------------"
performance.show()




