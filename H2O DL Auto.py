# h2o testing
import h2o
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators import H2ORandomForestEstimator
from dataprocessor import ProcessData

# initialize server
from featureeng.Math import moving_probability, probabilty_distribution

h2o.init()

# define response variable
response_column = 'RUL'

# load pre-processed data frames
training_frame = ProcessData.trainData(moving_k_closest_average=True, standard_deviation=True)
testing_frame = ProcessData.testData(moving_k_closest_average=True, standard_deviation=True)

# create h2o frames
train = h2o.H2OFrame(training_frame)
test = h2o.H2OFrame(testing_frame)
train.set_names(list(training_frame.columns))
test.set_names(list(testing_frame.columns))

# Feature selection
training_columns = ['UnitNumber', 'Time', 'Setting1', 'Setting2', 'Setting3', 'Sensor1', 'Sensor2', 'Sensor3', 'Sensor4', 'Sensor5', 'Sensor6', 'Sensor7', 'Sensor8', 'Sensor9', 'Sensor10', 'Sensor11', 'Sensor12', 'Sensor13', 'Sensor14', 'Sensor15', 'Sensor16', 'Sensor17', 'Sensor18', 'Sensor19', 'Sensor20', 'Sensor21', 'RUL', 'sd_10_Sensor1', 'sd_10_Sensor2', 'sd_10_Sensor3', 'sd_10_Sensor4', 'sd_10_Sensor5', 'sd_10_Sensor6', 'sd_10_Sensor7', 'sd_10_Sensor8', 'sd_10_Sensor9', 'sd_10_Sensor10', 'sd_10_Sensor11', 'sd_10_Sensor12', 'sd_10_Sensor13', 'sd_10_Sensor14', 'sd_10_Sensor15', 'sd_10_Sensor16', 'sd_10_Sensor17', 'sd_10_Sensor18', 'sd_10_Sensor19', 'sd_10_Sensor20', 'sd_10_Sensor21', 'k_closest_Sensor1', 'k_closest_Sensor2', 'k_closest_Sensor3', 'k_closest_Sensor4', 'k_closest_Sensor5', 'k_closest_Sensor6', 'k_closest_Sensor7', 'k_closest_Sensor8', 'k_closest_Sensor9', 'k_closest_Sensor10', 'k_closest_Sensor11', 'k_closest_Sensor12', 'k_closest_Sensor13', 'k_closest_Sensor14', 'k_closest_Sensor15', 'k_closest_Sensor16', 'k_closest_Sensor17', 'k_closest_Sensor18', 'k_closest_Sensor19', 'k_closest_Sensor20', 'k_closest_Sensor21']
training_columns.remove(response_column)
training_columns.remove("UnitNumber")
training_columns.remove("Time")

# Building mode
model = H2ODeepLearningEstimator(hidden=[200, 200], score_each_iteration=True, variable_importances=True)
model1 = H2ORandomForestEstimator(ntrees=50, max_depth=20, nbins=100, seed=12345)
#model.show()

# Training model
model.train(x=training_columns, y=response_column, training_frame=train)
model1.train(x=training_columns, y=response_column, training_frame=train)

# # Performance testing
# performance = model.model_performance(test_data=test)
#
# print "\nPerformance data"
# print "----------------------------------------------------------------------------------------------------------------"
# performance.show()

print "\nPrediction"
print "----------------------------------------------------------------------------------------------------------------"
print model.predict(test[0,:])
print model1.predict(test[0,:])




