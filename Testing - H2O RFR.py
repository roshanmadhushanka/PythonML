# h2o testing
import h2o
from h2o.estimators import H2ORandomForestEstimator
from dataprocessor import ProcessData

# initialize server
from featureeng.Math import moving_entropy

h2o.init()
# define response column
response_column = u'RUL'

# load pre-processed data frames
training_frame = ProcessData.trainData()
testing_frame = ProcessData.testData()

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

# Building model
# model = H2ORandomForestEstimator()
model = H2ORandomForestEstimator(ntrees=50, max_depth=20, nbins=100, seed=12345)
model.show()

# Training model
model.train(x=training_columns, y=response_column, training_frame=train)

# Performance testing
performance = model.model_performance(test_data=test)

print "\nPerformance data"
print "----------------------------------------------------------------------------------------------------------------"
performance.show()




