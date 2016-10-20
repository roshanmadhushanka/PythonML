from dataprocessor import TestData, TrainData
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import time

# Connect to the local H2O server
h2o.init(ip="127.0.0.1", port="54321")
# In case cluster was already running
h2o.remove_all()

start = time.clock()
print "Flow initialized"
print "----------------"
training_frame = TrainData.process()
testing_frame = TestData.process()

# Creating h2o frames
train = h2o.H2OFrame(training_frame)
test = h2o.H2OFrame(testing_frame)

# Applying haders
train.set_names(list(training_frame.columns))
test.set_names(list(testing_frame.columns))

training_columns = [u'Setting1', u'Setting2', u'Setting3', u'Sensor1', u'Sensor2', u'Sensor3', u'Sensor4', u'Sensor5',
                    u'Sensor6', u'Sensor7', u'Sensor8', u'Sensor9', u'Sensor10', u'Sensor11', u'Sensor12', u'Sensor13',
                    u'Sensor14', u'Sensor15', u'Sensor16', u'Sensor17', u'Sensor18', u'Sensor19', u'Sensor20',
                    u'Sensor21', u'Sensor1_ma_5', u'Sensor2_ma_5', u'Sensor3_ma_5', u'Sensor4_ma_5', u'Sensor5_ma_5',
                    u'Sensor6_ma_5', u'Sensor7_ma_5', u'Sensor8_ma_5', u'Sensor9_ma_5', u'Sensor10_ma_5',
                    u'Sensor11_ma_5', u'Sensor12_ma_5', u'Sensor13_ma_5', u'Sensor14_ma_5', u'Sensor15_ma_5',
                    u'Sensor16_ma_5', u'Sensor17_ma_5', u'Sensor18_ma_5', u'Sensor19_ma_5', u'Sensor20_ma_5',
                    u'Sensor21_ma_5', u'Sensor1_mm_5', u'Sensor2_mm_5', u'Sensor3_mm_5', u'Sensor4_mm_5',
                    u'Sensor5_mm_5', u'Sensor6_mm_5', u'Sensor7_mm_5', u'Sensor8_mm_5', u'Sensor9_mm_5',
                    u'Sensor10_mm_5', u'Sensor11_mm_5', u'Sensor12_mm_5', u'Sensor13_mm_5', u'Sensor14_mm_5',
                    u'Sensor15_mm_5', u'Sensor16_mm_5', u'Sensor17_mm_5', u'Sensor18_mm_5', u'Sensor19_mm_5',
                    u'Sensor20_mm_5', u'Sensor21_mm_5', u'Sensor1_sd_10', u'Sensor2_sd_10', u'Sensor3_sd_10',
                    u'Sensor4_sd_10', u'Sensor5_sd_10', u'Sensor6_sd_10', u'Sensor7_sd_10', u'Sensor8_sd_10',
                    u'Sensor9_sd_10', u'Sensor10_sd_10', u'Sensor11_sd_10', u'Sensor12_sd_10', u'Sensor13_sd_10',
                    u'Sensor14_sd_10', u'Sensor15_sd_10', u'Sensor16_sd_10', u'Sensor17_sd_10', u'Sensor18_sd_10',
                    u'Sensor19_sd_10', u'Sensor20_sd_10', u'Sensor21_sd_10', u'Sensor1_entropy_250',
                    u'Sensor2_entropy_250', u'Sensor3_entropy_250', u'Sensor4_entropy_250', u'Sensor5_entropy_250',
                    u'Sensor6_entropy_250', u'Sensor7_entropy_250', u'Sensor8_entropy_250', u'Sensor9_entropy_250',
                    u'Sensor10_entropy_250', u'Sensor11_entropy_250', u'Sensor12_entropy_250', u'Sensor13_entropy_250',
                    u'Sensor14_entropy_250', u'Sensor15_entropy_250', u'Sensor16_entropy_250', u'Sensor17_entropy_250',
                    u'Sensor18_entropy_250', u'Sensor19_entropy_250', u'Sensor20_entropy_250', u'Sensor21_entropy_250']

response_column = u'RUL'

# Building model
model = H2ODeepLearningEstimator(hidden=[1000, 1000, 1000], score_each_iteration=True, variable_importances=True)
model.show()

# Training model
model.train(x=training_columns, y=response_column, training_frame=train)

# Performance testing
performance = model.model_performance(test_data=test)
print "\nPerformance data"
print "----------------------------------------------------------------------------------------------------------------"
performance.show()

print "\nTime taken : ", time.clock() - start

