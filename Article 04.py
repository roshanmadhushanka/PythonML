import h2o
from h2o.estimators import H2OAutoEncoderEstimator
import numpy as np
import pandas as pd
from h2o.estimators import H2ODeepLearningEstimator

from dataprocessor import ProcessData, Filter

# Connect to h2o instance
from parser import DataFrameParser
from presenting import Chart

h2o.init()

# AutoEncoder anomaly removal process
p_train = ProcessData.trainData()

h_train = h2o.H2OFrame(p_train)
h_train.set_names(list(p_train.columns))

anomaly_model = H2OAutoEncoderEstimator(
        activation="Rectifier",
        hidden=[25, 12, 25],
        sparse=True,
        l1=1e-4,
        epochs=100
    )

# Select relevant features
anomaly_train_columns = list(p_train.columns)
anomaly_train_columns.remove('RUL')
anomaly_train_columns.remove('UnitNumber')
anomaly_train_columns.remove('Time')
anomaly_train_columns.remove('Setting1')
anomaly_train_columns.remove('Setting2')
anomaly_train_columns.remove('Setting3')

# Train model
anomaly_model.train(x=anomaly_train_columns, training_frame=h_train)

# Get reconstruction error
reconstruction_error = anomaly_model.anomaly(test_data=h_train, per_feature=False)
error_str = reconstruction_error.get_frame_data()
err_list = map(float, error_str.split("\n")[1:-1])
err_list = np.array(err_list)

# Threshold
threshold = np.amax(err_list) * 0.97

print "Max Reconstruction Error       :", reconstruction_error.max()
print "Threshold Reconstruction Error :", threshold

p_filter = Filter.filterDataAutoEncoder(panda_frame=p_train, reconstruction_error=err_list, threshold=threshold)
p_test = pd.read_csv('datasets/test.csv')
print p_filter

del p_filter['Setting3']
del p_filter['Sensor1']
del p_filter['Sensor5']
del p_filter['Sensor10']
del p_filter['Sensor16']
del p_filter['Sensor18']
del p_filter['Sensor19']

# Feature engineering process
columns = ['Sensor14', 'Sensor9', 'Sensor11', 'Sensor12', 'Sensor13', 'Sensor7', 'Sensor4', 'Sensor8', 'Sensor20', 'Sensor21', 'Sensor15', 'Sensor6', 'Sensor2', 'Sensor17', 'Sensor3']
p_featured_train = ProcessData.trainDataToFrame(training_frame=p_filter, selected_column_names=columns, probability_distribution=True)
p_featured_test = ProcessData.testDataToFrame(testing_frame=p_test, selected_column_names=columns, probability_from_file=True)

h_filter = h2o.H2OFrame(p_featured_train)
h_filter.set_names(list(p_featured_train.columns))


h_test = h2o.H2OFrame(p_featured_test)
h_test.set_names(list(p_featured_test.columns))

training_columns = list(p_featured_train.columns)
training_columns.remove('UnitNumber')
training_columns.remove('Time')
training_columns.remove('RUL')

model = H2ODeepLearningEstimator(variable_importances=True)
model.train(x=columns, y='RUL', training_frame=h_filter, nfolds=10)

print model.model_performance(test_data=h_test)

predict = DataFrameParser.h2oToNumpyArray(model.predict(test_data=h_test))
actual = DataFrameParser.h2oToNumpyArray(h_test['RUL'])
# var_imp = model.varimp()
# for detail in var_imp:
#     print detail[0]

Chart.residual_histogram(actual, predict)
Chart.residual_vs_estimated(actual, predict)
Chart.acutal_and_predict(actual, predict)




