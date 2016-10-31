import h2o
import numpy as np
from h2o.estimators import H2OAutoEncoderEstimator
from h2o.estimators import H2ODeepLearningEstimator
from dataprocessor import ProcessData, Filter
from featureeng import Measures
from parser import DataFrameParser

# Initialize server
h2o.init()

# AutoEncoder anomaly removal process
p_train = ProcessData.trainData(standard_deviation=True, probability_distribution=True, bin_classification=True)
p_test = ProcessData.testData(standard_deviation=True, probability_from_file=True, bin_classification=True)

# Converting to h2o frane
h_test = h2o.H2OFrame(p_test)
h_test.set_names(list(p_test.columns))

h_train = h2o.H2OFrame(p_train)
h_train.set_names(list(p_train.columns))

# Define autoencoder
anomaly_model = H2OAutoEncoderEstimator(
        activation="Rectifier",
        hidden=[25, 12, 25],
        sparse=True,
        l1=1e-4,
        epochs=100
    )

# Select relevant features
anomaly_train_columns = list(p_train.columns)
print anomaly_train_columns
anomaly_train_columns.remove('RUL')
anomaly_train_columns.remove('BIN')
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

# Filter anomalies based on reconstruction error
p_filter = Filter.filterDataAutoEncoder(panda_frame=p_train, reconstruction_error=err_list, threshold=threshold)

# Drop features
del p_filter['Setting3']
del p_filter['Sensor1']
del p_filter['Sensor5']
del p_filter['Sensor10']
del p_filter['Sensor16']
del p_filter['Sensor18']
del p_filter['Sensor19']


h_filter = h2o.H2OFrame(p_filter)
h_filter.set_names(list(p_filter.columns))

h_test = h2o.H2OFrame(p_test)
h_test.set_names(list(p_test.columns))

training_columns = list(p_filter.columns)
training_columns.remove('UnitNumber')
training_columns.remove('Time')
training_columns.remove('RUL')
training_columns.remove('BIN')

h_filter['BIN'] = h_filter['BIN'].asfactor()
h_test['BIN'] = h_test['BIN'].asfactor()

model = H2ODeepLearningEstimator(variable_importances=True)
model.train(x=training_columns, y='BIN', training_frame=h_filter, nfolds=10)

predict = model.predict(test_data=h_test)
predict = DataFrameParser.h2oToList(predict['predict'])
actual = DataFrameParser.h2oToList(h_test['BIN'])

Measures.confusion_matrix(actual, predict)
print predict
print actual







