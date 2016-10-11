# try with using pandas dat frame

import h2o
import numpy as np
import pandas as pd
from h2o.estimators import H2OAutoEncoderEstimator
from h2o.estimators import H2ODeepLearningEstimator

from dataprocessor import ProcessData
from featureeng import Progress

_validation_ratio = 0.9
_reconstruction_error_rate = 0.9

# Define response column
response_column = 'RUL'

# initialize server
h2o.init()

# Load data frames
pData = ProcessData.trainData()
pTest = ProcessData.testData()

# Build h2o frames
hData = h2o.H2OFrame(pData)
hData.set_names(list(pData.columns))

hTest = h2o.H2OFrame(pTest)
hTest.set_names(list(pData.columns))

# Split data into training and validating frames
hTrain, hValidate = hData.split_frame(ratios=[_validation_ratio])

# Define model
anomaly_model = H2OAutoEncoderEstimator(
        activation="Rectifier",
        hidden=[25, 12, 25],
        sparse=True,
        l1=1e-4,
        epochs=100,
    )

# Select relevant features
anomaly_train_columns = list(hTrain.columns)
anomaly_train_columns.remove(response_column)
anomaly_train_columns.remove('UnitNumber')
anomaly_train_columns.remove('Time')

# Train model
anomaly_model.train(x=anomaly_train_columns, training_frame=hTrain, validation_frame=hValidate)

# Get reconstruction error
reconstruction_error = anomaly_model.anomaly(test_data=hTrain, per_feature=False)

# Threshold
threshold = reconstruction_error.max() * _reconstruction_error_rate

print "Max Reconstruction Error       :", reconstruction_error.max()
print "Threshold Reconstruction Error :", threshold

# Filter rows
print "\nRemoving Anomalies"
print "----------------------------------------------------------------------------------------------------------------"

filtered_train = pd.DataFrame()
count = 0
for i in range(hTrain.nrow):
    if reconstruction_error[i,0] < threshold:
        df1 = pd.DataFrame( hTrain[i, 0:].as_data_frame(use_pandas=True))
        filtered_train = filtered_train.append(df1, ignore_index=True)
        count += 1
    Progress.printProgress(iteration=(i+1), total=hTrain.nrow, decimals=1, prefix="Progress", suffix="Complete")


print "\nTraining Model"
print "----------------------------------------------------------------------------------------------------------------"
training_columns = list(pData.columns)
training_columns.remove(response_column)
training_columns.remove('UnitNumber')
training_columns.remove('Time')

filtered = h2o.H2OFrame(filtered_train)
filtered.set_names(list(filtered_train.columns))

model = H2ODeepLearningEstimator(hidden=[500, 500], score_each_iteration=True, variable_importances=True, epochs=100)
model.train(x=training_columns, y=response_column, training_frame=filtered)

print model.model_performance(test_data=hTest)




# df = pd.DataFrame(hTest[0,:].as_data_frame(use_pandas=True), index=None)
# print df
# for i in range(1, hTest.nrow):
#     df1 = pd.DataFrame(hTest[i,:].as_data_frame(use_pandas=True), index=None)
#     df =df.append(df1, ignore_index=True)
#
# print df

# df = pd.DataFrame()
# print df
# for i in range(0, hTest.nrow):
#     df1 = pd.DataFrame(hTest[i,:].as_data_frame(use_pandas=True))
#     df =df.append(df1, ignore_index=True)
#
# print df
#
#


