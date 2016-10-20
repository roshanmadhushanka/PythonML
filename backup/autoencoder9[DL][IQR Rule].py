# Deep Learning
import h2o
import numpy as np
import pandas as pd
from h2o.estimators import H2OAutoEncoderEstimator
from h2o.estimators import H2ODeepLearningEstimator

from dataprocessor import ProcessData
from featureeng import Progress


def getReconstructionError(recon_error, percentile):
    error_str = recon_error.get_frame_data()
    err_list = map(float, error_str.split("\n")[1:-1])
    var = np.array(err_list)  # input array
    return np.percentile(var, percentile * 100)


_validation_ratio_1 = 0.2
_validation_ratio_2 = 0.2
# Define response column
response_column = 'RUL'

# initialize server
h2o.init()

# Load data frames
pData = ProcessData.trainData()
#pTest = ProcessData.testData()

# Split data frame
pValidate = pData.sample(frac=_validation_ratio_1, random_state=200)
pTrain = pData.drop(pValidate.index)

# Convert pandas to h2o frame - for anomaly detection
hValidate = h2o.H2OFrame(pValidate)
hValidate.set_names(list(pValidate.columns))

hTrain = h2o.H2OFrame(pTrain)
hTrain.set_names(list(pTrain.columns))

# Save validate and train frames
pValidate.to_csv("Auto-Validate.csv", index=False)
pTrain.to_csv("Auto-Train.csv", index=False)

# Select relevant features
anomaly_train_columns = list(hTrain.columns)
anomaly_train_columns.remove(response_column)
anomaly_train_columns.remove('UnitNumber')
anomaly_train_columns.remove('Time')

column_count = len(anomaly_train_columns)

layers = [20, 6, 20]
print "Layers:", layers
# Define model
anomaly_model = H2OAutoEncoderEstimator(
        activation="Rectifier",
        hidden=layers,
        l1=1e-4,
        epochs=100,
    )

# Train model
anomaly_model.train(x=anomaly_train_columns, training_frame=hTrain, validation_frame=hValidate)

# Get reconstruction error
reconstruction_error = anomaly_model.anomaly(test_data=hTrain, per_feature=False)

print "Max Reconstruction Error       :", reconstruction_error.max()

# Reconstruction error detail
error_str = reconstruction_error.get_frame_data()
err_list = map(float, error_str.split("\n")[1:-1])
err_list = np.array(err_list)

'''
IQR Rule
----------------
Q25 = 25 th percentile
Q75 = 75 th percentile
IQR = Q75 - Q25 Inner quartile range
if abs(x-Q75) > 1.5 * IQR : A mild outlier
if abs(x-Q75) > 3.0 * IQR : An extreme outlier

'''

q25 = np.percentile(err_list, 25)
q75 = np.percentile(err_list, 75)
iqr = q75 - q25

# Filter rows
print "\nRemoving Anomalies"
print "----------------------------------------------------------------------------------------------------------------"
print "Reconstruction Error Array Size :", len(reconstruction_error)
filtered_train = pd.DataFrame()
count = 0
for i in range(hTrain.nrow):
    if abs(err_list[i] - q75) < 3 * iqr:
        df1 = pTrain.iloc[i, :]
        filtered_train = filtered_train.append(df1, ignore_index=True)
        count += 1
    Progress.printProgress(iteration=(i+1), total=hTrain.nrow, decimals=1, prefix="Progress", suffix="Complete")

print filtered_train
print "Original Size :", hTrain.nrow
print "Filtered Size :", len(filtered_train)
print "Removed Rows  :", (hTrain.nrow-len(filtered_train))

# Feature Engineering
pData = ProcessData.trainDataToFrame(filtered_train, moving_average=True, standard_deviation=True)
pTest = ProcessData.testData(moving_average=True, standard_deviation=True)

# Convert pandas to h2o frame - for model training
hData = h2o.H2OFrame(pData)
hData.set_names(list(pData.columns))

hTrain, hValidate = hData.split_frame(ratios=[_validation_ratio_2])

hTest = h2o.H2OFrame(pTest)
hTest.set_names(list(pTest.columns))

# Training model
print "\nTraining Model"
print "----------------------------------------------------------------------------------------------------------------"
training_columns = list(pData.columns)
training_columns.remove(response_column)
training_columns.remove('UnitNumber')
training_columns.remove('Time')

# Create h2o frame using filtered pandas frame
filtered = h2o.H2OFrame(filtered_train)
filtered.set_names(list(filtered_train.columns))

model = H2ODeepLearningEstimator(hidden=[64, 64, 64], score_each_iteration=True, variable_importances=True, epochs=100, activation='Tanh')
model.train(x=training_columns, y=response_column, training_frame=filtered, validation_frame=hValidate)

print "\nModel Performance"
print "----------------------------------------------------------------------------------------------------------------"
# Evaluate model
print model.model_performance(test_data=hTest)

