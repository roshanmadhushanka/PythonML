import h2o
import pandas as pd
import numpy as np
from h2o.estimators import H2OAutoEncoderEstimator
from h2o.estimators import H2ODeepLearningEstimator
from tqdm import tqdm, tnrange
from tqdm import trange

from dataprocessor import ProcessData
from featureeng import Progress

def getReconstructionError(recon_error, percentile):
    error_str = recon_error.get_frame_data()
    err_list = map(float, error_str.split("\n")[1:-1])
    var = np.array(err_list)  # input array
    return np.percentile(var, percentile * 100)


_validation_ratio = 0.1
_reconstruction_error_rate = 0.9

# Define response column
response_column = 'RUL'

# initialize server
h2o.init()

# Load data frames
pData = ProcessData.trainData()
#pTest = ProcessData.testData()

# Split data frame
pValidate = pData.sample(frac=_validation_ratio, random_state=200)
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

inner_layer_size = column_count / 4
print "Inner Layer Size :", inner_layer_size
# Define model
anomaly_model = H2OAutoEncoderEstimator(
        activation="Rectifier",
        hidden=[inner_layer_size, inner_layer_size, inner_layer_size],
        sparse=True,
        l1=1e-4,
        epochs=100,
    )

# Train model
anomaly_model.train(x=anomaly_train_columns, training_frame=hTrain, validation_frame=hValidate)

# Get reconstruction error
reconstruction_error = anomaly_model.anomaly(test_data=hTrain, per_feature=False)

# Threshold
#threshold = reconstruction_error.max() * _reconstruction_error_rate
threshold = getReconstructionError(reconstruction_error, 0.99)

print "Max Reconstruction Error       :", reconstruction_error.max()
print "Threshold Reconstruction Error :", threshold

# Filter rows
print "\nRemoving Anomalies"
print "----------------------------------------------------------------------------------------------------------------"
print "Reconstruction Error Array Size :", len(reconstruction_error)
filtered_train = pd.DataFrame()
count = 0
for i in range(hTrain.nrow):
    if reconstruction_error[i,0] < threshold:
        df1 = pTrain.iloc[i, :]
        filtered_train = filtered_train.append(df1, ignore_index=True)
        count += 1
    Progress.printProgress(iteration=(i+1), total=hTrain.nrow, decimals=1, prefix="Progress", suffix="Complete")

print filtered_train
print "Original Size :", hTrain.nrow
print "Filtered Size :", len(filtered_train)
print "Removed Rows  :", (hTrain.nrow-len(filtered_train))

# Feature Engineering
pTrain = ProcessData.trainDataToFrame(filtered_train, moving_k_closest_average=True, standard_deviation=True, probability_distribution=True)
pTest = ProcessData.testData(moving_k_closest_average=True, standard_deviation=True, probability_from_file=True)

# Convert pandas to h2o frame - for model training
hValidate = h2o.H2OFrame(pValidate)
hValidate.set_names(list(pValidate.columns))

hTrain = h2o.H2OFrame(pTrain)
hTrain.set_names(list(pTrain.columns))

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

model = H2ODeepLearningEstimator(hidden=[500, 500], score_each_iteration=True, variable_importances=True, epochs=100)
model.train(x=training_columns, y=response_column, training_frame=filtered)

print "\nModel Performance"
print "----------------------------------------------------------------------------------------------------------------"
# Evaluate model
print model.model_performance(test_data=hTest)

