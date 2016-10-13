# Deep Learning
import math
import h2o
import time
import numpy as np
import pandas as pd
from h2o.estimators import H2OAutoEncoderEstimator
from h2o.estimators import H2ODeepLearningEstimator
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from dataprocessor import ProcessData
from featureeng import Progress


def getReconstructionError(recon_error, percentile):
    error_str = recon_error.get_frame_data()
    err_list = map(float, error_str.split("\n")[1:-1])
    var = np.array(err_list)  # input array
    return np.percentile(var, percentile * 100)

# Configuration
_validation_ratio_1 = 0.1   # For Auto Encoder
_validation_ratio_2 = 0.1   # For Predictive Model
_reconstruction_error_rate = 0.9
_nmodels = 50
_smodels = 10
_lim = 3

# Define response column
response_column = 'RUL'

clock = time.clock();

# initialize server
h2o.init()

# Load data frames
pData = ProcessData.trainData()

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

print "Auto Encoder Model"
print "----------------------------------------------------------------------------------------------------------------"
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


print "Max Reconstruction Error       :", reconstruction_error.max()

# Filter rows
print "\nRemoving Anomalies"
print "----------------------------------------------------------------------------------------------------------------"
print "Reconstruction Error Array Size :", len(reconstruction_error)
filtered_train = pd.DataFrame()
count = 0
for i in range(hTrain.nrow):
    if abs(err_list[i] - q75) < 2 * iqr:
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
hData = h2o.H2OFrame(pTrain)
hData.set_names(list(pTrain.columns))

hTrain, hValidate = hData.split_frame(ratios=[_validation_ratio_2])

hTest = h2o.H2OFrame(pTest)
hTest.set_names(list(pTest.columns))

# Ground truth data
ground_truth = np.array(pTest['RUL'])

# Select training columns
training_columns = list(pData.columns)
training_columns.remove(response_column)
training_columns.remove('UnitNumber')
training_columns.remove('Time')


# Building models
model_arr = range(_nmodels)
print "Building models"
print "---------------"
for i in range(_nmodels):
    model_arr[i] = H2ODeepLearningEstimator(hidden=[500, 500], score_each_iteration=True, variable_importances=True)
print "Build model complete...\n"

# Training models
print "Train models"
print "------------"
for i in range(_nmodels):
    print "Train : " + str(i + 1) + "/" + str(_nmodels)
    model_arr[i].train(x=training_columns, y=response_column, training_frame=hTrain)
print "Train model complete...\n"

# Validate models using hValidate data [anomaly removed]
print "Validate models"
print "---------------"
mse_val = np.zeros(shape=_nmodels)
for i in range(_nmodels):
    mse_val[i] = model_arr[i].mae(model_arr[i].model_performance(test_data=hValidate))
print "Validation model complete...\n"

# Evaluating each model for prediction
print "Calculating weights"
print "-----------------"
weight_arr = np.amax(mse_val)/mse_val
print "Weights",weight_arr
print "Calculation weights complete...\n"

# Select best models
print "Select Models"
print "-------------"
selected_models = weight_arr.argsort()[-_smodels:][::-1]
model_arr = [model_arr[i] for i in selected_models]
weight_arr = [weight_arr[i] for i in selected_models]
_nmodels = _smodels
print "Select complete...\n"

# Predict from selected model
print "Predicting"
print "----------"
predicted_arr = range(_nmodels)
for i in range(_nmodels):
    predicted_arr[i] = model_arr[i].predict(hTest)
print "Prediction complete...\n"

# Filter predicted values
print "Filter Predictions"
print "------------------"
predicted_vals = np.zeros(shape=100)
for i in range(len(hTest[:,0])):
    tmp = []
    for j in range(_nmodels):
        tmp.append({'value':(predicted_arr[j][i, 0]*weight_arr[j]), 'weight':weight_arr[j]})

    print "Original :", tmp
    tmp = sorted(tmp, key=lambda k: k['value'])[_lim:-_lim]
    print "Selected :", tmp
    val = sum(d['value'] for d in tmp) / float(sum(d['weight'] for d in tmp))
    print "Val      :", val
    predicted_vals[i] = val
    print "------------------------------------------------------------------------------------------------------------"

print "Filter predictions complete...\n"

# Summary
print "Result"
print "------"
print "Root Mean Squared Error :", math.sqrt(mean_squared_error(ground_truth, predicted_vals))
print "Mean Absolute Error     :", mean_absolute_error(ground_truth, predicted_vals)



print "Time Taken :", time.clock() - clock;