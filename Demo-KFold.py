import h2o
import math
import pandas as pd
import numpy as np
from h2o.estimators import H2OAutoEncoderEstimator
from h2o.estimators import H2ODeepLearningEstimator
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from presenting import Chart

# Initialize server
h2o.init()

# Configuration parameters
_vr_auto_encoder = 0.1  # Validation ratio for AutoEncoder
_vr_model = 0.1         # Validation ratio for DeepLearning model
_nmodels = 10 # Number of models going to train
_smodels = 5  # Number of models select to predict
_lim = 1      # Number of outliers removed from predictions, both maximums and minimums

# Load CSV data frames
p_data = pd.read_csv('Training.csv')
p_test = pd.read_csv('Testing.csv')

# Define columns
response_column = 'RUL'  # Define response column in the dataset

# Split p_data into validate and train
p_validate = p_data.sample(frac=_vr_auto_encoder, random_state=200) # Take sample data to validate AutoEncoder
p_train = p_data.drop(p_validate.index)

# Convert pandas frame to h2o
h_train = h2o.H2OFrame(p_train)
h_train.set_names(list(p_train.columns))
h_validate = h2o.H2OFrame(p_validate)
h_validate.set_names(list(p_validate.columns))
h_test = h2o.H2OFrame(p_test)
h_test.set_names(list(p_test.columns))

# Select columns for AutoEncoder
ac_train_columns = list(p_data.columns) # Define autoencoder train columns
rm_columns = ['RUL', 'UnitNumber', 'Time', 'Setting1', 'Setting2', 'Setting3'] # Columns need to be removed
'''
Because we are using auto encoders to remove noises in sensor readings. So we have to select only sensor readings
'''
for column in rm_columns:
    ac_train_columns.remove(column)

# Define AutoEncoder model
auto_encoder_model = H2OAutoEncoderEstimator(
        activation="Tanh",
        hidden=18,
        epochs=150,
        loss='Quadratic',
        distribution='gaussian'
    )

# Train AutoEncoder model
auto_encoder_model.train(x=ac_train_columns, training_frame=h_train, validation_frame=h_validate)

# Get reconstruction error
reconstruction_error = auto_encoder_model.anomaly(test_data=h_train, per_feature=False)
error_str = reconstruction_error.get_frame_data()
err_list = map(float, error_str.split("\n")[1:-1])

# Filter anomalies in reconstruction error
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

rm_index = [] # Stores row numbers which have anomalies
for i in range(h_train.nrow):
    if abs(err_list[i] - q75) > 4 * iqr:
        rm_index.append(i)

# Remove anomalies
p_filtered = p_train.drop(p_train.index[rm_index])

# Convert pandas to H2OFrame
h_data = h2o.H2OFrame(p_filtered)
h_data.set_names(list(p_data.columns))

# Define columns
dl_train_columns = list(p_filtered.columns)
rm_columns = ['RUL', 'UnitNumber', 'Time']
for column in rm_columns:
    dl_train_columns.remove(column)

model = H2ODeepLearningEstimator(epochs=100, loss='Automatic', activation='RectifierWithDropout', distribution='poisson', hidden=[512], nfolds=10)
model.train(x=dl_train_columns, y=response_column, training_frame=h_data)
performance = model.model_performance(test_data=h_test)
print performance

