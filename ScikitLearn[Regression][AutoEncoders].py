import h2o
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn_pandas import DataFrameMapper
from anomaly import Test
from featureeng import Measures

h2o.init()

# Configuration parameters
mapper_pkl = "mapper.pkl"
estimator_pkl = "estimator.pkl"
estimator_pmml = "model.pmml"
_vr_auto_encoder = 0.1  # Validation ratio for AutoEncoder

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

# Select columns for AutoEncoder
ac_train_columns = list(p_data.columns) # Define autoencoder train columns
rm_columns = ['RUL', 'UnitNumber', 'Time', 'Setting1', 'Setting2', 'Setting3'] # Columns need to be removed
'''
Because we are using auto encoders to remove noises in sensor readings. So we have to select only sensor readings
'''
for column in rm_columns:
    ac_train_columns.remove(column)

# # Define AutoEncoder model
# auto_encoder_model = H2OAutoEncoderEstimator(
#         activation="Tanh",
#         hidden=18,
#         epochs=150,
#         loss='Quadratic',
#         distribution='gaussian'
#     )
#
# # Train AutoEncoder model
# auto_encoder_model.train(x=ac_train_columns, training_frame=h_train, validation_frame=h_validate)
#
# # Get reconstruction error
# reconstruction_error = auto_encoder_model.anomaly(test_data=h_train, per_feature=False)
# error_str = reconstruction_error.get_frame_data()
# err_list = map(float, error_str.split("\n")[1:-1])
#
# # Filter anomalies in reconstruction error
# '''
# IQR Rule
# ----------------
# Q25 = 25 th percentile
# Q75 = 75 th percentile
# IQR = Q75 - Q25 Inner quartile range
# if abs(x-Q75) > 1.5 * IQR : A mild outlier
# if abs(x-Q75) > 3.0 * IQR : An extreme outlier
#
# '''
# q25 = np.percentile(err_list, 25)
# q75 = np.percentile(err_list, 75)
# iqr = q75 - q25
#
# rm_index = [] # Stores row numbers which have anomalies
# for i in range(h_train.nrow):
#     if abs(err_list[i] - q75) > 6 * iqr:
#         rm_index.append(i)

# Begin My Filter
rm_index = []
for column in ac_train_columns:
    series = p_data[column]
    anomaly = Test.iqr(series, threshold=4)
    rm_index.extend(anomaly)

# Sort indexes
rm_index.sort()
anomaly_series = list(set(rm_index))
p_train = p_data
print "Number of removed rows", len(rm_index)
# End My Filter

# Remove anomalies
p_filtered = p_train.drop(p_train.index[rm_index])

training_columns = list(p_filtered.columns)
training_columns.remove('RUL')
training_columns.remove('UnitNumber')
training_columns.remove('Time')

# Set mapper
df_mapper = DataFrameMapper([(training_columns, None), (response_column, None)])

# Train data - pandas to sklearn
train = df_mapper.fit_transform(p_filtered)

# Test data - pandas to sklearn
test = df_mapper.fit_transform(p_test)

# [row : column]
column_count = len(train[0, :])

# train
x = train[:, 0:column_count-1]
# response
y = train[:, column_count-1]

# test
tX = test[:, 0:column_count-1]
# ground truth
tY = test[:, column_count-1]

# Setting up algorithm
rf = RandomForestRegressor(max_depth=20, n_estimators=50)

# Train model
rf.fit(X=x, y=y)

# Get prediction results
result = rf.predict(tX)

print "Result"
print "------"
print result

# Analyze performance
print "Performance"
print "-----------"
print "Root Mean Squared Error", mean_squared_error(tY, np.array(result)) ** 0.5
print "Mean Absolute Error", mean_absolute_error(tY, np.array(result))
print "R2", Measures.r2(tY, np.array(result))
# Dump pickle files
print df_mapper.features
print rf.get_params()

joblib.dump(df_mapper, mapper_pkl, compress = 3)
joblib.dump(rf, estimator_pkl, compress = 3)

# # Build pmml
# command = "java -jar converter-executable-1.1-SNAPSHOT.jar --pkl-mapper-input mapper.pkl --pkl-estimator-input estimator.pkl --pmml-output mapper-estimator.pmml"
# os.system(command)


