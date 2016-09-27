# MLP Regressor

import h2o
from sklearn.metrics import  mean_squared_error, mean_absolute_error
from sklearn_pandas import DataFrameMapper
from sklearn.externals import joblib
from dataprocessor import ProcessData
import numpy as np
import os

h2o.init()

# Define response variable
response_column = "RUL"

# Process data
training_frame = ProcessData.trainData(moving_average=True, standard_deviation=True, probability_distribution=True)
testing_frame = ProcessData.testData(moving_average=True, standard_deviation=True, probability_distribution=True)

# Select training columns
training_columns = list(training_frame.columns)
training_columns.remove(response_column) # Remove RUL
training_columns.remove("UnitNumber")    # Remove UnitNumber
training_columns.remove("Time")          # Remove Time




