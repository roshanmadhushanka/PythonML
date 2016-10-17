# Deep Learning
import h2o
import pandas as pd
import numpy as np
from h2o.estimators import H2OAutoEncoderEstimator
from h2o.estimators import H2ODeepLearningEstimator
from tqdm import tqdm, tnrange
from tqdm import trange

from dataprocessor import ProcessData
from featureeng import Progress

# Define response column
response_column = 'RUL'

# initialize server
h2o.init()

# Load data frames
pData = ProcessData.trainData(moving_k_closest_average=True, standard_deviation=True, probability_distribution=True)
pTest = ProcessData.testData(moving_k_closest_average=True, standard_deviation=True, probability_from_file=True)

# Training model
print "\nTraining Model"
print "----------------------------------------------------------------------------------------------------------------"
training_columns = list(pData.columns)
training_columns.remove(response_column)
training_columns.remove('UnitNumber')
training_columns.remove('Time')

# Create h2o frame using filtered pandas frame
hTrain = h2o.H2OFrame(pData)
hTrain.set_names(list(pData.columns))

hTest = h2o.H2OFrame(pTest)
hTest.set_names(list(pTest.columns))

model = H2ODeepLearningEstimator(hidden=[64, 64, 64], score_each_iteration=True, variable_importances=True, epochs=100, activation='Tanh')
model.train(x=training_columns, y=response_column, training_frame=hTrain)

print "\nModel Performance"
print "----------------------------------------------------------------------------------------------------------------"
# Evaluate model
print model.model_performance(test_data=hTest)

