# try with using pandas dat frame

import h2o
import numpy as np
import pandas as pd
from h2o.estimators import H2OAutoEncoderEstimator
from h2o.estimators import H2ODeepLearningEstimator
from sklearn.cross_validation import train_test_split

from dataprocessor import ProcessData
from featureeng import Progress

_validation_ratio = 0.1
_reconstruction_error_rate = 0.9

# Define response column
response_column = 'RUL'

# initialize server
h2o.init()

# Load data frames
pData = ProcessData.trainData()
pTest = ProcessData.testData()

# Split methods
# method 1
train, validate = train_test_split(pData, test_size=_validation_ratio)
print len(train)
print train

# method 2
validate = pData.sample(frac=_validation_ratio, random_state=200)
train = pData.drop(validate.index)
print len(train)
print train