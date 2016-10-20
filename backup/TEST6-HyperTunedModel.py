# Without autoencoders chech anomalies
from h2o.estimators import H2ODeepLearningEstimator

from dataprocessor import ProcessData
from anomaly import Test
import numpy as np
import h2o
import pandas as pd

# Initialize h2o server
h2o.init()

pTrain = pd.read_csv("hTrainMy.csv")
pValidate = pd.read_csv("hValidateMy.csv")
pTest = pd.read_csv("hTestingMy.csv")

hTrain = h2o.H2OFrame(pTrain)
hTrain.set_names(list(pTrain.columns))

hValidate = h2o.H2OFrame(pValidate)
hValidate.set_names(list(pValidate.columns))

hTest = h2o.H2OFrame(pTest)
hTest.set_names(list(pTest.columns))

training_columns = list(pTrain.columns)
training_columns.remove('UnitNumber')
training_columns.remove('Time')
training_columns.remove('RUL')

response_column = 'RUL'
print "OK"

model = H2ODeepLearningEstimator(hidden=[1024], activation='Maxout', epochs=100)
#model = H2ORandomForestEstimator(ntrees=50, max_depth=20, nbins=100, seed=12345)
model.train(x=training_columns, y=response_column, training_frame=hTrain, validation_frame=hValidate)

print model.model_performance(test_data=hTest)






