# Without autoencoders chech anomalies
import h2o
import thread
import pandas as pd
from dataprocessor import Filter
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from parser import DataFrameParser
from presenting import Chart

# Initialize h2o server
h2o.init()

pTrain = pd.read_csv("hTrainMy.csv")
pValidate = pd.read_csv("hValidateMy.csv")
pTest = pd.read_csv("hTestingMy.csv")

all_columns = list(pTrain.columns)
removing_columns = ['UnitNumber', 'Time', 'RUL', 'Setting1', 'Setting2', 'Setting3']
selected_columns = [x for x in all_columns if x not in removing_columns]

filtered = Filter.filterData(panda_frame=pTrain, columns=selected_columns, removal_method="iqr", threshold=3)

hTrain = h2o.H2OFrame(filtered)
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

model = H2ODeepLearningEstimator(epochs=100, loss='Automatic', activation='Rectifier', distribution='poisson', hidden=[512])
model.train(x=training_columns, y=response_column, training_frame=hTrain)

print model.model_performance(test_data=hTest)

predict = DataFrameParser.h2oToNumpyArray(model.predict(test_data=hTest))
actual = DataFrameParser.h2oToNumpyArray(hTest['RUL'])


Chart.residual_histogram(actual, predict)
Chart.residual_vs_estimated(actual, predict)
Chart.acutal_and_predict(actual, predict)









