import pandas as pd
import h2o
from h2o.estimators import H2ODeepLearningEstimator

from dataprocessor import ProcessData, Process

# Initialize server
h2o.init()

#y
train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')

sustain = ['UnitNumber', 'Sensor14', 'Sensor9', 'Sensor4', 'Sensor13', 'Sensor11', 'Sensor7', 'Sensor8', 'RUL']
all_columns = list(train.columns)

for column in all_columns:
    if column not in sustain:
        del train[column]
        del test[column]


training_columns = sustain
training_columns.remove('UnitNumber')
training_columns.remove('RUL')

filter_train = Process.filterData(panda_frame=train, columns=sustain, removal_method='iqr', threshold=4)

feature_engineered_train = ProcessData.trainDataToFrame(training_frame=filter_train, moving_average=True, standard_deviation=True)
feature_engineered_test = ProcessData.trainDataToFrame(training_frame=test, moving_average=True, standard_deviation=True)

h_train = h2o.H2OFrame(feature_engineered_train)

h_train.set_names(list(feature_engineered_train.columns))

h_test = h2o.H2OFrame(feature_engineered_test)
h_test.set_names(list(feature_engineered_test.columns))

model = H2ODeepLearningEstimator(epochs=100, hidden=[512])
model.train(x=training_columns, y='RUL', training_frame=h_train, nfold=10)

print model.model_performance(test_data=h_test)
