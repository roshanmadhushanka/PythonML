import pandas as pd
import h2o
from h2o.estimators import H2ODeepLearningEstimator

from dataprocessor import ProcessData, Filter

# Initialize server
h2o.init()

#y
train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')

sustain = ['UnitNumber', 'Time','Sensor14', 'Sensor9', 'Sensor4', 'Sensor13', 'Sensor11', 'Sensor7', 'Sensor8', 'RUL']
all_columns = list(train.columns)

for column in all_columns:
    if column not in sustain:
        del train[column]
        del test[column]


training_columns = sustain
training_columns.remove('UnitNumber')
training_columns.remove('RUL')
training_columns.remove('Time')

#filter_train = Process.filterData(panda_frame=train, columns=sustain, removal_method='iqr', threshold=4)
filter_train = train

feature_engineered_train = ProcessData.trainDataToFrame(training_frame=filter_train, moving_k_closest_average=True, standard_deviation=True)
feature_engineered_test = ProcessData.trainDataToFrame(training_frame=test, moving_k_closest_average=True, standard_deviation=True, rul=True)

h_train = h2o.H2OFrame(feature_engineered_train)

h_train.set_names(list(feature_engineered_train.columns))

h_test = h2o.H2OFrame(feature_engineered_test)
h_test.set_names(list(feature_engineered_test.columns))

model = H2ODeepLearningEstimator(epochs=100, hidden=[200, 200], score_each_iteration=True)
model.train(x=training_columns, y='RUL', training_frame=h_train)

print model.model_performance(test_data=h_test)
