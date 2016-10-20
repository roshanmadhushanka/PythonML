# Without autoencoders check anomalies
from h2o.estimators import H2ODeepLearningEstimator
from h2o.grid import H2OGridSearch

from dataprocessor import ProcessData
from anomaly import Test
import numpy as np
import h2o

# Initialize h2o server
h2o.init()

# Load training data frame
pData = ProcessData.trainData()

# Select columns
selected_columns = list(pData.columns)
selected_columns.remove('UnitNumber')
selected_columns.remove('Time')
selected_columns.remove('RUL')
selected_columns.remove('Setting1')
selected_columns.remove('Setting2')
selected_columns.remove('Setting3')


tot = 0
anomaly_series = []
for column in selected_columns:
    series = pData[column]
    anomaly = Test.iqr(series, threshold=3)
    anomaly_series.extend(anomaly)

# Sort indexes
anomaly_series.sort()
anomaly_series = list(set(anomaly_series))
print anomaly_series
print len(anomaly_series)

# Remove anomalies
df = pData.drop(pData.index[anomaly_series])

# Feature engineering
data_frame = ProcessData.trainDataToFrame(df, moving_k_closest_average=True, standard_deviation=True)
testing_frame = ProcessData.testData(moving_k_closest_average=True, standard_deviation=True)

# Create h2o frame
hData = h2o.H2OFrame(data_frame)
hData.set_names(list(data_frame.columns))

hTesting = h2o.H2OFrame(testing_frame)
hTesting.set_names(list(testing_frame.columns))

# Split data inti training and validation
hTrain, hValidate = hData.split_frame(ratios=[0.8])

h2o.export_file(hTrain, "hTrainMy.csv", force=True)
h2o.export_file(hValidate, "hValidateMy.csv", force=True)
h2o.export_file(hTesting, "hTestingMy.csv", force=True)

training_columns = list(pData.columns)
training_columns.remove('UnitNumber')
training_columns.remove('Time')
training_columns.remove('RUL')

response_column = 'RUL'

hyper_parameters = {'distribution': ['auto', 'bernoulli', 'multinomial', 'gaussian', 'poisson', 'gamma', 'tweedie', 'laplace', 'quantile', 'huber'],
                    'fold_assignment':['auto', 'random', 'modulo', 'stratified'],
                    'histogram_type':['auto', 'uniform_adaptive', 'random', 'quantiles_global', 'round_robin']
                    }

grid_search = H2OGridSearch(H2ODeepLearningEstimator, hyper_params=hyper_parameters)
grid_search.train(x=training_columns, y='RUL', training_frame=hTrain, validation_frame=hValidate)
grid_search.show()
models = grid_search.sort_by("mse")
print models






