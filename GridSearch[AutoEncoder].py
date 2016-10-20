# Grid Search AutoEncoders
from h2o.estimators import H2OAutoEncoderEstimator
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

# Create h2o frame
hData = h2o.H2OFrame(pData)
hData.set_names(list(pData.columns))

# Split data inti training and validation
hTrain, hValidate = hData.split_frame(ratios=[0.8])

h2o.export_file(hTrain, "hTrainMy.csv", force=True)
h2o.export_file(hValidate, "hValidateMy.csv", force=True)

response_column = 'RUL'

hyper_parameters = {'activation': ['Tanh', 'TanhWithDropout', 'Rectifier', 'RectifierWithDropout',
          'Maxout', 'MaxoutWithDropout'], 'hidden': [4 , 6, 8, 10, 12, 14, 16, 18, 20], 'epochs': [50, 100, 150], 'loss': ['Quadratic', 'Absolute', 'Huber'],
                    'distribution': ['AUTO', 'bernoulli', 'multinomial', 'poisson', 'gamma',
           'tweedie', 'laplace', 'huber', 'quantile', 'gaussian']}

grid_search = H2OGridSearch(H2OAutoEncoderEstimator, hyper_params=hyper_parameters)
grid_search.train(x=selected_columns, training_frame=hTrain, validation_frame=hValidate)
grid_search.show()
models = grid_search.sort_by("mse")
print models






