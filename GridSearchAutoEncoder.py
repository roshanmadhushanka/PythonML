import h2o
from h2o.estimators import H2ODeepLearningEstimator

from dataprocessor import ProcessData

from h2o.estimators import H2ORandomForestEstimator
from h2o.grid import H2OGridSearch

# Initialize server
h2o.init()

data = ProcessData.trainData(moving_k_closest_average=True, standard_deviation=True, probability_distribution=True)

hData = h2o.H2OFrame(data)
hData.set_names(list(data.columns))

training_columns = list(data.columns)
training_columns.remove('RUL')
training_columns.remove('UnitNumber')
training_columns.remove('Time')

# hyper_parameters = {'ntrees': [10, 50], 'max_depth': [20, 10]}
# grid_search = H2OGridSearch(H2ORandomForestEstimator, hyper_params=hyper_parameters)
# grid_search.train(x=training_columns, y='RUL', training_frame=hData)
# grid_search.show()
# models = grid_search.sort_by("mse")
# print models

hyper_parameters = {'activation': ['Tanh', 'TanhWithDropout', 'Rectifier', 'RectifierWithDropout', 'Maxout',
                                   'MaxoutWithDropout'], 'epochs': [10, 50, 100], 'hidden':[[20, 6, 20], [64, 64, 64]]}
grid_search = H2OGridSearch(H2ODeepLearningEstimator, hyper_params=hyper_parameters)
grid_search.train(x=training_columns, y='RUL', training_frame=hData)
grid_search.show()
models = grid_search.sort_by("mse")
print models
