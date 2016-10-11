import h2o
from h2o.estimators import H2ODeepLearningEstimator, H2OGradientBoostingEstimator
from h2o.estimators import H2ORandomForestEstimator
from sklearn_pandas import DataFrameMapper
from dataprocessor import ProcessData

# Initialize server
h2o.init()

# Define response variable
response_column = "RUL"

# Process data
training_frame = ProcessData.trainData(moving_k_closest_average=True, standard_deviation=True, probability_distribution=True)
testing_frame = ProcessData.testData(moving_k_closest_average=True, standard_deviation=True, probability_from_file=True)

# Select training columns
training_columns = list(training_frame.columns)
training_columns.remove(response_column) # Remove RUL
training_columns.remove("UnitNumber")    # Remove UnitNumber
training_columns.remove("Time")          # Remove Time

# Set mapper
df_mapper = DataFrameMapper([(training_columns, None), (response_column, None)])

# Train data - pandas to sklearn
data = df_mapper.fit_transform(training_frame)

# Test data - pandas to sklearn
test = df_mapper.fit_transform(testing_frame)

h2o_train = h2o.H2OFrame(data)
h2o_test = h2o.H2OFrame(test)

training_columns = h2o_train.names[:-1]
response_column = h2o_train.names[-1]

model = H2ODeepLearningEstimator(hidden=[500, 500], score_each_iteration=True, variable_importances=True, epochs=100)
#model = H2ORandomForestEstimator(ntrees=50, max_depth=20, nbins=100, seed=12345)
model.train(x=training_columns, y=response_column, training_frame=h2o_train)

print model.model_performance(test_data=h2o_test)
