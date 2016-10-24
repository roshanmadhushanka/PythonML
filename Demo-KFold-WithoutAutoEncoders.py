import h2o
import pandas as pd
from h2o.estimators import H2ODeepLearningEstimator

# Initialize server
h2o.init()

# Load CSV data frames
p_train = pd.read_csv('Training.csv')
p_test = pd.read_csv('Testing.csv')

# Define columns
response_column = 'RUL'  # Define response column in the dataset

# Convert pandas frame to h2o
h_train = h2o.H2OFrame(p_train)
h_train.set_names(list(p_train.columns))
h_test = h2o.H2OFrame(p_test)
h_test.set_names(list(p_test.columns))

# Define columns
dl_train_columns = list(p_train.columns)
rm_columns = ['RUL', 'UnitNumber', 'Time']
for column in rm_columns:
    dl_train_columns.remove(column)


model = H2ODeepLearningEstimator(epochs=100, loss='Automatic', activation='RectifierWithDropout', distribution='poisson', hidden=[512], nfolds=10)
model.train(x=dl_train_columns, y=response_column, training_frame=h_train)
performance = model.model_performance(test_data=h_test)
print performance


