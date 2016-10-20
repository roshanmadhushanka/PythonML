from dataprocessor import ProcessData
from featureeng import Measures

train = ProcessData.trainData()

training_columns = list(train.columns)
training_columns.remove('RUL')

for column in training_columns:
    print column, Measures.correlation(train[column].values, train['RUL'].values)

