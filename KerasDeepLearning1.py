from keras.layers import Dense, LSTM, Dropout, Activation
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

from dataprocessor import ProcessData
from sklearn_pandas import DataFrameMapper
import numpy
import math

# define response variable
response_column = 'RUL'

# load pre-processed data frames
training_frame = ProcessData.trainData()
testing_frame = ProcessData.testData()

del training_frame['UnitNumber']
del training_frame['Time']

del testing_frame['UnitNumber']
del testing_frame['Time']

# feature column names
training_columns = list(training_frame.columns)

# Set mapper
df_mapper = DataFrameMapper([(training_columns, None), (response_column, None)])

# Normalize the data set
scaler = MinMaxScaler(feature_range=(0, 1))

train = df_mapper.fit_transform(training_frame)
trainX = train[:, 0:24]
trainX = scaler.fit_transform(trainX)

trainY = train[:, 24]

test = df_mapper.fit_transform(testing_frame)
testX = train[:, 0:24]
testX = scaler.fit_transform(testX)

testY = train[:, 24]

trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

model = Sequential()
layers = [1, 50, 100, 1]

model.add(LSTM(input_dim=layers[0],output_dim=layers[1],return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(layers[2],return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(output_dim=layers[3]))
model.add(Activation("linear"))


model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=1, batch_size=1, verbose=2)


# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
trainScore = math.sqrt(trainScore)
print trainScore

predictions = model.predict(testX)
print predictions


