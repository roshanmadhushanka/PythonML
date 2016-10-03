import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn_pandas import DataFrameMapper

from dataprocessor import ProcessData

# Define response variable
response_column = "RUL"

# load the data set
training_frame = ProcessData.trainData()
testing_frame = ProcessData.testData()

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

# [row : column]
column_count = len(data[0, :])

# train
x = data[:, 0:column_count-1]
# response
y = data[:, column_count-1]
# test
tX = test[:, 0:column_count-1]
# ground truth
tY = test[:, column_count-1]

model = Sequential()
model.add(LSTM(1, input_shape=(20631, 24)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x, y, nb_epoch=100, batch_size=1, verbose=2)

# Estimate model performance
trainScore = model.evaluate(x, y, verbose=0)
trainScore = math.sqrt(trainScore)
#trainScore = scaler.inverse_transform(numpy.array([[trainScore]]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = model.evaluate(tX, tY, verbose=0)
testScore = math.sqrt(testScore)
#testScore = scaler.inverse_transform(numpy.array([[testScore]]))
print('Test Score: %.2f RMSE' % (testScore))

# # generate predictions for training
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)
#
# # shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
#
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
#
# # plot baseline and predictions
# plt.plot(dataset)
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()
