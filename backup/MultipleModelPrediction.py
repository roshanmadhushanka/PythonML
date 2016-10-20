# h2o testing
import h2o
from h2o.estimators import H2ODeepLearningEstimator
from h2o.estimators import H2ORandomForestEstimator
from sklearn_pandas import DataFrameMapper

from dataprocessor import ProcessData
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# initialize server
from featureeng.Math import moving_probability, probabilty_distribution

h2o.init()

# define response variable
response_column = 'RUL'

# load pre-processed data frames
training_frame = ProcessData.trainData(standard_deviation=True, moving_k_closest_average=True)
testing_frame = ProcessData.testData(standard_deviation=True, moving_k_closest_average=True)

# create h2o frames
train = h2o.H2OFrame(training_frame)
test = h2o.H2OFrame(testing_frame)
train.set_names(list(training_frame.columns))
test.set_names(list(testing_frame.columns))

# Feature selection
training_columns = list(training_frame.columns)
training_columns.remove(response_column)
training_columns.remove("UnitNumber")
training_columns.remove("Time")

# Set mapper
df_mapper = DataFrameMapper([(training_columns, None), (response_column, None)])

# Test data - pandas to sklearn
test_tmp = df_mapper.fit_transform(testing_frame)

# [row : column]
column_count = len(test_tmp[0, :])

# ground truth
tY = np.array(testing_frame['RUL'])


# Building model
model1 = H2ODeepLearningEstimator(hidden=[200, 200], score_each_iteration=False, variable_importances=True)
model2 = H2ODeepLearningEstimator(hidden=[200, 200], score_each_iteration=False, variable_importances=True)
model3 = H2ODeepLearningEstimator(hidden=[200, 200], score_each_iteration=False, variable_importances=True)
model4 = H2ODeepLearningEstimator(hidden=[200, 200], score_each_iteration=False, variable_importances=True)
model5 = H2ODeepLearningEstimator(hidden=[200, 200], score_each_iteration=False, variable_importances=True)

# train model
model1.train(x=training_columns, y=response_column, training_frame=train)
model2.train(x=training_columns, y=response_column, training_frame=train)
model3.train(x=training_columns, y=response_column, training_frame=train)
model4.train(x=training_columns, y=response_column, training_frame=train)
model5.train(x=training_columns, y=response_column, training_frame=train)

predicted1 = model1.predict(test)
predicted2 = model2.predict(test)
predicted3 = model3.predict(test)
predicted4 = model4.predict(test)
predicted5 = model5.predict(test)


predicted_vals = np.zeros(shape=100)
for i in range(len(test[:,0])):
    tmp = []
    tmp.append(predicted1[i, 0])
    tmp.append(predicted2[i, 0])
    tmp.append(predicted3[i, 0])
    tmp.append(predicted4[i, 0])
    tmp.append(predicted5[i, 0])
    tmp.sort()
    print tmp
    predicted_vals[i] = (sum(tmp[1:-1]) / 3.0)


print mean_squared_error(tY, predicted_vals)
#
# [179.1842303466797, 150.96599319458008, 70.80777778625489, 99.05035125732422, 119.63671813964844, 128.86824996948243, 118.02661895751953, 106.66666656494141, 131.7653952026367, 127.47224365234375, 76.26, 91.70666664123536, 96.35037231445312, 119.52677352905273, 184.71269622802734, 140.2331527709961, 53.2566667175293, 55.080000000000005, 137.52383422851562, 22.74, 69.98, 144.20655700683594, 180.331904296875, 24.7, 157.57544845581054, 113.10122207641602, 150.76982788085937, 101.22108978271484, 101.65000000000002, 104.72840209960937, 20.758571434020997, 52.5, 116.82628967285156, 7.533166670799257, 8.984285726547242, 26.931538467407226, 63.42333335876466, 48.07500000000001, 148.25097427368163, 28.385, 74.66, 20.531111125946044, 66.477142868042, 136.6753369140625, 80.64831169128418, 56.391428604125984, 132.07758880615233, 143.0389826965332, 17.207142848968505, 132.45973678588868, 154.41443450927736, 30.28, 33.78, 163.3543894958496, 152.45653671264648, 13.21333333015442, 123.04247634887696, 51.88, 164.14496795654296, 138.5383087158203, 44.9220000076294, 53.345714263916015, 61.419666595458985, 45.30160007476807, 168.96177230834962, 15.5, 173.2016665649414, 19.67466667175293, 152.3450747680664, 132.98717849731446, 149.40943710327147, 67.61699996948242, 160.48600006103516, 105.58555084228516, 178.19096267700195, 14.337999992370605, 33.18, 199.46790802001954, 141.84486709594728, 75.355, 10.025, 9.191428579092026, 135.55609313964842, 57.40455513000487, 157.9752490234375, 81.32833328247071, 145.2040530395508, 161.41076705932616, 135.524375, 41.44, 28.44599998474121, 19.47599998474121, 50.78, 44.745, 156.4703305053711, 160.25928787231445, 81.02, 93.98233337402344, 149.64566665649414, 16.76]
#
#
