import h2o
import pandas as pd
from dataprocessor import ProcessData, Filter

h2o.init()

p_train = ProcessData.trainData()
p_test = ProcessData.testData()

columns = list(p_train.columns)
columns.remove('UnitNumber')
columns.remove('Time')
columns.remove('RUL')
columns.remove('Setting1')
columns.remove('Setting2')
columns.remove('Setting3')
p_train = Filter.filterDataPercentile(panda_frame=p_train, columns=columns, lower_percentile=0.01, upper_percentile=0.99, column_err_threshold=1)
p_train.to_csv("Filtered.csv")
