# import pandas as pd
# import numpy as np
from featureeng import Math
# from sklearn.metrics import  mean_squared_error, mean_absolute_error
#
# java_train = pd.read_csv("JavaTrain.csv")
# python_train = pd.read_csv("PythonTrain.csv")
#
# columns = list(java_train.columns)
#
# for column in columns:
#     mse = mean_squared_error(python_train[column], java_train[column])
#     if mse > 0.000001:
#         print column
#         print python_train[column][0], java_train[column][0]
#
#     # mae = mean_absolute_error(python_train[column], java_train[column])
#     # if mae > 0.000001:
#     #     print column, mae
#     #     print python_train[column][0], java_train[column][0]

lst = [1.0,2.0,1.0,3.0,2.0,4.0,2.0,1.0,5.0,5.0,7.0,6.0,6.0,5.0,1.0,4.0,7.0,3.0,5.0,2.0,3.0,5.0,7.0,5.0,5.0,3.0,6.0,3.0,4.0,5.0,4.0,4.0,2.0,3.0,1.0,7.0,8.0,8.0,2.0,4.0]
print lst.count(5.0)
prob_dist = Math.probabilty_distribution(lst, 1)
print list(prob_dist)
