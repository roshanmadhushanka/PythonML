import pandas as pd
from sklearn.metrics import  mean_squared_error, mean_absolute_error

java_train = pd.read_csv("JavaTrain.csv")
python_train = pd.read_csv("PythonTrain.csv")

columns = list(java_train.columns)

for column in columns:
    # mse = mean_squared_error(python_train[column], java_train[column])
    # if mse > 0.000001:
    #     print column
    #     print python_train[column][0], java_train[column][0]

    mae = mean_absolute_error(python_train[column], java_train[column])
    if mae > 0.000001:
        print column
        print python_train[column][0], java_train[column][0]


