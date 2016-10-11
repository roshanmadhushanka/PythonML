import pandas as pd
import numpy as np
from file import FileHandler

def probabilty_distribution(series, no_of_bins=5):
    '''
            Calculate the probability of data for whole data set

            :param series: Input number series
            :param no_of_bins: Number of discrete levels
            :return: calculated result in numpy array
        '''

    # Convert pandas.series to list
    series = list(series)

    # Calculate bin size
    min_value = min(series)
    max_value = max(series)

    bin_size = 1.0 * (max_value - min_value) / no_of_bins

    '''
     Bin size becomes zero when the values in the series are not changing
     That means probability of occuring that value is 1 which means entropy is zero
    '''


    # Generating histogram
    p, x = np.histogram(series, bins=no_of_bins)



    # Calculate probability
    p = 1.0 * p / sum(p)

    yield p
    yield x
    yield bin_size
    return

def saveToFile():
    # Data set preprocessor
    training_frame = pd.read_csv("train.csv")

    # Obtain all column names
    all_column_names = list(training_frame.columns)

    # Selected column names
    selected_column_names = all_column_names[5:-1]

    dict = {}
    for column_name in selected_column_names:
        tmp = {}
        prob, rang, bin_size = probabilty_distribution(training_frame[column_name], no_of_bins=250)
        tmp["prob"] = ",".join(map(str, prob))
        tmp["rang"] = ",".join(map(str, rang))
        tmp["sbin"] = bin_size
        dict[column_name] = tmp

    FileHandler.write_json("json.txt", dict)
