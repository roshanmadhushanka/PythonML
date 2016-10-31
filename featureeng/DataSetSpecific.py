import numpy as np

def remaining_usefullifetime(indices=np.array([]), time_series=np.array([])):
    '''
    Calculate remaining useful life
    :param indices: Separations at time series
    :param time_series: Time series values
    :return: Remaining useful time
    '''

    # add final index
    indices = np.insert(indices, len(indices), len(time_series)-1, axis=0)

    rul = np.array([])

    # generate remaining useful array
    for i in indices:
        rul = np.concatenate((rul, range(time_series[i])[::-1]), axis=0)

    return rul

def binary_classification(indices=np.array([]), time_series=np.array([])):
    '''
    Add binary classification label

    :param indices: Seperation at time series
    :param time_series: Time series values
    :return: Binary classified classes
    '''
    # add final index
    indices = np.insert(indices, len(indices), len(time_series) - 1, axis=0)

    bin_classification = np.array([])

    for i in indices:
        class_label = range(time_series[i])[::-1]
        class_label = [0 if x >= 30 else 1 for x in class_label]
        bin_classification = np.concatenate((bin_classification, class_label), axis=0)

    return bin_classification