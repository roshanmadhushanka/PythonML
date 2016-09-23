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