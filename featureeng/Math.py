import numpy as np
import sys
import time

from scipy.special import entr

def moving_average(series, window=5, default=False):
    '''
    Calculate average within a moving window

    :param series: Number series to compute
    :param window: Selected time window
    :param default: True -> Replace initial values inside the time window to zero
                    Fale -> Neglect and continue
    :return: calculated result in numpy array
    '''

    # Convert pandas.series to list
    series = list(series)

    # Calculate cumulative sum
    ret = np.cumsum(series, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    ret = ret[window - 1:] / window

    # Add default values for initial window
    if(default):
        return np.concatenate((np.zeros(shape=window-1), ret), axis=0)
    else:
        return ret

def moving_median(series, window=5, default=False):
    '''
        Calculate median within a moving window

        :param series: Number series to compute
        :param window: Selected time window
        :param default: True -> Replace initial values inside the time window to zero
                        Fale -> Neglect and continue
        :return: calculated result in numpy array
        '''
    # Convert pandas.series to list
    series = list(series)

    size = len(series)
    ret = np.zeros(shape=size - window + 1)
    for i in range(size - window + 1):
        subset = series[i:i+window]
        ret[i] = np.median(subset)

    # Add default values for initial window
    if (default):
        return np.concatenate((np.zeros(shape=window - 1), ret), axis=0)
    else:
        return ret

def moving_standard_deviation(series, window=5, default=False):
    '''
        Calculate standard deviation within a moving window

        :param series: Number series to compute
        :param window: Selected time window
        :param default: True -> Replace initial values inside the time window to zero
                        Fale -> Neglect and continue
        :return: calculated result in numpy array
        '''
    # Convert pandas.series to list
    series = list(series)

    size = len(series)
    ret = np.zeros(shape=size - window + 1)
    for i in range(size - window + 1):
        subset = series[i:i+window]
        ret[i] = np.std(subset, dtype=float)

    # Add default values for initial window
    if (default):
        return np.concatenate((np.zeros(shape=window - 1), ret), axis=0)
    else:
        return ret

def moving_entropy(series, window=10, no_of_bins=5, default=False):
    '''
        Calculate entropy sum within a moving window

        :param series: Input number series
        :param window: Selected time window
        :param default: True -> Replace initial values inside the time window to zero
                        Fale -> Neglect and continue
        :return: Calculated result in numpy array

        Reference : http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html
    '''
    # Convert pandas.series to list
    series = list(series)

    size = len(series)
    ret = np.zeros(shape=size - window + 1)
    for i in range(size - window + 1):
        # Select window
        subset = series[i:i + window]

        # Generating histogram
        p, x = np.histogram(subset, bins=no_of_bins)

        # Calculate probabilities
        p = 1.0 * p / window

        # Calculate entropy
        entropy_val = entr(p)

        ret[i] = entropy_val.sum(axis=0)

    # Add default values for initial window
    if (default):
        return np.concatenate((np.zeros(shape=window - 1), ret), axis=0)
    else:
        return ret

def entropy(series, no_of_bins=5):
    '''
        Calculate the entropy of data for whole data set

        :param series: Input number series
        :param no_of_bins: Number of discrete levels
        :return: calculated result in numpy array
    '''
    series = list(series)
    # Calculate bin size
    min_value = min(series)
    max_value = max(series)
    bin_size = 1.0 * (max_value - min_value) / no_of_bins

    '''
     Bin size becomes zero when the values in the series are not changing
     That means probability of occuring that value is 1 which means entropy is zero
    '''

    if bin_size == 0.0:
        return np.zeros(shape=len(series))

    # Generating histogram
    p, x = np.histogram(series, bins=no_of_bins)

    # Calculate probability
    p = 1.0 * p / sum(p)

    # Calculate entropy
    entropy_val = entr(p)

    ret = []
    for num in series:
        bin = int((num - min_value) / bin_size)
        if 0 <= bin < no_of_bins:
            ret.append(entropy_val[bin])
        else:
            ret.append(entropy_val[bin-1])
    return np.array(ret)

def probabilty_distribution(series, no_of_bins=5):
    '''
            Calculate the probability of data for whole data set

            :param series: Input number series
            :param no_of_bins: Number of discrete levels
            :return: calculated result in numpy array
        '''
    series = list(series)
    # Calculate bin size
    min_value = min(series)
    max_value = max(series)

    bin_size = 1.0 * (max_value - min_value) / no_of_bins

    '''
     Bin size becomes zero when the values in the series are not changing
     That means probability of occuring that value is 1 which means entropy is zero
    '''

    if bin_size == 0.0:
        return np.zeros(shape=len(series))

    # Generating histogram
    p, x = np.histogram(series, bins=no_of_bins)

    # Calculate probability
    p = 1.0 * p / sum(p)

    ret = []
    for num in series:
        bin = int((num - min_value) / bin_size)
        if 0 <= bin < no_of_bins:
            ret.append(p[bin])
        else:
            ret.append(p[bin - 1])
    return np.array(ret)
