from heapq import nsmallest
from scipy.special import entr
import numpy as np

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

def moving_threshold_average(series, threshold=-1, window=5, default=False):
    '''
    Calculate moving threshold average

    :param series: Number series to compute
    :param threshold: Threshold error, -1 for automatic calculate
    :param window: Selected time window
    :param default: True -> Replace initial values inside the time window to zero
                    False -> Neglect and continue
    :return: calculated result in numpy array
    '''

    # Convert pandas.series to list
    series = list(series)

    if threshold == -1:
        # calculate threshol value
        _min = min(series)
        _max = max(series)
        _avg = np.mean(series)

        limit = 0

        if(abs(_min - _avg) > abs(_max - _avg)):
            limit = abs(_max - _avg)
        else:
            limit = abs(_min - _avg)

        threshold = limit / 2.0


    size = len(series)
    ret = np.zeros(shape=size - window + 1)
    for i in range(size - window + 1):
        subset = series[i:i + window]
        average = sum(subset) / float(len(subset))
        if abs(subset[-1] - average) < threshold:
            ret[i] = average
        else:
            ret[i] = subset[-1]

    # Add default values for initial window
    if (default):
        return np.concatenate((np.zeros(shape=window - 1), ret), axis=0)
    else:
        return ret

def moving_median_centered_average(series, window=5, boundary=1, default=False):
    '''
    Median centered average

    :param series: Number series to compute
    :param window: Selected time window
    :param boundry: Boundary neglect from both ends
    :param default: True -> Replace initial values inside the time window to zero
                    False -> Neglect and continue
    :return: calculated result in numpy array
    '''

    # Convert pandas.series to list
    series = list(series)

    size = len(series)
    ret = np.zeros(shape=size - window + 1)
    for i in range(size - window + 1):
        subset = series[i:i + window]
        subset.sort()
        subset = subset[boundary:-boundary]
        ret[i] = sum(subset) / float(len(subset))

    # Add default values for initial window
    if (default):
        return np.concatenate((np.zeros(shape=window - 1), ret), axis=0)
    else:
        return ret

def moving_k_closest_average(series, window=5, kclosest=3, default=False):
    '''

    Calculate moving k closest average

    :param series: Number series to compute
    :param window: Selected time window
    :param kclosest: Number of closet value to the original value. always less than window
    :param default: True -> Replace initial values inside the time window to zero
                    False -> Neglect and continue
    :return: calculated result in numpy array
    '''

    # Convert pandas.series to list
    series = list(series)

    size = len(series)
    ret = np.zeros(shape=size - window + 1)
    for i in range(size - window + 1):
        subset = series[i:i + window]
        k_closest = nsmallest(kclosest, subset, key=lambda x: abs(x - subset[-1]))
        ret[i] = sum(k_closest) / float(len(k_closest))

    # Add default values for initial window
    if (default):
        return np.concatenate((np.zeros(shape=window - 1), ret), axis=0)
    else:
        return ret

def moving_weighted_average(series, window=5, weights=[1, 2, 3, 4, 5], default=False):
    if len(weights) <> window:
        return np.zeros(shape=len(series))

    # Convert pandas.series to list
    series = list(series)

    size = len(series)
    ret = np.zeros(shape=size - window + 1)
    for i in range(size - window + 1):
        subset = np.array(series[i:i + window])
        weights = np.array(weights)
        ret[i] = sum(subset*weights) / float(sum(weights))

    # Add default values for initial window
    if (default):
        return np.concatenate((np.zeros(shape=window - 1), ret), axis=0)
    else:
        return ret

def moving_median(series, window=5, default=False):
    '''
        Calculate median within a moving window

        :param series: Number series to compute
        :param window: Selected time window
        :param default: True -> Replace initial values inside the time window to zero
                        False -> Neglect and continue
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

    if bin_size == 0.0:
        # if value is not changing probability is one
        return np.ones(shape=len(series))

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

def moving_probability(series, window=10, no_of_bins=5, default=False):
    '''
            Calculate probability for a given window

            :param series: Input number series
            :param window: Selected time window
            :param default: True -> Replace initial values inside the time window to zero
                            Fale -> Neglect and continue
            :return: Calculated result in numpy array


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

        # calculate bin size
        min_value = x[0]
        max_value = x[-1]
        bin_size = 1.0 * (max_value - min_value) / no_of_bins

        # Calculate probabilities
        p = 1.0 * p / window

        # assigning to relevant bin
        bin = int((subset[-1] - min_value) / bin_size)
        if 0 <= bin < no_of_bins:
            ret[i] = p[bin]
        else:
            ret[i] = p[bin - 1]

    # Add default values for initial window
    if (default):
        return np.concatenate((np.zeros(shape=window - 1), ret), axis=0)
    else:
        return ret

def time_series(series, window=3):
    result = []
    for i in xrange(window):
        tmp = [0]*(i+1)
        result.append(tmp.extend(series[i+1:]))
    return np.array(result)
