import numpy as np

def threeSigma(series, threshold=3):
    '''
    Identify anomalies according to three sigma rule

    Three Sigma Rule
    ----------------
    std  = standard deviation of data
    mean = mean of data
    if abs(x - mean) > 3 * std then x is an outlier

    :param threshold: 3 is the default value. Change at your own risk
    :param series: input data array
    :return: Index array of where anomalies are
    '''
    series = np.array(list(series))

    std = np.std(series)    # Standard deviation
    avg = np.average(series)# Mean

    anomaly_indexes = []
    for i in range(series.size):
        if (series[i] - avg) > threshold * std:
            anomaly_indexes.append(i)

    return anomaly_indexes

def iqr(series, threshold=3):
    '''
    Identify anomalies according to Inner-Quartile Range

    IQR Rule
    ----------------
    Q25 = 25 th percentile
    Q75 = 75 th percentile
    IQR = Q75 - Q25 Inner quartile range
    if abs(x-Q75) > 1.5 * IQR : A mild outlier
    if abs(x-Q75) > 3.0 * IQR : An extreme outlier

    :param series: input data array
    :param threshold: 1.5 mild, 3 extreme
    :return: Index array of where anomalies are
    '''

    series = np.array(list(series))
    q25 = np.percentile(series, 25)
    q75 = np.percentile(series, 75)
    iqr = q75 - q25

    anomaly_indexes = []
    for i in range(series.size):
        if (series[i] - q75) > threshold * iqr:
            anomaly_indexes.append(i)

    return anomaly_indexes






