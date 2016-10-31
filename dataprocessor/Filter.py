from anomaly import Test
import numpy as np
import matplotlib.pyplot as plt

def filterData(panda_frame, columns, removal_method, threshold):
    # Anomaly index container
    rm_index = []

    # Select anomaly removal type
    if removal_method == "iqr":
        for column in columns:
            series = panda_frame[column]
            anomaly = Test.iqr(series, threshold)
            rm_index.extend(anomaly)
    elif removal_method == "threesigma":
        for column in columns:
            series = panda_frame[column]
            anomaly = Test.iqr(series, threshold)
            rm_index.extend(anomaly)

    # Sort indexes
    rm_index.sort()
    anomaly_series = list(set(rm_index))

    # Remove anomalies
    p_filtered = panda_frame.drop(panda_frame.index[anomaly_series])
    return p_filtered

def filterDataPercentile(panda_frame, columns, lower_percentile, upper_percentile, column_err_threshold, order='under'):
    '''
    Filter anomalies based on

    :param panda_frame: Input data frame
    :param columns: Columns that need to apply filter
    :param lower_percentile: Below this level consider as an anomaly
    :param upper_percentile: Beyond this level consider as an anomaly
    :param column_err_threshold: Per column threshold. If a particular row detects as an anomaly how many columns that
                                needs to show as an anomaly
    :return:
    '''
    # Anomaly index container
    rm_index = []

    for column in columns:
        series = panda_frame[column]
        anomaly = Test.percentile_based(series, lower_percentile, upper_percentile)
        rm_index.extend(anomaly)

    dict = {}
    for i in rm_index:
        if dict.has_key(i):
            dict[i] += 1
        else:
            dict[i] = 1

    if order == 'under':
        anomaly_index = [x for x in dict.keys() if dict[x] <= column_err_threshold]
    elif order == 'above':
        anomaly_index = [x for x in dict.keys() if dict[x] >= column_err_threshold]
    # anomaly_count = [dict[x] for x in dict.keys() if dict[x] > column_err_threshold]
    #
    #
    # plt.stem(anomaly_index, anomaly_count)
    # plt.legend(['index', 'count'], loc='upper left')
    # plt.title('Anomaly count')
    # plt.show()

    # Remove anomalies
    p_filtered = panda_frame.drop(panda_frame.index[anomaly_index])
    return p_filtered

def filterDataAutoEncoder(panda_frame, reconstruction_error, threshold):
    '''
    :param panda_frame: Input data frame
    :param reconstruction_error: Reconstruction error fromauto encoders
    :param threshold: Anomaly removal threshold
    :return:
    '''
    rm_index = []
    for i in range(len(reconstruction_error)):
        if reconstruction_error[i] > threshold:
            rm_index.append(i)


    p_filtered = panda_frame.drop(panda_frame.index[rm_index])
    return p_filtered