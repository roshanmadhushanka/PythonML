from anomaly import Test

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