import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def correlation(seriesX, seriesY, method='pearson'):
    '''
    :param seriesX:
    :param seriesY:
    :param method:
        pearson : standard correlation coefficient
        kendall : Kendall Tau correlation coefficient
        spearman : Spearman rank correlation
    :return: correlation
    '''
    _lenX = len(seriesX)
    _lenY = len(seriesY)

    if (_lenX <> _lenY):
        # If series are not in equal lengths, it is not possible to calculate correlation between series
        return

    _seriesX = pd.Series(seriesX)
    _seriesY = pd.Series(seriesY)

    return _seriesX.corr(_seriesY, method=method)

def r2(actual, predicted):
    return 1 - (mean_squared_error(actual, np.array(predicted)) / np.var(actual))

def confusion_matrix(actual, predict):
    if len(actual) <> len(predict):
        return

    tn = 0
    tp = 0
    fn = 0
    fp = 0

    for i in range(len(actual)):
        if actual[i] == u'"0"' and predict[i] == u'"0"':
            tn += 1
        elif actual[i] == u'"0"' and predict[i] == u'"1"':
            fp += 1
        elif actual[i] == u'"1"' and predict[i] == u'"0"':
            fn += 1
        elif actual[i] == u'"1"' and predict[i] == u'"1"':
            tp += 1


    print "TP", tp
    print "TN", tn
    print "FP", fp
    print "FN", fn

    print "Accuracy", 100.0 * (tp + tn) / (tp + tn + fp + fn), "%"

    recall = float(tp) / (tp + fn)
    precision = float(tp) / (tp + fp)
    f1 = 2*precision*recall / (precision + recall)

    print "Recall    :", recall
    print "Precision :", precision
    print "F1 Score  :", f1

