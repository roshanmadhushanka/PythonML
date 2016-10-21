import matplotlib.pyplot as plt
import numpy as np

def residual(actual, predicted):
    '''
    Calculate residual error

    :param actual: Actual data
    :param predicted: Predicted data
    :return: Residual error array
    '''
    if (len(actual) <> len(predicted)):
        return
    return np.array(actual) - np.array(predicted)

def residual_histogram(actual, predicted):
    res = residual(actual, predicted)
    p, x = np.histogram(res, bins=10)
    plt.plot(x[1:], p)
    plt.legend(['Residual Error'], loc='upper left')
    plt.title('Residual Error Histogram')
    plt.show()

def residual_vs_estimated(actual, predict):
    res = residual(actual, predict)
    plt.plot(predict, res, 'ro')
    plt.legend(['Residual Error'], loc='upper left')
    plt.title('Residual Error vs Predict')
    plt.show()

def acutal_and_predict(actual, predicted):
    if(len(actual) <> len(predicted)):
        return

    size = len(actual)
    index = range(size)

    plt.plot(index, actual)
    plt.plot(index, predicted)
    plt.legend(['Actual', 'Predicted'], loc='upper left')
    plt.title('Actual and Predict')
    plt.show()




