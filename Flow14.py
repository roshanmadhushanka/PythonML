import random
import matplotlib.pyplot as plt
import tkinter as tk
from featureeng import Math

index = range(0, 100)
lst = [9, 0, 9, 4, 0, 5, 3, 7, 0, 4, 8, 7, 7, 9, 0, 2, 2, 4, 7, 0, 8, 0, 0, 4, 3, 0, 8, 2, 9, 9, 1, 0, 4, 5, 0, 7, 7, 4, 5, 4, 0, 9, 3, 1, 8, 5, 1, 1, 7, 2, 3, 0, 9, 7, 0, 4, 9, 6, 9, 1, 2, 9, 2, 0, 1, 0, 3, 8, 5, 4, 0, 6, 4, 6, 6, 9, 1, 7, 7, 8, 5, 6, 1, 0, 6, 6, 9, 7, 9, 3, 3, 4, 6, 6, 6, 4, 5, 9, 1, 7]
moving_average = Math.moving_average(series=lst, window=5, default=True)
moving_threshold_average = Math.moving_threshold_average(series=lst, window=5, threshold=-1, default=True)
moving_k_closest_average = Math.moving_k_closest_average(series=lst, window=5, kclosest=3, default=True)
moving_median_centered_average = Math.moving_median_centered_average(series=lst, window=5, boundary=1, default=True)
moving_median = Math.moving_median(series=lst, window=5, default=True)

# plt.plot(index, lst)
# plt.plot(index, moving_average)
# plt.legend(['normal', 'moving_avg'], loc='upper left')

# plt.plot(index, lst)
# plt.plot(index, moving_threshold_average)
# plt.legend(['normal', 'moving_threshold_avg'], loc='upper left')

# plt.plot(index, lst)
# plt.plot(index, moving_k_closest_average)
# plt.legend(['normal', 'moving_k_closest_avg'], loc='upper left')

# plt.plot(index, lst)
# plt.plot(index, moving_median_centered_average)
# plt.legend(['normal', 'moving_median_centered'], loc='upper left')

# plt.plot(index, lst)
# plt.plot(index, moving_median)
# plt.legend(['normal', 'moving_median'], loc='upper left')

plt.show()
